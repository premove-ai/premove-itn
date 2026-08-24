from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from premove_itn.dataset.records import TrainingRecord, TrainingSpan
from premove_itn.dataset.selection import (
    DatasetSelectionRecord,
    SelectionQuotas,
    SelectionResult,
    select_candidates,
)
from premove_itn.labels import SpanKind
from premove_itn.types import WordToken

GOOGLE_TN_V1_CONTRIBUTION_QUOTAS = SelectionQuotas(
    {
        SpanKind.DATE: 5_000,
        SpanKind.CARDINAL: 5_000,
        SpanKind.MEASUREMENT: 3_000,
        SpanKind.ORDINAL: 2_500,
        SpanKind.DECIMAL: 3_000,
        SpanKind.MONEY: 4_000,
        SpanKind.TIME: 5_000,
        SpanKind.DIGIT_SEQUENCE: 5_000,
        SpanKind.PHONE: 500,
        SpanKind.ELECTRONIC: 23,
    },
    context_only_records=8_000,
)

_SOURCE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True, slots=True)
class GoogleTnCandidateManifest:
    kind: str
    selector: str
    schema_version: int
    input_manifest: str
    input_compiler: str
    input_schema_version: int
    source_id: str
    input_shards: int
    input_records: int
    seed: str
    span_targets: dict[str, int]
    context_only_target: int
    records_scanned: int
    retained_unique_candidates: int
    duplicates_removed: int
    selected_record_count: int
    selected_span_count: int
    selected_context_only_count: int
    actual_by_kind: dict[str, int]
    shortfall_by_kind: dict[str, int]
    overshoot_by_kind: dict[str, int]
    multi_span_selected: int
    source_counts: dict[str, int]
    output_records: int


@dataclass(frozen=True, slots=True)
class _CorpusInput:
    manifest_name: str
    compiler: str
    schema_version: int
    source_id: str
    records: int
    shard_paths: tuple[tuple[Path, int], ...]


def _require_mapping(value: object, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _require_int(value: object, *, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{description} must be an integer")
    return value


def _load_corpus_input(manifest_path: Path) -> _CorpusInput:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = _require_mapping(
            json.load(handle), description="Google TN corpus manifest"
        )

    if manifest.get("kind") != "google_tn_accepted_corpus":
        raise ValueError("input is not a Google TN accepted-corpus manifest")
    compiler = manifest.get("compiler")
    if not isinstance(compiler, str) or not compiler:
        raise ValueError("corpus compiler must be a non-empty string")
    schema_version = _require_int(
        manifest.get("schema_version"), description="corpus schema_version"
    )
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("corpus shards must be a JSON array")
    shard_count = _require_int(
        manifest.get("shard_count"), description="corpus shard_count"
    )
    if shard_count != len(raw_shards):
        raise ValueError("corpus shard_count does not match shards")

    accepted_directory = manifest_path.parent.parent / "accepted"
    shard_paths: list[tuple[Path, int]] = []
    source_id: str | None = None
    previous_end: int | None = None
    total_records = 0
    for index, raw_shard in enumerate(raw_shards):
        shard = _require_mapping(raw_shard, description=f"corpus shard {index}")
        if shard.get("kind") != "google_tn_accepted_corpus_shard":
            raise ValueError(f"corpus shard {index} has an unexpected kind")
        if shard.get("compiler") != compiler:
            raise ValueError(f"corpus shard {index} has a different compiler")
        if shard.get("schema_version") != schema_version:
            raise ValueError(f"corpus shard {index} has a different schema version")

        shard_source_id = shard.get("source_id")
        if not isinstance(shard_source_id, str) or not _SOURCE_ID_PATTERN.fullmatch(
            shard_source_id
        ):
            raise ValueError(f"corpus shard {index} has an unsafe source_id")
        if source_id is None:
            source_id = shard_source_id
        elif shard_source_id != source_id:
            raise ValueError("corpus shards must have one source_id")

        source_start = _require_int(
            shard.get("source_start"),
            description=f"corpus shard {index} source_start",
        )
        source_end = _require_int(
            shard.get("source_end_exclusive"),
            description=f"corpus shard {index} source_end_exclusive",
        )
        output_records = _require_int(
            shard.get("output_records"),
            description=f"corpus shard {index} output_records",
        )
        if source_start < 0 or source_end <= source_start:
            raise ValueError(f"corpus shard {index} has an invalid source range")
        if previous_end is not None and source_start != previous_end:
            raise ValueError("corpus shards are not in contiguous source order")
        if output_records < 0:
            raise ValueError(f"corpus shard {index} has negative output_records")
        previous_end = source_end

        stem = f"{shard_source_id}_{source_start:06d}_{source_end:06d}"
        shard_path = accepted_directory / f"{stem}.jsonl"
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"accepted shard referenced by manifest does not exist: {shard_path}"
            )
        shard_paths.append((shard_path, output_records))
        total_records += output_records

    if source_id is None:
        raise ValueError("corpus manifest must contain at least one shard")
    manifest_records = _require_int(
        manifest.get("output_records"), description="corpus output_records"
    )
    if manifest_records != total_records:
        raise ValueError("corpus output_records does not match shard totals")

    return _CorpusInput(
        manifest_path.name,
        compiler,
        schema_version,
        source_id,
        total_records,
        tuple(shard_paths),
    )


def _deserialize_record(payload: Mapping[str, Any]) -> DatasetSelectionRecord:
    raw_record = _require_mapping(payload["record"], description="record")
    raw_provenance = _require_mapping(payload["provenance"], description="provenance")
    raw_spans = raw_record["spans"]
    raw_tokens = raw_record["tokens"]
    raw_labels = raw_record["bio_labels"]
    if not isinstance(raw_spans, list):
        raise ValueError("record spans must be a JSON array")
    if not isinstance(raw_tokens, list):
        raise ValueError("record tokens must be a JSON array")
    if not isinstance(raw_labels, list) or not all(
        isinstance(label, str) for label in raw_labels
    ):
        raise ValueError("record bio_labels must be a string array")

    spans = tuple(
        TrainingSpan(
            kind=SpanKind(span["kind"]),
            start=span["start"],
            end=span["end"],
            source=span["source"],
            replacement=span["replacement"],
        )
        for raw_span in raw_spans
        for span in (_require_mapping(raw_span, description="training span"),)
    )
    tokens = tuple(
        WordToken(text=token["text"], start=token["start"], end=token["end"])
        for raw_token in raw_tokens
        for token in (_require_mapping(raw_token, description="word token"),)
    )
    record = TrainingRecord(
        text=raw_record["text"],
        expected_text=raw_record["expected_text"],
        spans=spans,
        tokens=tokens,
        bio_labels=tuple(raw_labels),
    )
    source = raw_provenance["source"]
    if source != "google_tn":
        raise ValueError("accepted Google TN provenance source must be google_tn")
    return DatasetSelectionRecord(
        record=record,
        source=source,
        source_file=raw_provenance["source_file"],
        sentence_number=raw_provenance["sentence_number"],
    )


def _iter_accepted_records(corpus: _CorpusInput) -> Iterator[DatasetSelectionRecord]:
    for shard_path, expected_records in corpus.shard_paths:
        records_read = 0
        with shard_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    raw_payload = json.loads(line)
                    payload = _require_mapping(
                        raw_payload, description="accepted corpus line"
                    )
                    record = _deserialize_record(payload)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid accepted record at {shard_path}:{line_number}"
                    ) from error
                records_read += 1
                yield record
        if records_read != expected_records:
            raise ValueError(
                f"accepted shard line count does not match manifest: {shard_path}"
            )


def _candidate_payload(candidate: DatasetSelectionRecord) -> dict[str, object]:
    return {
        "record": asdict(candidate.record),
        "provenance": {
            "source": candidate.source,
            "source_file": candidate.source_file,
            "sentence_number": candidate.sentence_number,
        },
    }


def _build_manifest(
    corpus: _CorpusInput,
    result: SelectionResult,
    quotas: SelectionQuotas,
    seed: str,
) -> GoogleTnCandidateManifest:
    return GoogleTnCandidateManifest(
        kind="google_tn_selection_candidates",
        selector="deterministic_reservoir_v1",
        schema_version=1,
        input_manifest=corpus.manifest_name,
        input_compiler=corpus.compiler,
        input_schema_version=corpus.schema_version,
        source_id=corpus.source_id,
        input_shards=len(corpus.shard_paths),
        input_records=corpus.records,
        seed=seed,
        span_targets=result.target_by_kind,
        context_only_target=quotas.context_only_records,
        records_scanned=result.records_scanned,
        retained_unique_candidates=result.retained_unique_candidates,
        duplicates_removed=result.duplicates_removed,
        selected_record_count=result.selected_record_count,
        selected_span_count=result.selected_span_count,
        selected_context_only_count=result.selected_context_only_count,
        actual_by_kind=result.actual_by_kind,
        shortfall_by_kind=result.shortfall_by_kind,
        overshoot_by_kind=result.overshoot_by_kind,
        multi_span_selected=result.multi_span_selected,
        source_counts=result.source_counts,
        output_records=result.selected_record_count,
    )


def _publish_no_overwrite(temporary_path: Path, final_path: Path) -> None:
    os.link(temporary_path, final_path)
    temporary_path.unlink()


def write_google_tn_candidates(
    corpus_manifest_path: Path,
    output_directory: Path,
    *,
    seed: str,
    quotas: SelectionQuotas = GOOGLE_TN_V1_CONTRIBUTION_QUOTAS,
) -> GoogleTnCandidateManifest:
    """Select and atomically persist Google's deterministic V1 contribution."""
    candidates_path = output_directory / "candidates.jsonl"
    manifest_path = output_directory / "manifest.json"
    for path in (candidates_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

    corpus = _load_corpus_input(corpus_manifest_path)
    result = select_candidates(_iter_accepted_records(corpus), quotas, seed=seed)
    if result.records_scanned != corpus.records:
        raise ValueError("records scanned does not match corpus manifest")
    manifest = _build_manifest(corpus, result, quotas, seed)

    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    published_paths: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_directory,
            prefix=".candidates.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            candidates_temporary_path = Path(handle.name)
            temporary_paths.append(candidates_temporary_path)
            for candidate in result.selected_records:
                json.dump(
                    _candidate_payload(candidate),
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_directory,
            prefix=".manifest.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            manifest_temporary_path = Path(handle.name)
            temporary_paths.append(manifest_temporary_path)
            json.dump(asdict(manifest), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        for temporary_path, final_path in (
            (candidates_temporary_path, candidates_path),
            (manifest_temporary_path, manifest_path),
        ):
            _publish_no_overwrite(temporary_path, final_path)
            temporary_paths.remove(temporary_path)
            published_paths.append(final_path)
        return manifest
    except BaseException:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        for path in published_paths:
            path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select Google's deterministic contribution to dataset V1."
    )
    parser.add_argument("corpus_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args()

    manifest = write_google_tn_candidates(
        args.corpus_manifest,
        args.output_directory,
        seed=args.seed,
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
