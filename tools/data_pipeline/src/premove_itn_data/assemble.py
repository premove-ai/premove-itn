from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from premove_itn.labels import SpanKind
from premove_itn.types import WordToken
from premove_itn_data.records import (
    TrainingRecord,
    TrainingSpan,
    compile_training_record,
)

_CANDIDATES_FILE = "candidates.jsonl"
_MANIFEST_FILE = "manifest.json"
_CHECKSUM_FILE = "candidates.sha256"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SOURCE_PRIORITY = {"google_tn": 0, "enrichment": 1}


@dataclass(frozen=True, slots=True)
class DatasetV1Manifest:
    kind: str
    assembler: str
    schema_version: int
    source_preference: list[str]
    google_input_manifest: str
    google_input_selector: str
    google_input_schema_version: int
    google_input_records: int
    google_input_artifact_sha256: str
    enrichment_input_manifest: str
    enrichment_input_selector: str
    enrichment_input_schema_version: int
    enrichment_input_records: int
    enrichment_input_artifact_sha256: str
    pre_dedup_records: int
    within_google_duplicates: int
    within_enrichment_duplicates: int
    cross_source_duplicates: int
    google_retained_from_cross_source_duplicates: int
    enrichment_discarded_from_cross_source_duplicates: int
    records_with_duplicate_origins: int
    source_counts: dict[str, int]
    spans_by_kind: dict[str, int]
    context_only_records: int
    multi_span_records: int
    output_records: int
    output_artifact: str
    output_artifact_sha256: str
    output_artifact_bytes: int


@dataclass(frozen=True, slots=True)
class _ArtifactInput:
    source: str
    manifest_path: Path
    artifact_path: Path
    selector: str
    schema_version: int
    records: int
    artifact_sha256: str
    artifact_bytes: int | None


@dataclass(frozen=True, slots=True)
class _Candidate:
    source: str
    record: TrainingRecord
    provenance: dict[str, object]


@dataclass(slots=True)
class _MergedCandidate:
    primary: _Candidate
    duplicates: list[dict[str, object]]


def _require_mapping(value: object, *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    return value


def _require_string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _require_int(value: object, *, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{location} must be an integer")
    return value


def _load_json_mapping(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            return _require_mapping(json.load(handle), location=description)
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is invalid JSON: {path}") from error


def _checksum_hash(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"artifact checksum does not exist: {path}")
    fields = path.read_text(encoding="ascii").split()
    if not fields or not _SHA256_PATTERN.fullmatch(fields[0]):
        raise ValueError(f"invalid SHA-256 checksum file: {path}")
    return fields[0]


def _plain_file_name(value: object, *, location: str) -> str:
    name = _require_string(value, location=location)
    if Path(name).name != name:
        raise ValueError(f"{location} must be a plain file name")
    return name


def _load_google_input(manifest_path: Path) -> _ArtifactInput:
    manifest = _load_json_mapping(
        manifest_path, description="Google candidate manifest"
    )
    if manifest.get("kind") != "google_tn_selection_candidates":
        raise ValueError("input is not a Google TN candidate manifest")
    schema_version = _require_int(
        manifest.get("schema_version"), location="Google manifest.schema_version"
    )
    if schema_version != 1:
        raise ValueError("unsupported Google candidate schema_version")
    cleanup = _require_mapping(
        manifest.get("cleanup"), location="Google manifest.cleanup"
    )
    if cleanup.get("structural_validation") != "PASS":
        raise ValueError("Google candidate cleanup is not structurally valid")
    artifact_hash = _require_string(
        cleanup.get("output_candidates_sha256"),
        location="Google manifest.cleanup.output_candidates_sha256",
    )
    if not _SHA256_PATTERN.fullmatch(artifact_hash):
        raise ValueError("Google candidate manifest has an invalid SHA-256")
    checksum_hash = _checksum_hash(manifest_path.parent / _CHECKSUM_FILE)
    if checksum_hash != artifact_hash:
        raise ValueError("Google candidate checksum does not match manifest")
    artifact_path = manifest_path.parent / _CANDIDATES_FILE
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Google candidates do not exist: {artifact_path}")
    return _ArtifactInput(
        source="google_tn",
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        selector=_require_string(
            manifest.get("selector"), location="Google manifest.selector"
        ),
        schema_version=schema_version,
        records=_require_int(
            manifest.get("output_records"), location="Google manifest.output_records"
        ),
        artifact_sha256=artifact_hash,
        artifact_bytes=None,
    )


def _load_enrichment_input(manifest_path: Path) -> _ArtifactInput:
    manifest = _load_json_mapping(
        manifest_path, description="enrichment candidate manifest"
    )
    if manifest.get("kind") != "enrichment_selection_candidates":
        raise ValueError("input is not an enrichment selection manifest")
    schema_version = _require_int(
        manifest.get("schema_version"),
        location="enrichment manifest.schema_version",
    )
    if schema_version != 1:
        raise ValueError("unsupported enrichment candidate schema_version")
    artifact_name = _plain_file_name(
        manifest.get("output_artifact"),
        location="enrichment manifest.output_artifact",
    )
    artifact_hash = _require_string(
        manifest.get("output_artifact_sha256"),
        location="enrichment manifest.output_artifact_sha256",
    )
    if not _SHA256_PATTERN.fullmatch(artifact_hash):
        raise ValueError("enrichment candidate manifest has an invalid SHA-256")
    checksum_hash = _checksum_hash(manifest_path.parent / _CHECKSUM_FILE)
    if checksum_hash != artifact_hash:
        raise ValueError("enrichment candidate checksum does not match manifest")
    artifact_path = manifest_path.parent / artifact_name
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"selected enrichment candidates do not exist: {artifact_path}"
        )
    return _ArtifactInput(
        source="enrichment",
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        selector=_require_string(
            manifest.get("selector"), location="enrichment manifest.selector"
        ),
        schema_version=schema_version,
        records=_require_int(
            manifest.get("output_records"),
            location="enrichment manifest.output_records",
        ),
        artifact_sha256=artifact_hash,
        artifact_bytes=_require_int(
            manifest.get("output_artifact_bytes"),
            location="enrichment manifest.output_artifact_bytes",
        ),
    )


def _deserialize_record(value: object, *, location: str) -> TrainingRecord:
    payload = _require_mapping(value, location=location)
    raw_spans = payload.get("spans")
    raw_tokens = payload.get("tokens")
    raw_labels = payload.get("bio_labels")
    if not isinstance(raw_spans, list):
        raise ValueError(f"{location}.spans must be a JSON array")
    if not isinstance(raw_tokens, list):
        raise ValueError(f"{location}.tokens must be a JSON array")
    if not isinstance(raw_labels, list) or not all(
        isinstance(label, str) for label in raw_labels
    ):
        raise ValueError(f"{location}.bio_labels must be a string array")

    spans = tuple(
        TrainingSpan(
            kind=SpanKind(_require_string(span.get("kind"), location=f"{where}.kind")),
            start=_require_int(span.get("start"), location=f"{where}.start"),
            end=_require_int(span.get("end"), location=f"{where}.end"),
            source=_require_string(span.get("source"), location=f"{where}.source"),
            replacement=_require_string(
                span.get("replacement"), location=f"{where}.replacement"
            ),
        )
        for index, raw_span in enumerate(raw_spans)
        for where in (f"{location}.spans[{index}]",)
        for span in (_require_mapping(raw_span, location=where),)
    )
    tokens = tuple(
        WordToken(
            text=_require_string(token.get("text"), location=f"{where}.text"),
            start=_require_int(token.get("start"), location=f"{where}.start"),
            end=_require_int(token.get("end"), location=f"{where}.end"),
        )
        for index, raw_token in enumerate(raw_tokens)
        for where in (f"{location}.tokens[{index}]",)
        for token in (_require_mapping(raw_token, location=where),)
    )
    text = _require_string(payload.get("text"), location=f"{location}.text")
    expected_text = _require_string(
        payload.get("expected_text"), location=f"{location}.expected_text"
    )
    record = TrainingRecord(
        text=text,
        expected_text=expected_text,
        spans=spans,
        tokens=tokens,
        bio_labels=tuple(raw_labels),
    )
    if record != compile_training_record(text, spans, expected_text=expected_text):
        raise ValueError(f"{location} does not match shared compiler output")
    return record


def _deserialize_candidate(
    value: object,
    *,
    source: str,
    location: str,
) -> _Candidate:
    payload = _require_mapping(value, location=location)
    provenance = dict(
        _require_mapping(payload.get("provenance"), location=f"{location}.provenance")
    )
    if provenance.get("source") != source:
        raise ValueError(f"{location}.provenance.source must be {source!r}")
    if source == "google_tn":
        _require_string(
            provenance.get("source_file"),
            location=f"{location}.provenance.source_file",
        )
        sentence_number = _require_int(
            provenance.get("sentence_number"),
            location=f"{location}.provenance.sentence_number",
        )
        if sentence_number < 0:
            raise ValueError(
                f"{location}.provenance.sentence_number must be non-negative"
            )
    else:
        _require_string(
            provenance.get("category"),
            location=f"{location}.provenance.category",
        )
        _require_string(
            provenance.get("context"),
            location=f"{location}.provenance.context",
        )
        if not isinstance(provenance.get("spans"), list):
            raise ValueError(f"{location}.provenance.spans must be a JSON array")
    record = _deserialize_record(payload.get("record"), location=f"{location}.record")
    return _Candidate(source, record, provenance)


def _canonical_record(record: TrainingRecord) -> str:
    return json.dumps(
        asdict(record),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _record_identity(record: TrainingRecord) -> str:
    return hashlib.sha256(_canonical_record(record).encode("utf-8")).hexdigest()


def _canonical_provenance(provenance: Mapping[str, object]) -> str:
    return json.dumps(
        provenance,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _candidate_preference(candidate: _Candidate) -> tuple[int, str]:
    return _SOURCE_PRIORITY[candidate.source], _canonical_provenance(
        candidate.provenance
    )


def _iter_artifact_candidates(artifact: _ArtifactInput):
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    with artifact.artifact_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            byte_count += len(line)
            try:
                raw_payload = json.loads(line)
                candidate = _deserialize_candidate(
                    raw_payload,
                    source=artifact.source,
                    location=f"{artifact.artifact_path}:{line_number}",
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid {artifact.source} candidate at "
                    f"{artifact.artifact_path}:{line_number}"
                ) from error
            record_count += 1
            yield candidate

    if digest.hexdigest() != artifact.artifact_sha256:
        raise ValueError(f"{artifact.source} artifact SHA-256 does not match manifest")
    if artifact.artifact_bytes is not None and byte_count != artifact.artifact_bytes:
        raise ValueError(
            f"{artifact.source} artifact byte count does not match manifest"
        )
    if record_count != artifact.records:
        raise ValueError(f"{artifact.source} record count does not match manifest")


def _duplicate_provenance(candidate: _Candidate) -> dict[str, object]:
    return dict(candidate.provenance)


def _offer_candidate(
    merged: dict[str, _MergedCandidate],
    candidate: _Candidate,
    *,
    duplicate_counts: Counter[str],
) -> None:
    identity = _record_identity(candidate.record)
    current = merged.get(identity)
    if current is None:
        merged[identity] = _MergedCandidate(candidate, [])
        return

    if current.primary.source == candidate.source:
        duplicate_counts[f"within_{candidate.source}"] += 1
    else:
        duplicate_counts["cross_source"] += 1

    if _candidate_preference(candidate) < _candidate_preference(current.primary):
        current.duplicates.append(_duplicate_provenance(current.primary))
        current.primary = candidate
    else:
        current.duplicates.append(_duplicate_provenance(candidate))


def _output_payload(candidate: _MergedCandidate) -> dict[str, object]:
    duplicate_winners: dict[str, dict[str, object]] = {}
    for provenance in candidate.duplicates:
        duplicate_winners.setdefault(_canonical_provenance(provenance), provenance)
    duplicates = tuple(
        duplicate_winners[key]
        for key in sorted(duplicate_winners)
        if key != _canonical_provenance(candidate.primary.provenance)
    )
    return {
        "provenance": {
            "duplicates": duplicates,
            "primary": candidate.primary.provenance,
        },
        "record": asdict(candidate.primary.record),
    }


def _publish_no_overwrite(temporary_path: Path, final_path: Path) -> None:
    os.link(temporary_path, final_path)
    temporary_path.unlink()


def assemble_dataset_v1(
    google_manifest_path: Path,
    enrichment_manifest_path: Path,
    output_directory: Path,
) -> DatasetV1Manifest:
    """Validate, deduplicate, and atomically persist Dataset V1 candidates."""
    candidates_path = output_directory / _CANDIDATES_FILE
    manifest_path = output_directory / _MANIFEST_FILE
    checksum_path = output_directory / _CHECKSUM_FILE
    for path in (candidates_path, manifest_path, checksum_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

    google = _load_google_input(google_manifest_path)
    enrichment = _load_enrichment_input(enrichment_manifest_path)
    merged: dict[str, _MergedCandidate] = {}
    duplicate_counts: Counter[str] = Counter()
    for artifact in (google, enrichment):
        for candidate in _iter_artifact_candidates(artifact):
            _offer_candidate(
                merged,
                candidate,
                duplicate_counts=duplicate_counts,
            )

    selected = tuple(merged[identity] for identity in sorted(merged))
    source_counts = Counter(item.primary.source for item in selected)
    spans_by_kind = Counter(
        span.kind.value for item in selected for span in item.primary.record.spans
    )
    cross_source_duplicates = duplicate_counts["cross_source"]

    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    published_paths: list[Path] = []
    try:
        output_digest = hashlib.sha256()
        output_bytes = 0
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_directory,
            prefix=".candidates.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            candidates_temporary = Path(handle.name)
            temporary_paths.append(candidates_temporary)
            for item in selected:
                serialized = (
                    json.dumps(
                        _output_payload(item),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
                handle.write(serialized)
                output_digest.update(serialized)
                output_bytes += len(serialized)
            handle.flush()
            os.fsync(handle.fileno())

        output_hash = output_digest.hexdigest()
        manifest = DatasetV1Manifest(
            kind="dataset_v1_candidates",
            assembler="dataset_v1_assembly_v1",
            schema_version=1,
            source_preference=["google_tn", "enrichment"],
            google_input_manifest=google.manifest_path.name,
            google_input_selector=google.selector,
            google_input_schema_version=google.schema_version,
            google_input_records=google.records,
            google_input_artifact_sha256=google.artifact_sha256,
            enrichment_input_manifest=enrichment.manifest_path.name,
            enrichment_input_selector=enrichment.selector,
            enrichment_input_schema_version=enrichment.schema_version,
            enrichment_input_records=enrichment.records,
            enrichment_input_artifact_sha256=enrichment.artifact_sha256,
            pre_dedup_records=google.records + enrichment.records,
            within_google_duplicates=duplicate_counts["within_google_tn"],
            within_enrichment_duplicates=duplicate_counts["within_enrichment"],
            cross_source_duplicates=cross_source_duplicates,
            google_retained_from_cross_source_duplicates=cross_source_duplicates,
            enrichment_discarded_from_cross_source_duplicates=cross_source_duplicates,
            records_with_duplicate_origins=sum(
                bool(item.duplicates) for item in selected
            ),
            source_counts=dict(sorted(source_counts.items())),
            spans_by_kind=dict(sorted(spans_by_kind.items())),
            context_only_records=sum(
                not item.primary.record.spans for item in selected
            ),
            multi_span_records=sum(
                len(item.primary.record.spans) > 1 for item in selected
            ),
            output_records=len(selected),
            output_artifact=_CANDIDATES_FILE,
            output_artifact_sha256=output_hash,
            output_artifact_bytes=output_bytes,
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_directory,
            prefix=".manifest.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            manifest_temporary = Path(handle.name)
            temporary_paths.append(manifest_temporary)
            json.dump(asdict(manifest), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            dir=output_directory,
            prefix=".candidates.sha256.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            checksum_temporary = Path(handle.name)
            temporary_paths.append(checksum_temporary)
            handle.write(f"{output_hash}  {_CANDIDATES_FILE}\n")
            handle.flush()
            os.fsync(handle.fileno())

        for temporary_path, final_path in (
            (candidates_temporary, candidates_path),
            (checksum_temporary, checksum_path),
            (manifest_temporary, manifest_path),
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
        description="Assemble the frozen Dataset V1 candidate artifact."
    )
    parser.add_argument("google_manifest", type=Path)
    parser.add_argument("enrichment_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    manifest = assemble_dataset_v1(
        args.google_manifest,
        args.enrichment_manifest,
        args.output_directory,
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
