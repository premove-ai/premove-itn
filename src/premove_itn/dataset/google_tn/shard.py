import argparse
import json
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

from premove_itn.dataset.google_tn.audit import audit_google_tn_outcomes
from premove_itn.dataset.google_tn.parser import GoogleTnSentence, read_google_sentences
from premove_itn.dataset.google_tn.pipeline import (
    GoogleTnSentenceOutcome,
    iter_google_tn_outcomes,
)
from premove_itn.dataset.google_tn.validation import GoogleTnRealizer


@dataclass(frozen=True, slots=True)
class GoogleTnShardManifest:
    kind: str
    compiler: str
    schema_version: int
    source_id: str
    source_start: int
    source_end_exclusive: int
    processed: int
    accepted: int
    rejected: int
    output_records: int


def _publish_no_overwrite(temporary_path: Path, final_path: Path) -> None:
    os.link(temporary_path, final_path)
    temporary_path.unlink()


def _write_json_temporary(directory: Path, stem: str, value: object) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        prefix=f".{stem}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def write_google_tn_accepted_shard(
    sentences: Iterable[GoogleTnSentence],
    realizer: GoogleTnRealizer,
    output_directory: Path,
    *,
    source_id: str,
    source_start: int,
    limit: int,
) -> GoogleTnShardManifest:
    if source_start < 0:
        raise ValueError("source_start must be non-negative")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", source_id):
        raise ValueError(
            "source_id must contain only letters, digits, periods, underscores, "
            "and hyphens, and must start with a letter or digit"
        )

    selected = islice(sentences, source_start, source_start + limit)
    accepted_directory = output_directory / "accepted"
    audit_directory = output_directory / "audits"
    accepted_directory.mkdir(parents=True, exist_ok=True)
    audit_directory.mkdir(parents=True, exist_ok=True)
    requested_stem = f"{source_id}_{source_start:06d}_{source_start + limit:06d}"

    temporary_paths: list[Path] = []
    published_paths: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=accepted_directory,
            prefix=f".{requested_stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            shard_temporary_path = Path(handle.name)
            temporary_paths.append(shard_temporary_path)

            def write_accepted(outcome: GoogleTnSentenceOutcome) -> None:
                if outcome.record is None:
                    return
                payload = {
                    "record": asdict(outcome.record),
                    "provenance": {
                        "source": "google_tn",
                        "source_file": outcome.sentence.source_name,
                        "sentence_number": outcome.sentence.sentence_number,
                    },
                }
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")

            report = audit_google_tn_outcomes(
                iter_google_tn_outcomes(selected, realizer), observer=write_accepted
            )
            handle.flush()
            os.fsync(handle.fileno())

        manifest = GoogleTnShardManifest(
            "google_tn_accepted_corpus_shard",
            "google_tn_v1",
            1,
            source_id,
            source_start,
            source_start + report.sentences_processed,
            report.sentences_processed,
            report.accepted_sentences,
            report.rejected_sentences,
            report.accepted_sentences,
        )
        stem = f"{source_id}_{source_start:06d}_{manifest.source_end_exclusive:06d}"
        shard_path = accepted_directory / f"{stem}.jsonl"
        manifest_path = audit_directory / f"{stem}_manifest.json"
        summary_path = audit_directory / f"{stem}_summary.json"
        for path in (shard_path, summary_path, manifest_path):
            if path.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing artifact: {path}"
                )
        manifest_temporary_path = _write_json_temporary(
            audit_directory, f"{stem}_manifest", asdict(manifest)
        )
        temporary_paths.append(manifest_temporary_path)
        summary_temporary_path = _write_json_temporary(
            audit_directory, f"{stem}_summary", asdict(report)
        )
        temporary_paths.append(summary_temporary_path)

        for temporary_path, final_path in (
            (shard_temporary_path, shard_path),
            (summary_temporary_path, summary_path),
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
        description="Materialize one immutable Google TN accepted-corpus shard."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--source-start", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--source-id")
    args = parser.parse_args()

    from premove_itn import _rust

    manifest = write_google_tn_accepted_shard(
        read_google_sentences(args.input),
        _rust.realize,
        args.output_directory,
        source_id=args.source_id or args.input.stem,
        source_start=args.source_start,
        limit=args.limit,
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
