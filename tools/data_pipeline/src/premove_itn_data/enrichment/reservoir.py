from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from premove_itn.labels import SpanKind
from premove_itn_data.enrichment.compile import compile_enrichment_candidate
from premove_itn_data.enrichment.donors import (
    EnrichmentDonor,
    Realize,
    generate_electronic_donors,
    generate_phone_donors,
    iter_google_donors,
    iter_rust_collision_donor_pairs,
)
from premove_itn_data.enrichment.generate import (
    EnrichmentCandidate,
    generate_collision_candidates,
    generate_purpose_built_candidate,
    generate_sgd_candidate,
    generate_sgd_context_only_candidate,
)
from premove_itn_data.enrichment.sgd_context import iter_sgd_train_user_turns

_RESERVOIR_FILE = "reservoir.jsonl"
_MANIFEST_FILE = "manifest.json"
_CHECKSUM_FILE = "reservoir.sha256"
_SGD_INPUT_PATTERN = re.compile(r"(?:schema|dialogues_[0-9]+)\.json")


@dataclass(frozen=True, slots=True)
class EnrichmentReservoirManifest:
    kind: str
    writer: str
    compiler: str
    schema_version: int
    seed: str
    source_split: str
    phone_donor_count: int
    electronic_donor_count: int
    google_donor_count: int
    google_donors_sha256: str
    sgd_train_sha256: str
    artifact_file: str
    artifact_sha256: str
    artifact_bytes: int
    output_records: int
    context_only_records: int
    multi_span_records: int
    records_by_category: dict[str, int]
    spans_by_kind: dict[str, int]


def _canonical_donor_bytes(donor: EnrichmentDonor) -> bytes:
    return json.dumps(
        {
            "kind": donor.kind.value,
            "provenance": donor.provenance,
            "replacement": donor.replacement,
            "spoken": donor.spoken,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _google_donors_sha256(donors: tuple[EnrichmentDonor, ...]) -> str:
    digest = hashlib.sha256()
    for donor in donors:
        digest.update(_canonical_donor_bytes(donor))
        digest.update(b"\n")
    return digest.hexdigest()


def _sgd_train_sha256(train_directory: Path) -> str:
    paths = tuple(
        sorted(
            path
            for path in train_directory.iterdir()
            if path.is_file() and _SGD_INPUT_PATTERN.fullmatch(path.name)
        )
    )
    has_schema = any(path.name == "schema.json" for path in paths)
    has_dialogues = any(path.name.startswith("dialogues_") for path in paths)
    if not has_schema or not has_dialogues:
        raise ValueError("SGD train inputs are incomplete")

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _donors_by_kind(
    donors: tuple[EnrichmentDonor, ...],
) -> dict[SpanKind, tuple[EnrichmentDonor, ...]]:
    return {
        kind: tuple(donor for donor in donors if donor.kind is kind)
        for kind in SpanKind
    }


def _iter_reservoir_candidates(
    train_directory: Path,
    google_donors: tuple[EnrichmentDonor, ...],
    phone_donors: tuple[EnrichmentDonor, ...],
    electronic_donors: tuple[EnrichmentDonor, ...],
    *,
    seed: str,
    realize: Realize,
) -> Iterator[tuple[str, EnrichmentCandidate]]:
    pools = _donors_by_kind(google_donors)
    digit_donors = pools[SpanKind.DIGIT_SEQUENCE]

    for turn in iter_sgd_train_user_turns(train_directory):
        positive = generate_sgd_candidate(turn, pools, seed=seed)
        if positive is not None:
            yield "sgd_positive", positive
        context = generate_sgd_context_only_candidate(turn)
        if context is not None:
            yield "sgd_context_only", context

    for category, donors in (
        ("purpose_phone", phone_donors),
        ("purpose_electronic", electronic_donors),
        ("purpose_digit_sequence", digit_donors),
    ):
        for index, donor in enumerate(donors):
            yield category, generate_purpose_built_candidate(
                donor,
                seed=seed,
                index=index,
            )

    for category, donors, first_kind in (
        (
            "collision_time_digit_sequence",
            (*pools[SpanKind.TIME], *digit_donors),
            SpanKind.TIME,
        ),
        (
            "collision_phone_digit_sequence",
            (*phone_donors, *digit_donors),
            SpanKind.PHONE,
        ),
    ):
        pairs = iter_rust_collision_donor_pairs(
            donors,
            first_kind=first_kind,
            second_kind=SpanKind.DIGIT_SEQUENCE,
            realize=realize,
        )
        for index, pair in enumerate(pairs):
            for candidate in generate_collision_candidates(
                *pair,
                seed=seed,
                index=index,
            ):
                yield category, candidate


def _reservoir_payload(
    category: str,
    candidate: EnrichmentCandidate,
) -> tuple[dict[str, object], int, bool]:
    record = compile_enrichment_candidate(candidate)
    provenance_spans = tuple(
        {
            "context": span.context_provenance,
            "donor": span.donor_provenance,
        }
        for span in candidate.spans
    )
    if len(provenance_spans) != len(record.spans):
        raise ValueError("record and provenance span counts must match")
    return (
        {
            "provenance": {
                "category": category,
                "context": candidate.context_provenance,
                "source": "enrichment",
                "spans": provenance_spans,
            },
            "record": asdict(record),
        },
        len(record.spans),
        len(record.spans) > 1,
    )


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


def _write_checksum_temporary(
    directory: Path,
    artifact_sha256: str,
) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="ascii",
        dir=directory,
        prefix=".reservoir.sha256.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(f"{artifact_sha256}  {_RESERVOIR_FILE}\n")
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _publish_no_overwrite(temporary_path: Path, final_path: Path) -> None:
    os.link(temporary_path, final_path)
    temporary_path.unlink()


def write_enrichment_reservoir(
    train_directory: Path,
    google_candidates_path: Path,
    output_directory: Path,
    *,
    seed: str,
    realize: Realize,
    phone_donor_count: int = 4_000,
    electronic_donor_count: int = 4_000,
) -> EnrichmentReservoirManifest:
    """Generate, compile, and atomically persist the enrichment reservoir."""
    if not seed:
        raise ValueError("seed must be non-empty")
    if phone_donor_count < 0:
        raise ValueError("phone_donor_count must be non-negative")
    if electronic_donor_count < 0:
        raise ValueError("electronic_donor_count must be non-negative")

    reservoir_path = output_directory / _RESERVOIR_FILE
    manifest_path = output_directory / _MANIFEST_FILE
    checksum_path = output_directory / _CHECKSUM_FILE
    for path in (reservoir_path, manifest_path, checksum_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

    google_donors = tuple(iter_google_donors(google_candidates_path))
    google_donors_hash = _google_donors_sha256(google_donors)
    sgd_train_hash = _sgd_train_sha256(train_directory)
    phone_donors = generate_phone_donors(
        phone_donor_count,
        seed=seed,
        realize=realize,
    )
    electronic_donors = generate_electronic_donors(
        electronic_donor_count,
        seed=seed,
        realize=realize,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    published_paths: list[Path] = []
    try:
        artifact_digest = hashlib.sha256()
        artifact_bytes = 0
        records_by_category: Counter[str] = Counter()
        spans_by_kind: Counter[str] = Counter()
        output_records = 0
        context_only_records = 0
        multi_span_records = 0

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_directory,
            prefix=".reservoir.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            reservoir_temporary_path = Path(handle.name)
            temporary_paths.append(reservoir_temporary_path)
            candidates = _iter_reservoir_candidates(
                train_directory,
                google_donors,
                phone_donors,
                electronic_donors,
                seed=seed,
                realize=realize,
            )
            for category, candidate in candidates:
                payload, span_count, is_multi_span = _reservoir_payload(
                    category,
                    candidate,
                )
                serialized = (
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
                handle.write(serialized)
                artifact_digest.update(serialized)
                artifact_bytes += len(serialized)
                output_records += 1
                records_by_category[category] += 1
                spans_by_kind.update(span.kind.value for span in candidate.spans)
                context_only_records += span_count == 0
                multi_span_records += is_multi_span
            handle.flush()
            os.fsync(handle.fileno())

        artifact_sha256 = artifact_digest.hexdigest()
        manifest = EnrichmentReservoirManifest(
            kind="enrichment_compiled_reservoir",
            writer="enrichment_reservoir_v1",
            compiler="training_record_v1",
            schema_version=1,
            seed=seed,
            source_split="train",
            phone_donor_count=phone_donor_count,
            electronic_donor_count=electronic_donor_count,
            google_donor_count=len(google_donors),
            google_donors_sha256=google_donors_hash,
            sgd_train_sha256=sgd_train_hash,
            artifact_file=_RESERVOIR_FILE,
            artifact_sha256=artifact_sha256,
            artifact_bytes=artifact_bytes,
            output_records=output_records,
            context_only_records=context_only_records,
            multi_span_records=multi_span_records,
            records_by_category=dict(sorted(records_by_category.items())),
            spans_by_kind=dict(sorted(spans_by_kind.items())),
        )
        manifest_temporary_path = _write_json_temporary(
            output_directory,
            "manifest",
            asdict(manifest),
        )
        temporary_paths.append(manifest_temporary_path)
        checksum_temporary_path = _write_checksum_temporary(
            output_directory,
            artifact_sha256,
        )
        temporary_paths.append(checksum_temporary_path)

        for temporary_path, final_path in (
            (reservoir_temporary_path, reservoir_path),
            (checksum_temporary_path, checksum_path),
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
        description="Write the immutable compiled enrichment reservoir."
    )
    parser.add_argument("train_directory", type=Path)
    parser.add_argument("google_candidates", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--phone-donors", type=int, default=4_000)
    parser.add_argument("--electronic-donors", type=int, default=4_000)
    args = parser.parse_args()

    from premove_itn import _rust

    manifest = write_enrichment_reservoir(
        args.train_directory,
        args.google_candidates,
        args.output_directory,
        seed=args.seed,
        realize=_rust.realize,
        phone_donor_count=args.phone_donors,
        electronic_donor_count=args.electronic_donors,
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
