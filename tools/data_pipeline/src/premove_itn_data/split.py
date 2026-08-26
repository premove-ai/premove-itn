from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from premove_itn_data.assemble import _deserialize_record

_TRAIN_FILE = "train.jsonl"
_VALIDATION_FILE = "validation.jsonl"
_MANIFEST_FILE = "manifest.json"
_TRAIN_CHECKSUM_FILE = "train.sha256"
_VALIDATION_CHECKSUM_FILE = "validation.sha256"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class SplitStatistics:
    records: int
    groups: int
    source_counts: dict[str, int]
    spans_by_kind: dict[str, int]
    context_only_records: int
    multi_span_records: int


@dataclass(frozen=True, slots=True)
class DatasetV1SplitManifest:
    kind: str
    splitter: str
    schema_version: int
    seed: str
    validation_fraction: float
    input_manifest: str
    input_assembler: str
    input_schema_version: int
    input_records: int
    input_artifact_sha256: str
    input_artifact_bytes: int
    grouping_rules: list[str]
    total_groups: int
    largest_group_records: int
    train: SplitStatistics
    validation: SplitStatistics
    group_leakage: int
    exact_record_leakage: int
    collision_groups_split: int
    donor_value_groups_split: int
    train_artifact: str
    train_artifact_sha256: str
    train_artifact_bytes: int
    validation_artifact: str
    validation_artifact_sha256: str
    validation_artifact_bytes: int


@dataclass(frozen=True, slots=True)
class _DatasetInput:
    manifest_path: Path
    artifact_path: Path
    assembler: str
    schema_version: int
    records: int
    artifact_sha256: str
    artifact_bytes: int


@dataclass(frozen=True, slots=True)
class _Record:
    payload: dict[str, object]
    identity: str
    source: str
    group_keys: tuple[str, ...]
    span_kinds: tuple[str, ...]
    context_only: bool
    multi_span: bool


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


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


def _load_input(manifest_path: Path) -> _DatasetInput:
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = _require_mapping(json.load(handle), location="manifest")
    except json.JSONDecodeError as error:
        raise ValueError("Dataset V1 manifest is invalid JSON") from error
    if manifest.get("kind") != "dataset_v1_candidates":
        raise ValueError("input is not a Dataset V1 candidate manifest")
    schema_version = _require_int(
        manifest.get("schema_version"), location="manifest.schema_version"
    )
    if schema_version != 1:
        raise ValueError("unsupported Dataset V1 schema_version")
    artifact_name = _require_string(
        manifest.get("output_artifact"), location="manifest.output_artifact"
    )
    if Path(artifact_name).name != artifact_name:
        raise ValueError("manifest.output_artifact must be a plain file name")
    artifact_hash = _require_string(
        manifest.get("output_artifact_sha256"),
        location="manifest.output_artifact_sha256",
    )
    if not _SHA256_PATTERN.fullmatch(artifact_hash):
        raise ValueError("manifest has an invalid artifact SHA-256")
    checksum_path = manifest_path.parent / "candidates.sha256"
    fields = checksum_path.read_text(encoding="ascii").split()
    if not fields or fields[0] != artifact_hash:
        raise ValueError("Dataset V1 checksum does not match manifest")
    artifact_path = manifest_path.parent / artifact_name
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Dataset V1 candidates do not exist: {artifact_path}")
    return _DatasetInput(
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        assembler=_require_string(
            manifest.get("assembler"), location="manifest.assembler"
        ),
        schema_version=schema_version,
        records=_require_int(
            manifest.get("output_records"), location="manifest.output_records"
        ),
        artifact_sha256=artifact_hash,
        artifact_bytes=_require_int(
            manifest.get("output_artifact_bytes"),
            location="manifest.output_artifact_bytes",
        ),
    )


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _normalized_donor(donor: str) -> str:
    if not donor.startswith("rust_collision/"):
        return donor
    body = donor.removeprefix("rust_collision/")
    for suffix in ("/phone", "/time", "/digit_sequence"):
        if body.endswith(suffix):
            return body[: -len(suffix)]
    return body


def _enrichment_group_keys(
    provenance: Mapping[str, Any], record: Mapping[str, Any], *, location: str
) -> tuple[str, ...]:
    category = _require_string(
        provenance.get("category"), location=f"{location}.category"
    )
    context = _require_string(provenance.get("context"), location=f"{location}.context")
    if category.startswith("sgd_"):
        try:
            dialogue_context, _ = context.rsplit("/", 1)
        except ValueError as error:
            raise ValueError(f"invalid SGD context: {context}") from error
        return (f"sgd_dialogue:{dialogue_context}",)

    keys: set[str] = set()
    if category.startswith("collision_"):
        parts = context.split("/")
        if len(parts) < 3 or parts[:2] != ["purpose_built", "collision"]:
            raise ValueError(f"invalid collision context: {context}")
        keys.add(f"collision:{parts[2]}")

    raw_span_provenance = provenance.get("spans")
    if not isinstance(raw_span_provenance, list):
        raise ValueError(f"{location}.spans must be a JSON array")
    raw_spans = record.get("spans")
    if not isinstance(raw_spans, list):
        raise ValueError(f"{location}.record.spans must be a JSON array")
    for index, raw_provenance in enumerate(raw_span_provenance):
        span_provenance = _require_mapping(
            raw_provenance, location=f"{location}.spans[{index}]"
        )
        donor = _require_string(
            span_provenance.get("donor"),
            location=f"{location}.spans[{index}].donor",
        )
        keys.add(f"donor:{_normalized_donor(donor)}")
    for index, raw_span in enumerate(raw_spans):
        span = _require_mapping(raw_span, location=f"{location}.record.spans[{index}]")
        source = _require_string(
            span.get("source"), location=f"{location}.record.spans[{index}].source"
        )
        keys.add(f"generated_value:{source.casefold()}")
    if not keys:
        keys.add(f"enrichment_context:{context}")
    return tuple(sorted(keys))


def _origin_group_keys(
    provenance: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    location: str,
) -> tuple[str, ...]:
    source = _require_string(provenance.get("source"), location=f"{location}.source")
    if source == "google_tn":
        source_file = _require_string(
            provenance.get("source_file"), location=f"{location}.source_file"
        )
        sentence = _require_int(
            provenance.get("sentence_number"),
            location=f"{location}.sentence_number",
        )
        if sentence < 0:
            raise ValueError(f"{location}.sentence_number must be non-negative")
        return (f"google_sentence:{source_file}:{sentence}",)
    if source == "enrichment":
        return _enrichment_group_keys(provenance, record, location=location)
    raise ValueError(f"{location} has unsupported source {source!r}")


def _deserialize_payload(value: object, *, location: str) -> _Record:
    payload = dict(_require_mapping(value, location=location))
    provenance_envelope = _require_mapping(
        payload.get("provenance"), location=f"{location}.provenance"
    )
    primary = _require_mapping(
        provenance_envelope.get("primary"),
        location=f"{location}.provenance.primary",
    )
    source = _require_string(
        primary.get("source"), location=f"{location}.provenance.primary.source"
    )
    record_payload = _require_mapping(
        payload.get("record"), location=f"{location}.record"
    )
    record = _deserialize_record(record_payload, location=f"{location}.record")
    canonical_record = _canonical(asdict(record))
    identity = hashlib.sha256(canonical_record.encode("utf-8")).hexdigest()
    raw_duplicates = provenance_envelope.get("duplicates")
    if not isinstance(raw_duplicates, list):
        raise ValueError(f"{location}.provenance.duplicates must be a JSON array")
    origins = (primary,) + tuple(
        _require_mapping(
            duplicate,
            location=f"{location}.provenance.duplicates[{index}]",
        )
        for index, duplicate in enumerate(raw_duplicates)
    )
    group_keys = tuple(
        sorted(
            {
                key
                for index, origin in enumerate(origins)
                for key in _origin_group_keys(
                    origin,
                    record_payload,
                    location=(
                        f"{location}.provenance.primary"
                        if index == 0
                        else f"{location}.provenance.duplicates[{index - 1}]"
                    ),
                )
            }
        )
    )
    return _Record(
        payload=payload,
        identity=identity,
        source=source,
        group_keys=group_keys,
        span_kinds=tuple(span.kind.value for span in record.spans),
        context_only=not record.spans,
        multi_span=len(record.spans) > 1,
    )


def _read_records(source: _DatasetInput) -> tuple[_Record, ...]:
    digest = hashlib.sha256()
    byte_count = 0
    records: list[_Record] = []
    identities: set[str] = set()
    with source.artifact_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            byte_count += len(line)
            try:
                item = _deserialize_payload(
                    json.loads(line),
                    location=f"{source.artifact_path}:{line_number}",
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid Dataset V1 record at {source.artifact_path}:{line_number}"
                ) from error
            if item.identity in identities:
                raise ValueError("Dataset V1 contains duplicate compiled records")
            identities.add(item.identity)
            records.append(item)
    if digest.hexdigest() != source.artifact_sha256:
        raise ValueError("Dataset V1 artifact SHA-256 does not match manifest")
    if byte_count != source.artifact_bytes:
        raise ValueError("Dataset V1 artifact byte count does not match manifest")
    if len(records) != source.records:
        raise ValueError("Dataset V1 record count does not match manifest")
    return tuple(records)


def _connected_groups(records: Sequence[_Record]) -> tuple[tuple[int, ...], ...]:
    disjoint = _DisjointSet(len(records))
    owner_by_key: dict[str, int] = {}
    for index, record in enumerate(records):
        for key in record.group_keys:
            owner = owner_by_key.setdefault(key, index)
            disjoint.union(index, owner)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[disjoint.find(index)].append(index)
    return tuple(
        sorted(
            (tuple(values) for values in grouped.values()),
            key=lambda group: min(records[index].identity for index in group),
        )
    )


def _group_rank(seed: str, group: Sequence[int], records: Sequence[_Record]) -> int:
    identity = min(records[index].identity for index in group)
    return int.from_bytes(
        hashlib.sha256(f"dataset-v1-split-v1\0{seed}\0{identity}".encode()).digest(),
        "big",
    )


def _select_validation_groups(
    groups: Sequence[tuple[int, ...]],
    records: Sequence[_Record],
    *,
    seed: str,
    validation_fraction: float,
) -> frozenset[int]:
    target = round(len(records) * validation_fraction)
    ranked = sorted(
        enumerate(groups),
        key=lambda item: (_group_rank(seed, item[1], records), item[0]),
    )
    selected: set[int] = set()
    count = 0
    for group_index, group in ranked:
        size = len(group)
        if abs(target - (count + size)) <= abs(target - count):
            selected.add(group_index)
            count += size
    return frozenset(selected)


def _statistics(records: Sequence[_Record], group_count: int) -> SplitStatistics:
    return SplitStatistics(
        records=len(records),
        groups=group_count,
        source_counts=dict(sorted(Counter(item.source for item in records).items())),
        spans_by_kind=dict(
            sorted(
                Counter(kind for item in records for kind in item.span_kinds).items()
            )
        ),
        context_only_records=sum(item.context_only for item in records),
        multi_span_records=sum(item.multi_span for item in records),
    )


def _write_artifact(directory: Path, name: str, records: Sequence[_Record]):
    digest = hashlib.sha256()
    byte_count = 0
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=directory, prefix=f".{name}.", suffix=".tmp", delete=False
    ) as handle:
        path = Path(handle.name)
        for item in sorted(records, key=lambda value: value.identity):
            serialized = (_canonical(item.payload) + "\n").encode("utf-8")
            handle.write(serialized)
            digest.update(serialized)
            byte_count += len(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    return path, digest.hexdigest(), byte_count


def _write_text_temporary(
    directory: Path, prefix: str, content: str, *, encoding: str
) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        dir=directory,
        prefix=prefix,
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _publish_no_overwrite(temporary_path: Path, final_path: Path) -> None:
    os.link(temporary_path, final_path)
    temporary_path.unlink()


def split_dataset_v1(
    dataset_manifest_path: Path,
    output_directory: Path,
    *,
    seed: str,
    validation_fraction: float = 0.1,
) -> DatasetV1SplitManifest:
    """Write a deterministic, provenance-grouped train/validation split."""
    if not seed:
        raise ValueError("seed must be non-empty")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    final_paths = {
        _TRAIN_FILE: output_directory / _TRAIN_FILE,
        _VALIDATION_FILE: output_directory / _VALIDATION_FILE,
        _MANIFEST_FILE: output_directory / _MANIFEST_FILE,
        _TRAIN_CHECKSUM_FILE: output_directory / _TRAIN_CHECKSUM_FILE,
        _VALIDATION_CHECKSUM_FILE: output_directory / _VALIDATION_CHECKSUM_FILE,
    }
    for path in final_paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

    source = _load_input(dataset_manifest_path)
    records = _read_records(source)
    groups = _connected_groups(records)
    validation_group_indexes = _select_validation_groups(
        groups,
        records,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    validation_indexes = {
        index
        for group_index in validation_group_indexes
        for index in groups[group_index]
    }
    train = tuple(
        record
        for index, record in enumerate(records)
        if index not in validation_indexes
    )
    validation = tuple(
        record for index, record in enumerate(records) if index in validation_indexes
    )
    train_identities = {item.identity for item in train}
    validation_identities = {item.identity for item in validation}
    group_leakage = sum(
        bool(set(group) & validation_indexes) and bool(set(group) - validation_indexes)
        for group in groups
    )
    key_splits = Counter()
    train_keys = {key for item in train for key in item.group_keys}
    validation_keys = {key for item in validation for key in item.group_keys}
    for key in train_keys & validation_keys:
        if key.startswith("collision:"):
            key_splits["collision"] += 1
        if key.startswith(("donor:", "generated_value:")):
            key_splits["donor_value"] += 1

    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    published_paths: list[Path] = []
    try:
        train_temporary, train_hash, train_bytes = _write_artifact(
            output_directory, "train", train
        )
        temporary_paths.append(train_temporary)
        validation_temporary, validation_hash, validation_bytes = _write_artifact(
            output_directory, "validation", validation
        )
        temporary_paths.append(validation_temporary)
        manifest = DatasetV1SplitManifest(
            kind="dataset_v1_split",
            splitter="grouped_dataset_v1_split_v1",
            schema_version=1,
            seed=seed,
            validation_fraction=validation_fraction,
            input_manifest=source.manifest_path.name,
            input_assembler=source.assembler,
            input_schema_version=source.schema_version,
            input_records=source.records,
            input_artifact_sha256=source.artifact_sha256,
            input_artifact_bytes=source.artifact_bytes,
            grouping_rules=[
                "google_source_sentence",
                "sgd_dialogue",
                "collision_pair",
                "generated_donor",
                "generated_spoken_value",
            ],
            total_groups=len(groups),
            largest_group_records=max(map(len, groups), default=0),
            train=_statistics(train, len(groups) - len(validation_group_indexes)),
            validation=_statistics(validation, len(validation_group_indexes)),
            group_leakage=group_leakage,
            exact_record_leakage=len(train_identities & validation_identities),
            collision_groups_split=key_splits["collision"],
            donor_value_groups_split=key_splits["donor_value"],
            train_artifact=_TRAIN_FILE,
            train_artifact_sha256=train_hash,
            train_artifact_bytes=train_bytes,
            validation_artifact=_VALIDATION_FILE,
            validation_artifact_sha256=validation_hash,
            validation_artifact_bytes=validation_bytes,
        )
        manifest_temporary = _write_text_temporary(
            output_directory,
            ".manifest.",
            json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_paths.append(manifest_temporary)
        train_checksum_temporary = _write_text_temporary(
            output_directory,
            ".train.sha256.",
            f"{train_hash}  {_TRAIN_FILE}\n",
            encoding="ascii",
        )
        temporary_paths.append(train_checksum_temporary)
        validation_checksum_temporary = _write_text_temporary(
            output_directory,
            ".validation.sha256.",
            f"{validation_hash}  {_VALIDATION_FILE}\n",
            encoding="ascii",
        )
        temporary_paths.append(validation_checksum_temporary)
        publications = (
            (train_temporary, final_paths[_TRAIN_FILE]),
            (validation_temporary, final_paths[_VALIDATION_FILE]),
            (train_checksum_temporary, final_paths[_TRAIN_CHECKSUM_FILE]),
            (
                validation_checksum_temporary,
                final_paths[_VALIDATION_CHECKSUM_FILE],
            ),
            (manifest_temporary, final_paths[_MANIFEST_FILE]),
        )
        for temporary_path, final_path in publications:
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
        description="Write the grouped Dataset V1 train/validation split."
    )
    parser.add_argument("dataset_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    manifest = split_dataset_v1(
        args.dataset_manifest,
        args.output_directory,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
