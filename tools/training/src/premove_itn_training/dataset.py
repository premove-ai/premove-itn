from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from premove_itn_training.labels import LABEL_TO_ID, TRAINED_KINDS
from premove_itn_training.metrics import WordSpan, extract_spans

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class TrainingExample:
    identity: str
    tokens: tuple[str, ...]
    label_ids: tuple[int, ...]
    context_only: bool
    multi_span: bool


@dataclass(frozen=True, slots=True)
class FrozenSplit:
    manifest_path: Path
    train: tuple[TrainingExample, ...]
    validation: tuple[TrainingExample, ...]
    train_sha256: str
    validation_sha256: str


def _mapping(value: object, *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    return value


def _string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _integer(value: object, *, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{location} must be an integer")
    return value


def _plain_file_name(value: object, *, location: str) -> str:
    name = _string(value, location=location)
    if Path(name).name != name:
        raise ValueError(f"{location} must be a plain file name")
    return name


def _sha256(value: object, *, location: str) -> str:
    digest = _string(value, location=location)
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{location} must be a lowercase SHA-256")
    return digest


def _checksum(path: Path, *, artifact_name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"split checksum does not exist: {path}")
    fields = path.read_text(encoding="ascii").split()
    if (
        len(fields) != 2
        or not _SHA256_PATTERN.fullmatch(fields[0])
        or fields[1] != artifact_name
    ):
        raise ValueError(f"invalid split checksum file: {path}")
    return fields[0]


def _canonical_record(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_record(value: object, *, location: str) -> TrainingExample:
    envelope = _mapping(value, location=location)
    provenance = _mapping(
        envelope.get("provenance"), location=f"{location}.provenance"
    )
    primary = _mapping(
        provenance.get("primary"), location=f"{location}.provenance.primary"
    )
    _string(primary.get("source"), location=f"{location}.provenance.primary.source")
    duplicates = provenance.get("duplicates")
    if not isinstance(duplicates, list):
        raise ValueError(f"{location}.provenance.duplicates must be a JSON array")

    record = _mapping(envelope.get("record"), location=f"{location}.record")
    text = _string(record.get("text"), location=f"{location}.record.text")
    _string(
        record.get("expected_text"), location=f"{location}.record.expected_text"
    )
    raw_tokens = record.get("tokens")
    raw_labels = record.get("bio_labels")
    raw_spans = record.get("spans")
    if not isinstance(raw_tokens, list):
        raise ValueError(f"{location}.record.tokens must be a JSON array")
    if not isinstance(raw_labels, list):
        raise ValueError(f"{location}.record.bio_labels must be a JSON array")
    if not isinstance(raw_spans, list):
        raise ValueError(f"{location}.record.spans must be a JSON array")
    if len(raw_tokens) != len(raw_labels):
        raise ValueError(f"{location}.record token and BIO label counts must match")

    tokens: list[str] = []
    token_offsets: list[tuple[int, int]] = []
    previous_end = 0
    for index, raw_token in enumerate(raw_tokens):
        token_location = f"{location}.record.tokens[{index}]"
        token = _mapping(raw_token, location=token_location)
        token_text = _string(token.get("text"), location=f"{token_location}.text")
        start = _integer(token.get("start"), location=f"{token_location}.start")
        end = _integer(token.get("end"), location=f"{token_location}.end")
        if start < previous_end or end <= start or end > len(text):
            raise ValueError(f"{token_location} has invalid ordered offsets")
        if text[start:end] != token_text:
            raise ValueError(f"{token_location}.text does not match its source slice")
        tokens.append(token_text)
        token_offsets.append((start, end))
        previous_end = end

    label_ids: list[int] = []
    for index, raw_label in enumerate(raw_labels):
        if not isinstance(raw_label, str) or raw_label not in LABEL_TO_ID:
            raise ValueError(
                f"{location}.record.bio_labels[{index}] is not a Model V1 label"
            )
        label_ids.append(LABEL_TO_ID[raw_label])

    decoded = extract_spans(label_ids)
    expected_spans: list[WordSpan] = []
    for index, raw_span in enumerate(raw_spans):
        span_location = f"{location}.record.spans[{index}]"
        span = _mapping(raw_span, location=span_location)
        kind = _string(span.get("kind"), location=f"{span_location}.kind")
        if kind not in TRAINED_KINDS:
            raise ValueError(f"{span_location}.kind is outside the Model V1 contract")
        start = _integer(span.get("start"), location=f"{span_location}.start")
        end = _integer(span.get("end"), location=f"{span_location}.end")
        source = _string(span.get("source"), location=f"{span_location}.source")
        _string(span.get("replacement"), location=f"{span_location}.replacement")
        if start < 0 or end <= start or end > len(text) or text[start:end] != source:
            raise ValueError(f"{span_location} does not match its source slice")
        covered = [
            token_index
            for token_index, (token_start, token_end) in enumerate(token_offsets)
            if token_start >= start and token_end <= end
        ]
        if (
            not covered
            or token_offsets[covered[0]][0] != start
            or token_offsets[covered[-1]][1] != end
        ):
            raise ValueError(f"{span_location} does not align to source words")
        expected_spans.append(WordSpan(covered[0], covered[-1] + 1, kind))
    if tuple(expected_spans) != decoded:
        raise ValueError(f"{location}.record spans and BIO labels disagree")

    canonical = _canonical_record(record)
    return TrainingExample(
        identity=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        tokens=tuple(tokens),
        label_ids=tuple(label_ids),
        context_only=not decoded,
        multi_span=len(decoded) > 1,
    )


def _load_partition(
    artifact_path: Path,
    *,
    expected_hash: str,
    expected_bytes: int,
    expected_statistics: Mapping[str, Any],
) -> tuple[TrainingExample, ...]:
    if not artifact_path.is_file():
        raise FileNotFoundError(f"split artifact does not exist: {artifact_path}")
    digest = hashlib.sha256()
    byte_count = 0
    examples: list[TrainingExample] = []
    identities: set[str] = set()
    source_counts: Counter[str] = Counter()
    spans_by_kind: Counter[str] = Counter()

    with artifact_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            byte_count += len(line)
            location = f"{artifact_path}:{line_number}"
            try:
                payload = json.loads(line)
                example = _validate_record(payload, location=location)
                envelope = _mapping(payload, location=location)
                provenance = _mapping(
                    envelope.get("provenance"), location=f"{location}.provenance"
                )
                primary = _mapping(
                    provenance.get("primary"),
                    location=f"{location}.provenance.primary",
                )
                source = _string(
                    primary.get("source"),
                    location=f"{location}.provenance.primary.source",
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid training record at {location}") from error
            if example.identity in identities:
                raise ValueError(f"duplicate compiled training record at {location}")
            identities.add(example.identity)
            examples.append(example)
            source_counts[source] += 1
            spans_by_kind.update(span.kind for span in extract_spans(example.label_ids))

    if digest.hexdigest() != expected_hash:
        raise ValueError(
            f"split artifact SHA-256 does not match manifest: {artifact_path}"
        )
    if byte_count != expected_bytes:
        raise ValueError(
            f"split artifact byte count does not match manifest: {artifact_path}"
        )

    expected_records = _integer(
        expected_statistics.get("records"), location="split statistics.records"
    )
    expected_context_only = _integer(
        expected_statistics.get("context_only_records"),
        location="split statistics.context_only_records",
    )
    expected_multi_span = _integer(
        expected_statistics.get("multi_span_records"),
        location="split statistics.multi_span_records",
    )
    expected_sources = _mapping(
        expected_statistics.get("source_counts"),
        location="split statistics.source_counts",
    )
    expected_kinds = _mapping(
        expected_statistics.get("spans_by_kind"),
        location="split statistics.spans_by_kind",
    )
    if len(examples) != expected_records:
        raise ValueError(
            f"split artifact record count does not match manifest: {artifact_path}"
        )
    if sum(item.context_only for item in examples) != expected_context_only:
        raise ValueError(f"context-only count does not match manifest: {artifact_path}")
    if sum(item.multi_span for item in examples) != expected_multi_span:
        raise ValueError(f"multi-span count does not match manifest: {artifact_path}")
    if dict(sorted(source_counts.items())) != dict(expected_sources):
        raise ValueError(f"source counts do not match manifest: {artifact_path}")
    if dict(sorted(spans_by_kind.items())) != dict(expected_kinds):
        raise ValueError(f"span counts do not match manifest: {artifact_path}")
    return tuple(examples)


def load_frozen_split(manifest_path: Path) -> FrozenSplit:
    """Verify and load the immutable Dataset V1 train/validation split."""
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = _mapping(json.load(handle), location="split manifest")
    except json.JSONDecodeError as error:
        raise ValueError(f"split manifest is invalid JSON: {manifest_path}") from error
    if manifest.get("kind") != "dataset_v1_split":
        raise ValueError("input is not a Dataset V1 split manifest")
    if manifest.get("splitter") != "grouped_dataset_v1_split_v1":
        raise ValueError("input was not produced by the frozen Dataset V1 splitter")
    schema_version = _integer(
        manifest.get("schema_version"), location="manifest.schema_version"
    )
    if schema_version != 1:
        raise ValueError("unsupported Dataset V1 split schema_version")
    if _integer(manifest.get("group_leakage"), location="manifest.group_leakage") != 0:
        raise ValueError("split manifest reports group leakage")
    if _integer(
        manifest.get("exact_record_leakage"),
        location="manifest.exact_record_leakage",
    ) != 0:
        raise ValueError("split manifest reports exact-record leakage")

    directory = manifest_path.parent
    train_name = _plain_file_name(
        manifest.get("train_artifact"), location="manifest.train_artifact"
    )
    validation_name = _plain_file_name(
        manifest.get("validation_artifact"),
        location="manifest.validation_artifact",
    )
    train_hash = _sha256(
        manifest.get("train_artifact_sha256"),
        location="manifest.train_artifact_sha256",
    )
    validation_hash = _sha256(
        manifest.get("validation_artifact_sha256"),
        location="manifest.validation_artifact_sha256",
    )
    if _checksum(directory / "train.sha256", artifact_name=train_name) != train_hash:
        raise ValueError("train checksum does not match split manifest")
    if (
        _checksum(directory / "validation.sha256", artifact_name=validation_name)
        != validation_hash
    ):
        raise ValueError("validation checksum does not match split manifest")

    train = _load_partition(
        directory / train_name,
        expected_hash=train_hash,
        expected_bytes=_integer(
            manifest.get("train_artifact_bytes"),
            location="manifest.train_artifact_bytes",
        ),
        expected_statistics=_mapping(manifest.get("train"), location="manifest.train"),
    )
    validation = _load_partition(
        directory / validation_name,
        expected_hash=validation_hash,
        expected_bytes=_integer(
            manifest.get("validation_artifact_bytes"),
            location="manifest.validation_artifact_bytes",
        ),
        expected_statistics=_mapping(
            manifest.get("validation"), location="manifest.validation"
        ),
    )
    if {item.identity for item in train} & {item.identity for item in validation}:
        raise ValueError("exact compiled records leak across train and validation")
    return FrozenSplit(manifest_path, train, validation, train_hash, validation_hash)
