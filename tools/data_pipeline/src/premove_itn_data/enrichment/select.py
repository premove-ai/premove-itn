from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from premove_itn.labels import SpanKind
from premove_itn.types import WordToken
from premove_itn_data.enrichment.reservoir import _sgd_train_sha256
from premove_itn_data.enrichment.sgd_context import iter_sgd_train_user_turns
from premove_itn_data.records import (
    TrainingRecord,
    TrainingSpan,
    compile_training_record,
)
from premove_itn_data.selection import (
    DatasetSelectionRecord,
    SelectionQuotas,
    select_candidates,
)

_CANDIDATES_FILE = "candidates.jsonl"
_MANIFEST_FILE = "manifest.json"
_CHECKSUM_FILE = "candidates.sha256"
_CATEGORY_ORDER = (
    "sgd_positive",
    "sgd_context_only",
    "purpose_phone",
    "purpose_electronic",
    "purpose_digit_sequence",
    "collision_time_digit_sequence",
    "collision_phone_digit_sequence",
)
_PURPOSE_CATEGORIES = {
    "purpose_phone": SpanKind.PHONE,
    "purpose_electronic": SpanKind.ELECTRONIC,
    "purpose_digit_sequence": SpanKind.DIGIT_SEQUENCE,
}
_COLLISION_CATEGORIES = {
    "collision_time_digit_sequence": SpanKind.TIME,
    "collision_phone_digit_sequence": SpanKind.PHONE,
}


@dataclass(frozen=True, slots=True)
class EnrichmentSelectionQuotas:
    sgd_span_targets: Mapping[SpanKind | str, int]
    sgd_context_only_records: int
    purpose_phone_records: int
    purpose_electronic_records: int
    purpose_digit_sequence_records: int
    time_digit_sequence_pairs: int
    phone_digit_sequence_pairs: int

    def __post_init__(self) -> None:
        normalized: dict[SpanKind, int] = {}
        for raw_kind, target in self.sgd_span_targets.items():
            kind = SpanKind(raw_kind)
            if target < 0:
                raise ValueError(f"SGD target for {kind.value} must be non-negative")
            normalized[kind] = target
        object.__setattr__(self, "sgd_span_targets", MappingProxyType(normalized))

        for name in (
            "sgd_context_only_records",
            "purpose_phone_records",
            "purpose_electronic_records",
            "purpose_digit_sequence_records",
            "time_digit_sequence_pairs",
            "phone_digit_sequence_pairs",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


ENRICHMENT_V1_SELECTION_QUOTAS = EnrichmentSelectionQuotas(
    sgd_span_targets={
        SpanKind.DATE: 1_000,
        SpanKind.TIME: 1_500,
        SpanKind.CARDINAL: 500,
        SpanKind.MONEY: 300,
        SpanKind.DECIMAL: 250,
    },
    sgd_context_only_records=2_500,
    purpose_phone_records=3_500,
    purpose_electronic_records=3_500,
    purpose_digit_sequence_records=1_000,
    time_digit_sequence_pairs=61,
    phone_digit_sequence_pairs=1_000,
)


@dataclass(frozen=True, slots=True)
class EnrichmentSelectionManifest:
    kind: str
    selector: str
    schema_version: int
    seed: str
    input_manifest: str
    input_writer: str
    input_schema_version: int
    input_seed: str
    input_records: int
    input_artifact_sha256: str
    sgd_train_sha256: str
    sgd_span_targets: dict[str, int]
    sgd_context_only_target: int
    purpose_record_targets: dict[str, int]
    collision_pair_targets: dict[str, int]
    available_records_by_category: dict[str, int]
    available_span_free_sgd_context_records: int
    excluded_ignored_span_sgd_context_records: int
    selected_records_by_category: dict[str, int]
    selected_spans_by_kind: dict[str, int]
    selected_sgd_spans_by_kind: dict[str, int]
    sgd_span_shortfall_by_kind: dict[str, int]
    sgd_context_only_shortfall: int
    purpose_record_shortfalls: dict[str, int]
    selected_collision_pairs_by_category: dict[str, int]
    selected_same_realization_pairs_by_category: dict[str, int]
    collision_pair_shortfalls: dict[str, int]
    selected_context_only_records: int
    selected_multi_span_records: int
    output_records: int
    output_artifact: str
    output_artifact_sha256: str
    output_artifact_bytes: int


@dataclass(frozen=True, slots=True)
class _ReservoirRecord:
    record: TrainingRecord
    provenance: dict[str, object]

    @property
    def category(self) -> str:
        return str(self.provenance["category"])

    @property
    def context(self) -> str:
        return str(self.provenance["context"])


@dataclass(frozen=True, slots=True)
class _ReservoirInput:
    manifest_path: Path
    artifact_path: Path
    writer: str
    schema_version: int
    seed: str
    records: int
    artifact_sha256: str
    artifact_bytes: int
    records_by_category: dict[str, int]
    sgd_train_sha256: str


class _RankedRecordReservoir:
    def __init__(self, capacity: int, *, seed: str, namespace: str) -> None:
        self.capacity = capacity
        self.seed = seed
        self.namespace = namespace
        self.items: dict[str, tuple[int, _ReservoirRecord]] = {}
        self.heap: list[tuple[int, str]] = []

    def offer(self, item: _ReservoirRecord) -> None:
        identity = hashlib.sha256(
            _canonical_record(item.record).encode("utf-8")
        ).hexdigest()
        rank = _rank(self.seed, self.namespace, identity)
        current = self.items.get(identity)
        if current is not None:
            if item.context < current[1].context:
                self.items[identity] = (rank, item)
            return
        if not self.capacity:
            return
        if len(self.items) < self.capacity:
            self.items[identity] = (rank, item)
            heapq.heappush(self.heap, (-rank, identity))
            return

        worst_rank, worst_identity = self.heap[0]
        if (rank, identity) >= (-worst_rank, worst_identity):
            return
        heapq.heapreplace(self.heap, (-rank, identity))
        del self.items[worst_identity]
        self.items[identity] = (rank, item)

    def ranked(self) -> tuple[_ReservoirRecord, ...]:
        return tuple(
            item
            for _, item in sorted(
                self.items.values(),
                key=lambda value: (
                    value[0],
                    hashlib.sha256(
                        _canonical_record(value[1].record).encode("utf-8")
                    ).hexdigest(),
                ),
            )
        )


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


def _load_reservoir_input(manifest_path: Path) -> _ReservoirInput:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = _require_mapping(
            json.load(handle), location="enrichment reservoir manifest"
        )
    if manifest.get("kind") != "enrichment_compiled_reservoir":
        raise ValueError("input is not an enrichment compiled-reservoir manifest")
    schema_version = _require_int(
        manifest.get("schema_version"), location="manifest.schema_version"
    )
    if schema_version != 1:
        raise ValueError("unsupported enrichment reservoir schema_version")
    artifact_file = _require_string(
        manifest.get("artifact_file"), location="manifest.artifact_file"
    )
    if Path(artifact_file).name != artifact_file:
        raise ValueError("manifest.artifact_file must be a plain file name")
    artifact_path = manifest_path.parent / artifact_file
    if not artifact_path.is_file():
        raise FileNotFoundError(f"reservoir artifact does not exist: {artifact_path}")

    raw_category_counts = _require_mapping(
        manifest.get("records_by_category"),
        location="manifest.records_by_category",
    )
    category_counts = {
        _require_string(category, location="manifest category"): _require_int(
            count, location=f"manifest.records_by_category.{category}"
        )
        for category, count in raw_category_counts.items()
    }
    return _ReservoirInput(
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        writer=_require_string(manifest.get("writer"), location="manifest.writer"),
        schema_version=schema_version,
        seed=_require_string(manifest.get("seed"), location="manifest.seed"),
        records=_require_int(
            manifest.get("output_records"), location="manifest.output_records"
        ),
        artifact_sha256=_require_string(
            manifest.get("artifact_sha256"), location="manifest.artifact_sha256"
        ),
        artifact_bytes=_require_int(
            manifest.get("artifact_bytes"), location="manifest.artifact_bytes"
        ),
        records_by_category=category_counts,
        sgd_train_sha256=_require_string(
            manifest.get("sgd_train_sha256"), location="manifest.sgd_train_sha256"
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
    compiled = compile_training_record(text, spans, expected_text=expected_text)
    if record != compiled:
        raise ValueError(f"{location} does not match shared compiler output")
    return record


def _deserialize_reservoir_record(
    value: object, *, location: str
) -> _ReservoirRecord:
    payload = _require_mapping(value, location=location)
    provenance = dict(
        _require_mapping(payload.get("provenance"), location=f"{location}.provenance")
    )
    if provenance.get("source") != "enrichment":
        raise ValueError(f"{location}.provenance.source must be 'enrichment'")
    category = _require_string(
        provenance.get("category"), location=f"{location}.provenance.category"
    )
    if category not in _CATEGORY_ORDER:
        raise ValueError(f"{location} has unsupported category {category!r}")
    _require_string(
        provenance.get("context"), location=f"{location}.provenance.context"
    )
    raw_span_provenance = provenance.get("spans")
    if not isinstance(raw_span_provenance, list):
        raise ValueError(f"{location}.provenance.spans must be a JSON array")
    record = _deserialize_record(payload.get("record"), location=f"{location}.record")
    if len(raw_span_provenance) != len(record.spans):
        raise ValueError(f"{location} span provenance count does not match spans")
    for index, raw_span in enumerate(raw_span_provenance):
        where = f"{location}.provenance.spans[{index}]"
        span = _require_mapping(raw_span, location=where)
        _require_string(span.get("context"), location=f"{where}.context")
        _require_string(span.get("donor"), location=f"{where}.donor")
    return _ReservoirRecord(record, provenance)


def _canonical_record(record: TrainingRecord) -> str:
    return json.dumps(
        asdict(record),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _record_identity(item: _ReservoirRecord) -> str:
    return hashlib.sha256(_canonical_record(item.record).encode("utf-8")).hexdigest()


def _rank(seed: str, namespace: str, identity: str) -> int:
    digest = hashlib.sha256(
        "\0".join((namespace, seed, identity)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big")


def _ranked_unique_records(
    records: Sequence[_ReservoirRecord],
    count: int,
    *,
    seed: str,
    namespace: str,
    excluded_identities: frozenset[str] = frozenset(),
) -> tuple[_ReservoirRecord, ...]:
    winners: dict[str, _ReservoirRecord] = {}
    for item in records:
        identity = _record_identity(item)
        if identity in excluded_identities:
            continue
        current = winners.get(identity)
        if current is None or item.context < current.context:
            winners[identity] = item
    ranked = sorted(
        winners.items(),
        key=lambda pair: (_rank(seed, namespace, pair[0]), pair[0]),
    )
    return tuple(item for _, item in ranked[:count])


def _sgd_context_index(
    train_directory: Path,
) -> dict[str, tuple[str, bool]]:
    contexts: dict[str, tuple[str, bool]] = {}
    for turn in iter_sgd_train_user_turns(train_directory):
        context = f"sgd/{turn.source_file}/{turn.dialogue_id}/{turn.turn_index}"
        if context in contexts:
            raise ValueError(f"duplicate SGD context identity: {context}")
        contexts[context] = (turn.utterance, bool(turn.spans))
    return contexts


def _collision_group(item: _ReservoirRecord) -> str:
    parts = item.context.split("/")
    if (
        len(parts) != 5
        or parts[0] != "purpose_built"
        or parts[1] != "collision"
        or not parts[2]
    ):
        raise ValueError(f"invalid collision context provenance: {item.context}")
    return parts[2]


def _validated_collision_pairs(
    category: str,
    records: Sequence[_ReservoirRecord],
) -> tuple[tuple[_ReservoirRecord, _ReservoirRecord], ...]:
    contextual_kind = _COLLISION_CATEGORIES[category]
    grouped: dict[str, list[_ReservoirRecord]] = defaultdict(list)
    for item in records:
        grouped[_collision_group(item)].append(item)

    pairs: list[tuple[_ReservoirRecord, _ReservoirRecord]] = []
    for group in sorted(grouped):
        values = grouped[group]
        if len(values) != 2:
            raise ValueError(
                f"collision group {group} must contain exactly two records"
            )
        by_kind = {
            item.record.spans[0].kind: item
            for item in values
            if len(item.record.spans) == 1
        }
        if set(by_kind) != {contextual_kind, SpanKind.DIGIT_SEQUENCE}:
            raise ValueError(f"collision group {group} has unexpected span kinds")
        contextual = by_kind[contextual_kind]
        digit = by_kind[SpanKind.DIGIT_SEQUENCE]
        if contextual.record.spans[0].source != digit.record.spans[0].source:
            raise ValueError(f"collision group {group} has different spoken forms")
        pairs.append((contextual, digit))
    return tuple(pairs)


def _select_collision_pairs(
    pairs: Sequence[tuple[_ReservoirRecord, _ReservoirRecord]],
    count: int,
    *,
    seed: str,
    namespace: str,
    excluded_identities: set[str],
) -> tuple[tuple[_ReservoirRecord, _ReservoirRecord], ...]:
    ranked = sorted(
        pairs,
        key=lambda pair: (
            _rank(
                seed,
                namespace,
                hashlib.sha256(
                    "\0".join(
                        sorted(_canonical_record(item.record) for item in pair)
                    ).encode("utf-8")
                ).hexdigest(),
            ),
            _collision_group(pair[0]),
        ),
    )
    selected: list[tuple[_ReservoirRecord, _ReservoirRecord]] = []
    for pair in ranked:
        identities = {_record_identity(item) for item in pair}
        if identities & excluded_identities:
            continue
        selected.append(pair)
        excluded_identities.update(identities)
        if len(selected) == count:
            break
    return tuple(selected)


def _selected_payload(item: _ReservoirRecord) -> dict[str, object]:
    return {"provenance": item.provenance, "record": asdict(item.record)}


def _write_temporary(
    directory: Path,
    *,
    prefix: str,
    mode: str,
    encoding: str | None = None,
):
    return tempfile.NamedTemporaryFile(
        mode=mode,
        encoding=encoding,
        dir=directory,
        prefix=prefix,
        suffix=".tmp",
        delete=False,
    )


def _publish_no_overwrite(temporary_path: Path, final_path: Path) -> None:
    os.link(temporary_path, final_path)
    temporary_path.unlink()


def write_enrichment_candidates(
    reservoir_manifest_path: Path,
    train_directory: Path,
    output_directory: Path,
    *,
    seed: str,
    quotas: EnrichmentSelectionQuotas = ENRICHMENT_V1_SELECTION_QUOTAS,
) -> EnrichmentSelectionManifest:
    """Select and atomically persist a balanced enrichment contribution."""
    if not seed:
        raise ValueError("seed must be non-empty")
    candidates_path = output_directory / _CANDIDATES_FILE
    manifest_path = output_directory / _MANIFEST_FILE
    checksum_path = output_directory / _CHECKSUM_FILE
    for path in (candidates_path, manifest_path, checksum_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

    reservoir = _load_reservoir_input(reservoir_manifest_path)
    train_hash = _sgd_train_sha256(train_directory)
    if train_hash != reservoir.sgd_train_sha256:
        raise ValueError("SGD train source does not match reservoir manifest")
    sgd_contexts = _sgd_context_index(train_directory)

    records_by_category: dict[str, list[_ReservoirRecord]] = {
        category: []
        for category in _CATEGORY_ORDER
        if category != "sgd_context_only"
    }
    context_reservoir = _RankedRecordReservoir(
        quotas.sgd_context_only_records,
        seed=seed,
        namespace="sgd-context-only-v1",
    )
    span_free_context_count = 0
    ignored_span_context_count = 0
    input_digest = hashlib.sha256()
    input_bytes = 0
    input_records = 0
    input_category_counts: Counter[str] = Counter()
    with reservoir.artifact_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            input_digest.update(line)
            input_bytes += len(line)
            try:
                raw_payload = json.loads(line)
                item = _deserialize_reservoir_record(
                    raw_payload,
                    location=f"{reservoir.artifact_path}:{line_number}",
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid reservoir record at "
                    f"{reservoir.artifact_path}:{line_number}"
                ) from error
            if item.category == "sgd_context_only":
                source = sgd_contexts.get(item.context)
                if source is None:
                    raise ValueError(
                        "reservoir references unknown SGD context: "
                        f"{item.context}"
                    )
                utterance, has_spans = source
                if item.record.text != utterance:
                    raise ValueError(
                        f"SGD context text does not match source: {item.context}"
                    )
                if has_spans:
                    ignored_span_context_count += 1
                else:
                    span_free_context_count += 1
                    context_reservoir.offer(item)
            else:
                records_by_category[item.category].append(item)
            input_category_counts[item.category] += 1
            input_records += 1

    if input_digest.hexdigest() != reservoir.artifact_sha256:
        raise ValueError("reservoir artifact SHA-256 does not match manifest")
    if input_bytes != reservoir.artifact_bytes:
        raise ValueError("reservoir artifact byte count does not match manifest")
    if input_records != reservoir.records:
        raise ValueError("reservoir record count does not match manifest")
    if dict(sorted(input_category_counts.items())) != reservoir.records_by_category:
        raise ValueError("reservoir category counts do not match manifest")

    sgd_positive_records = records_by_category["sgd_positive"]
    for item in sgd_positive_records:
        if item.context not in sgd_contexts:
            raise ValueError(
                f"reservoir references unknown SGD context: {item.context}"
            )
    sgd_by_context = {item.context: item for item in sgd_positive_records}
    if len(sgd_by_context) != len(sgd_positive_records):
        raise ValueError("SGD positive contexts must be unique")
    sgd_result = select_candidates(
        (
            DatasetSelectionRecord(
                record=item.record,
                source="enrichment",
                source_file=item.context,
                sentence_number=0,
            )
            for item in sgd_positive_records
        ),
        SelectionQuotas(quotas.sgd_span_targets, context_only_records=0),
        seed=f"{seed}/sgd-positive",
    )
    selected_by_category: dict[str, tuple[_ReservoirRecord, ...]] = {
        "sgd_positive": tuple(
            sgd_by_context[record.source_file]
            for record in sgd_result.selected_records
        ),
        "sgd_context_only": context_reservoir.ranked(),
    }
    selected_identities = {
        _record_identity(item)
        for category in ("sgd_positive", "sgd_context_only")
        for item in selected_by_category[category]
    }

    collision_targets = {
        "collision_time_digit_sequence": quotas.time_digit_sequence_pairs,
        "collision_phone_digit_sequence": quotas.phone_digit_sequence_pairs,
    }
    selected_pairs: dict[
        str, tuple[tuple[_ReservoirRecord, _ReservoirRecord], ...]
    ] = {}
    for category, target in collision_targets.items():
        pairs = _validated_collision_pairs(category, records_by_category[category])
        chosen = _select_collision_pairs(
            pairs,
            target,
            seed=seed,
            namespace=f"{category}-v1",
            excluded_identities=selected_identities,
        )
        selected_pairs[category] = chosen
        selected_by_category[category] = tuple(
            item for pair in chosen for item in pair
        )

    purpose_targets = {
        "purpose_phone": quotas.purpose_phone_records,
        "purpose_electronic": quotas.purpose_electronic_records,
        "purpose_digit_sequence": quotas.purpose_digit_sequence_records,
    }
    for category, target in purpose_targets.items():
        expected_kind = _PURPOSE_CATEGORIES[category]
        values = records_by_category[category]
        if any(
            len(item.record.spans) != 1
            or item.record.spans[0].kind is not expected_kind
            for item in values
        ):
            raise ValueError(f"{category} contains an unexpected span")
        chosen = _ranked_unique_records(
            values,
            target,
            seed=seed,
            namespace=f"{category}-v1",
            excluded_identities=frozenset(selected_identities),
        )
        selected_by_category[category] = chosen
        selected_identities.update(_record_identity(item) for item in chosen)

    selected = tuple(
        item
        for category in _CATEGORY_ORDER
        for item in selected_by_category[category]
    )
    unique_selected: dict[str, _ReservoirRecord] = {}
    for item in selected:
        identity = _canonical_record(item.record)
        current = unique_selected.get(identity)
        if current is not None:
            raise ValueError(
                "selected enrichment records must have unique content: "
                f"{current.category}/{current.context} and "
                f"{item.category}/{item.context}"
            )
        unique_selected[identity] = item

    selected_category_counts = {
        category: len(selected_by_category[category])
        for category in _CATEGORY_ORDER
    }
    selected_spans = Counter(
        span.kind.value for item in selected for span in item.record.spans
    )
    same_realization_pairs = {
        category: sum(
            pair[0].record.spans[0].replacement
            == pair[1].record.spans[0].replacement
            for pair in pairs
        )
        for category, pairs in selected_pairs.items()
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    published_paths: list[Path] = []
    try:
        output_digest = hashlib.sha256()
        output_bytes = 0
        with _write_temporary(
            output_directory, prefix=".candidates.", mode="wb"
        ) as handle:
            candidates_temporary = Path(handle.name)
            temporary_paths.append(candidates_temporary)
            for item in selected:
                serialized = (
                    json.dumps(
                        _selected_payload(item),
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

        output_sha256 = output_digest.hexdigest()
        manifest = EnrichmentSelectionManifest(
            kind="enrichment_selection_candidates",
            selector="balanced_enrichment_v1",
            schema_version=1,
            seed=seed,
            input_manifest=reservoir.manifest_path.name,
            input_writer=reservoir.writer,
            input_schema_version=reservoir.schema_version,
            input_seed=reservoir.seed,
            input_records=reservoir.records,
            input_artifact_sha256=reservoir.artifact_sha256,
            sgd_train_sha256=train_hash,
            sgd_span_targets=dict(
                sorted(
                    (kind.value, target)
                    for kind, target in quotas.sgd_span_targets.items()
                )
            ),
            sgd_context_only_target=quotas.sgd_context_only_records,
            purpose_record_targets=dict(sorted(purpose_targets.items())),
            collision_pair_targets=dict(sorted(collision_targets.items())),
            available_records_by_category=dict(
                sorted(input_category_counts.items())
            ),
            available_span_free_sgd_context_records=span_free_context_count,
            excluded_ignored_span_sgd_context_records=ignored_span_context_count,
            selected_records_by_category=selected_category_counts,
            selected_spans_by_kind=dict(sorted(selected_spans.items())),
            selected_sgd_spans_by_kind=sgd_result.actual_by_kind,
            sgd_span_shortfall_by_kind=sgd_result.shortfall_by_kind,
            sgd_context_only_shortfall=max(
                0,
                quotas.sgd_context_only_records
                - len(selected_by_category["sgd_context_only"]),
            ),
            purpose_record_shortfalls={
                category: target - len(selected_by_category[category])
                for category, target in sorted(purpose_targets.items())
                if len(selected_by_category[category]) < target
            },
            selected_collision_pairs_by_category={
                category: len(pairs)
                for category, pairs in sorted(selected_pairs.items())
            },
            selected_same_realization_pairs_by_category=dict(
                sorted(same_realization_pairs.items())
            ),
            collision_pair_shortfalls={
                category: target - len(selected_pairs[category])
                for category, target in sorted(collision_targets.items())
                if len(selected_pairs[category]) < target
            },
            selected_context_only_records=len(
                selected_by_category["sgd_context_only"]
            ),
            selected_multi_span_records=sum(
                len(item.record.spans) > 1 for item in selected
            ),
            output_records=len(selected),
            output_artifact=_CANDIDATES_FILE,
            output_artifact_sha256=output_sha256,
            output_artifact_bytes=output_bytes,
        )

        with _write_temporary(
            output_directory,
            prefix=".manifest.",
            mode="w",
            encoding="utf-8",
        ) as handle:
            manifest_temporary = Path(handle.name)
            temporary_paths.append(manifest_temporary)
            json.dump(asdict(manifest), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        with _write_temporary(
            output_directory,
            prefix=".candidates.sha256.",
            mode="w",
            encoding="ascii",
        ) as handle:
            checksum_temporary = Path(handle.name)
            temporary_paths.append(checksum_temporary)
            handle.write(f"{output_sha256}  {_CANDIDATES_FILE}\n")
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
        description="Select the balanced Dataset V1 enrichment contribution."
    )
    parser.add_argument("reservoir_manifest", type=Path)
    parser.add_argument("train_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args()

    manifest = write_enrichment_candidates(
        args.reservoir_manifest,
        args.train_directory,
        args.output_directory,
        seed=args.seed,
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
