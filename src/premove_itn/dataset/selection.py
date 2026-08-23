from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType

from premove_itn.dataset.records import TrainingRecord
from premove_itn.labels import SpanKind

_RARE_FIRST_ORDER = (
    SpanKind.ELECTRONIC,
    SpanKind.PHONE,
    SpanKind.DIGIT_SEQUENCE,
    SpanKind.TIME,
    SpanKind.MONEY,
    SpanKind.DECIMAL,
    SpanKind.MEASUREMENT,
    SpanKind.ORDINAL,
    SpanKind.CARDINAL,
    SpanKind.DATE,
)


@dataclass(frozen=True, slots=True)
class DatasetSelectionRecord:
    record: TrainingRecord
    source: str
    source_file: str
    sentence_number: int

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source must be non-empty")
        if not self.source_file:
            raise ValueError("source_file must be non-empty")
        if self.sentence_number < 0:
            raise ValueError("sentence_number must be non-negative")


@dataclass(frozen=True, slots=True)
class SelectionQuotas:
    span_targets: Mapping[SpanKind | str, int]
    context_only_records: int

    def __post_init__(self) -> None:
        normalized: dict[SpanKind, int] = {}
        for raw_kind, target in self.span_targets.items():
            kind = SpanKind(raw_kind)
            if target < 0:
                raise ValueError(f"target for {kind.value} must be non-negative")
            normalized[kind] = target
        if self.context_only_records < 0:
            raise ValueError("context_only_records must be non-negative")
        object.__setattr__(self, "span_targets", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_records: tuple[DatasetSelectionRecord, ...]
    records_scanned: int
    retained_unique_candidates: int
    duplicates_removed: int
    selected_record_count: int
    selected_span_count: int
    selected_context_only_count: int
    target_by_kind: dict[str, int]
    actual_by_kind: dict[str, int]
    shortfall_by_kind: dict[str, int]
    overshoot_by_kind: dict[str, int]
    multi_span_selected: int
    source_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class _RankedCandidate:
    selection_record: DatasetSelectionRecord
    content_hash: str
    rank: int


class _CandidateReservoir:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: dict[str, _RankedCandidate] = {}
        self.heap: list[tuple[int, str]] = []
        self.duplicates_removed = 0

    def offer(self, candidate: _RankedCandidate) -> None:
        existing = self.items.get(candidate.content_hash)
        if existing is not None:
            self.duplicates_removed += 1
            if _provenance_key(candidate.selection_record) < _provenance_key(
                existing.selection_record
            ):
                self.items[candidate.content_hash] = candidate
            return

        if len(self.items) < self.capacity:
            self.items[candidate.content_hash] = candidate
            heapq.heappush(self.heap, (-candidate.rank, candidate.content_hash))
            return

        worst_rank, worst_hash = self.heap[0]
        candidate_key = (candidate.rank, candidate.content_hash)
        worst_key = (-worst_rank, worst_hash)
        if candidate_key >= worst_key:
            return

        heapq.heapreplace(self.heap, (-candidate.rank, candidate.content_hash))
        del self.items[worst_hash]
        self.items[candidate.content_hash] = candidate

    def ranked_items(self) -> tuple[_RankedCandidate, ...]:
        return tuple(
            sorted(self.items.values(), key=lambda item: (item.rank, item.content_hash))
        )


def _provenance_key(candidate: DatasetSelectionRecord) -> tuple[str, int, str]:
    return candidate.source_file, candidate.sentence_number, candidate.source


def _canonical_record(record: TrainingRecord) -> str:
    return json.dumps(
        asdict(record),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _rank(seed: str, content_hash: str) -> int:
    digest = hashlib.sha256(
        seed.encode("utf-8") + b"\0" + content_hash.encode("ascii")
    ).digest()
    return int.from_bytes(digest, byteorder="big")


def _ordered_kinds(targets: Mapping[SpanKind, int]) -> tuple[SpanKind, ...]:
    known = tuple(kind for kind in _RARE_FIRST_ORDER if kind in targets)
    remaining = tuple(sorted(set(targets) - set(known), key=lambda kind: kind.value))
    return known + remaining


def _merge_candidate(
    candidates: dict[str, _RankedCandidate], candidate: _RankedCandidate
) -> None:
    existing = candidates.get(candidate.content_hash)
    if existing is None or (
        _provenance_key(candidate.selection_record)
        < _provenance_key(existing.selection_record)
    ):
        candidates[candidate.content_hash] = candidate


def select_candidates(
    records: Iterable[DatasetSelectionRecord],
    quotas: SelectionQuotas,
    *,
    seed: str,
) -> SelectionResult:
    """Select a deterministic, bounded candidate set from compiled records.

    The selector ranks record content, not provenance. Each requested class has
    a reservoir no larger than its span target, and context-only records have a
    separate reservoir. Allocation visits classes in rare-first order. A record
    is selected only while it contributes to an underfilled target class.

    ``retained_unique_candidates`` counts unique content identities retained by
    the bounded reservoirs. ``duplicates_removed`` counts duplicate content
    offers removed from the
    bounded class reservoirs. The final selected records are always unique by
    compiled record content.
    """
    if not seed:
        raise ValueError("seed must be non-empty")

    reservoirs = {
        kind: _CandidateReservoir(target)
        for kind, target in quotas.span_targets.items()
        if target > 0
    }
    context_reservoir = _CandidateReservoir(quotas.context_only_records)
    records_scanned = 0

    for selection_record in records:
        records_scanned += 1
        serialized = _canonical_record(selection_record.record)
        content_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        candidate = _RankedCandidate(
            selection_record,
            content_hash,
            _rank(seed, content_hash),
        )
        kinds = {
            span.kind
            for span in selection_record.record.spans
            if span.kind in reservoirs
        }
        if kinds:
            for kind in kinds:
                reservoirs[kind].offer(candidate)
        elif not selection_record.record.spans and context_reservoir.capacity:
            context_reservoir.offer(candidate)

    candidates: dict[str, _RankedCandidate] = {}
    for reservoir in (*reservoirs.values(), context_reservoir):
        for candidate in reservoir.items.values():
            _merge_candidate(candidates, candidate)

    selected: list[_RankedCandidate] = []
    selected_hashes: set[str] = set()
    actual: Counter[SpanKind] = Counter()
    target_kinds = _ordered_kinds(quotas.span_targets)

    for kind in target_kinds:
        target = quotas.span_targets[kind]
        if actual[kind] >= target:
            continue
        reservoir = reservoirs.get(kind)
        if reservoir is None:
            continue
        for candidate in reservoir.ranked_items():
            if actual[kind] >= target:
                break
            if candidate.content_hash in selected_hashes:
                continue
            selected.append(candidate)
            selected_hashes.add(candidate.content_hash)
            actual.update(span.kind for span in candidate.selection_record.record.spans)

    selected_context_only = 0
    if quotas.context_only_records:
        for candidate in context_reservoir.ranked_items():
            if selected_context_only >= quotas.context_only_records:
                break
            if candidate.content_hash in selected_hashes:
                continue
            selected.append(candidate)
            selected_hashes.add(candidate.content_hash)
            selected_context_only += 1

    selected_records = tuple(candidate.selection_record for candidate in selected)
    actual_by_kind = dict(
        sorted(((kind.value, count) for kind, count in actual.items()))
    )
    target_by_kind = dict(
        sorted((kind.value, target) for kind, target in quotas.span_targets.items())
    )
    shortfall_by_kind = {
        kind: target - actual_by_kind.get(kind, 0)
        for kind, target in target_by_kind.items()
        if actual_by_kind.get(kind, 0) < target
    }
    overshoot_by_kind = {
        kind: actual_by_kind[kind] - target
        for kind, target in target_by_kind.items()
        if actual_by_kind.get(kind, 0) > target
    }
    source_counts = dict(
        sorted(Counter(record.source for record in selected_records).items())
    )

    return SelectionResult(
        selected_records,
        records_scanned,
        len(candidates),
        sum(reservoir.duplicates_removed for reservoir in reservoirs.values())
        + context_reservoir.duplicates_removed,
        len(selected_records),
        sum(actual.values()),
        selected_context_only,
        target_by_kind,
        actual_by_kind,
        shortfall_by_kind,
        overshoot_by_kind,
        sum(len(record.record.spans) > 1 for record in selected_records),
        source_counts,
    )
