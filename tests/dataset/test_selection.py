from premove_itn.dataset.records import TrainingRecord, TrainingSpan
from premove_itn.dataset.selection import (
    DatasetSelectionRecord,
    SelectionQuotas,
    select_candidates,
)
from premove_itn.labels import SpanKind
from premove_itn.types import WordToken


def _record(*kinds: SpanKind, prefix: str = "word") -> TrainingRecord:
    words = tuple(f"{prefix}{index}" for index in range(len(kinds) or 1))
    text = " ".join(words)
    tokens = tuple(
        WordToken(word, index * 6, index * 6 + len(word))
        for index, word in enumerate(words)
    )
    spans = tuple(
        TrainingSpan(
            kind,
            index * 6,
            index * 6 + len(words[index]),
            words[index],
            f"target{index}",
        )
        for index, kind in enumerate(kinds)
    )
    expected_words = list(words)
    for index in range(len(kinds)):
        expected_words[index] = f"target{index}"
    expected_text = " ".join(expected_words)
    labels = tuple(f"B-{kind.value}" for kind in kinds)
    if not kinds:
        labels = ("O",)
    return TrainingRecord(text, expected_text, spans, tokens, labels)


def _candidate(
    record: TrainingRecord,
    source_file: str,
    sentence_number: int,
    source: str = "test",
) -> DatasetSelectionRecord:
    return DatasetSelectionRecord(
        record=record,
        source=source,
        source_file=source_file,
        sentence_number=sentence_number,
    )


def test_selection_deduplicates_content_and_keeps_smallest_provenance() -> None:
    record = _record(SpanKind.TIME)
    result = select_candidates(
        (
            _candidate(record, "z.tsv", 2),
            _candidate(record, "a.tsv", 9),
        ),
        SelectionQuotas({SpanKind.TIME: 1}, context_only_records=0),
        seed="test",
    )

    assert result.selected_record_count == 1
    assert result.selected_records[0].source_file == "a.tsv"
    assert result.retained_unique_candidates == 1
    assert result.duplicates_removed == 1


def test_selection_uses_rare_first_multi_span_credit() -> None:
    multi_span = _candidate(_record(SpanKind.DATE, SpanKind.TIME), "source.tsv", 1)
    date_only = _candidate(_record(SpanKind.DATE), "source.tsv", 2)

    result = select_candidates(
        (date_only, multi_span),
        SelectionQuotas({SpanKind.DATE: 1, SpanKind.TIME: 1}, context_only_records=0),
        seed="test",
    )

    assert result.selected_records == (multi_span,)
    assert result.actual_by_kind == {"DATE": 1, "TIME": 1}
    assert result.multi_span_selected == 1


def test_selection_measures_targets_in_spans_not_records() -> None:
    result = select_candidates(
        (_candidate(_record(SpanKind.TIME, SpanKind.TIME), "source.tsv", 1),),
        SelectionQuotas({SpanKind.TIME: 2}, context_only_records=0),
        seed="test",
    )

    assert result.selected_record_count == 1
    assert result.selected_span_count == 2
    assert result.actual_by_kind == {"TIME": 2}
    assert result.overshoot_by_kind == {}


def test_selection_reports_overshoot_from_a_multi_span_record() -> None:
    result = select_candidates(
        (_candidate(_record(SpanKind.TIME, SpanKind.TIME), "source.tsv", 1),),
        SelectionQuotas({SpanKind.TIME: 1}, context_only_records=0),
        seed="test",
    )

    assert result.actual_by_kind == {"TIME": 2}
    assert result.shortfall_by_kind == {}
    assert result.overshoot_by_kind == {"TIME": 1}


def test_selection_keeps_context_quota_separate() -> None:
    context_one = _candidate(_record(prefix="first"), "source.tsv", 1)
    context_two = _candidate(_record(prefix="second"), "source.tsv", 2)
    span = _candidate(_record(SpanKind.DATE), "source.tsv", 3)

    result = select_candidates(
        (context_one, context_two, span),
        SelectionQuotas({SpanKind.DATE: 1}, context_only_records=2),
        seed="test",
    )

    assert result.selected_record_count == 3
    assert result.selected_context_only_count == 2
    assert result.actual_by_kind == {"DATE": 1}


def test_selection_is_deterministic_for_a_seed() -> None:
    records = tuple(
        _candidate(_record(SpanKind.CARDINAL), "source.tsv", number)
        for number in range(10)
    )
    quotas = SelectionQuotas({SpanKind.CARDINAL: 3}, context_only_records=0)

    first = select_candidates(records, quotas, seed="stable")
    second = select_candidates(reversed(records), quotas, seed="stable")

    assert first.selected_records == second.selected_records


def test_selection_reports_shortfalls_and_source_counts() -> None:
    result = select_candidates(
        (_candidate(_record(SpanKind.PHONE), "phone.tsv", 1, source="google_tn"),),
        SelectionQuotas(
            {SpanKind.PHONE: 2, SpanKind.ELECTRONIC: 1}, context_only_records=0
        ),
        seed="test",
    )

    assert result.shortfall_by_kind == {"ELECTRONIC": 1, "PHONE": 1}
    assert result.source_counts == {"google_tn": 1}
