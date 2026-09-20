from datetime import UTC, datetime

import pytest

from premove_itn import (
    NormalizationContext,
    NormalizationResult,
    NormalizedSpan,
    SpanKind,
)
from premove_itn.temporal import annotate_missing_year_dates, annotate_temporal


def test_relative_dates_are_annotated_and_resolved_from_reference_date() -> None:
    source = "today tomorrow yesterday day after tomorrow day before yesterday"
    result = annotate_temporal(
        source,
        NormalizationResult(source, source, ()),
        NormalizationContext(reference_datetime=datetime(2026, 9, 19, 12)),
    )

    assert tuple(span.source_text for span in result.spans) == (
        "today",
        "tomorrow",
        "yesterday",
        "day after tomorrow",
        "day before yesterday",
    )
    assert tuple(span.resolved_value for span in result.spans) == (
        "2026-09-19",
        "2026-09-20",
        "2026-09-18",
        "2026-09-21",
        "2026-09-17",
    )
    assert result.resolved_text == (
        "2026-09-19 2026-09-20 2026-09-18 2026-09-21 2026-09-17"
    )
    for span in result.spans:
        assert (
            result.text[span.normalized_start : span.normalized_end]
            == span.normalized_text
        )


def test_relative_dates_remain_unresolved_without_reference_datetime() -> None:
    result = annotate_temporal(
        "tomorrow",
        NormalizationResult("tomorrow", "tomorrow", ()),
        NormalizationContext(),
    )

    assert result.resolved_text == "tomorrow"
    assert result.spans == (
        NormalizedSpan(
            source_start=0,
            source_end=8,
            normalized_start=0,
            normalized_end=8,
            source_text="tomorrow",
            normalized_text="tomorrow",
            kinds=(SpanKind.DATE,),
            resolved_value=None,
        ),
    )


def test_relative_date_matching_respects_word_boundaries() -> None:
    result = annotate_temporal(
        "tomorrows tomorrow yesterdayday",
        NormalizationResult(
            "tomorrows tomorrow yesterdayday",
            "tomorrows tomorrow yesterdayday",
            (),
        ),
        NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert tuple(span.source_text for span in result.spans) == ("tomorrow",)
    assert result.resolved_text == "tomorrows 2026-09-20 yesterdayday"


def test_relative_date_matching_is_case_insensitive() -> None:
    result = annotate_temporal(
        "Tomorrow TODAY Day After Tomorrow",
        NormalizationResult(
            "Tomorrow TODAY Day After Tomorrow",
            "Tomorrow TODAY Day After Tomorrow",
            (),
        ),
        NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert tuple(span.source_text for span in result.spans) == (
        "Tomorrow",
        "TODAY",
        "Day After Tomorrow",
    )
    assert tuple(span.normalized_text for span in result.spans) == (
        "Tomorrow",
        "TODAY",
        "Day After Tomorrow",
    )
    assert result.resolved_text == "2026-09-20 2026-09-19 2026-09-21"


def test_timezone_converts_aware_reference_before_resolving_date() -> None:
    result = annotate_temporal(
        "today",
        NormalizationResult("today", "today", ()),
        NormalizationContext(
            reference_datetime=datetime(2026, 9, 19, 23, tzinfo=UTC),
            timezone="Asia/Kolkata",
        ),
    )

    assert result.spans[0].resolved_value == "2026-09-20"


@pytest.mark.parametrize(
    ("reference", "expected"),
    (
        (datetime(2026, 12, 31), "2027-01-01"),
        (datetime(2028, 2, 28), "2028-02-29"),
        (datetime(2027, 2, 28), "2027-03-01"),
    ),
)
def test_relative_date_resolution_handles_calendar_boundaries(
    reference, expected
) -> None:
    result = annotate_temporal(
        "tomorrow",
        NormalizationResult("tomorrow", "tomorrow", ()),
        NormalizationContext(reference_datetime=reference),
    )

    assert result.spans[0].resolved_value == expected


def test_invalid_timezone_is_rejected_when_temporal_resolution_runs() -> None:
    context = NormalizationContext(
        reference_datetime=datetime(2026, 9, 19),
        timezone="Not/A_Timezone",
    )
    unchanged = NormalizationResult("hello", "hello", ())
    assert annotate_temporal("hello", unchanged, context) is unchanged

    with pytest.raises(ValueError, match="unknown timezone"):
        annotate_temporal(
            "today",
            NormalizationResult("today", "today", ()),
            context,
        )


def test_contextual_span_offsets_follow_a_prior_length_change() -> None:
    result = annotate_temporal(
        "pay twenty dollars tomorrow",
        NormalizationResult(
            text="pay $20 tomorrow",
            resolved_text="pay $20 tomorrow",
            spans=(
                NormalizedSpan(
                    source_start=4,
                    source_end=18,
                    normalized_start=4,
                    normalized_end=7,
                    source_text="twenty dollars",
                    normalized_text="$20",
                    kinds=(SpanKind.MONEY,),
                ),
            ),
        ),
        NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    tomorrow = result.spans[1]
    assert tomorrow.source_start == 19
    assert tomorrow.normalized_start == 8
    assert tomorrow.normalized_text == "tomorrow"
    assert result.resolved_text == "pay $20 2026-09-20"


def test_contextual_span_offsets_follow_implicit_space_deletion() -> None:
    result = annotate_temporal(
        "five percent tomorrow",
        NormalizationResult(
            text="5% tomorrow",
            resolved_text="5% tomorrow",
            spans=(
                NormalizedSpan(
                    source_start=0,
                    source_end=4,
                    normalized_start=0,
                    normalized_end=1,
                    source_text="five",
                    normalized_text="5",
                    kinds=(SpanKind.CARDINAL,),
                ),
                NormalizedSpan(
                    source_start=5,
                    source_end=12,
                    normalized_start=1,
                    normalized_end=2,
                    source_text="percent",
                    normalized_text="%",
                    kinds=(SpanKind.MEASUREMENT,),
                ),
            ),
        ),
        NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    tomorrow = result.spans[2]
    assert tomorrow.normalized_start == 3
    assert result.resolved_text == "5% 2026-09-20"


@pytest.mark.parametrize(
    "normalized_text",
    (
        "September 30",
        "30 September",
        "the 30th of September",
        "Sep. 30",
    ),
)
def test_missing_year_named_month_dates_use_reference_year(normalized_text) -> None:
    span = NormalizedSpan(
        source_start=0,
        source_end=len(normalized_text),
        normalized_start=0,
        normalized_end=len(normalized_text),
        source_text=normalized_text,
        normalized_text=normalized_text,
        kinds=(SpanKind.DATE,),
    )
    result = annotate_missing_year_dates(
        NormalizationResult(normalized_text, normalized_text, (span,)),
        NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert result.spans[0].resolved_value == "2026-09-30"
    assert result.resolved_text == "2026-09-30"


def test_missing_year_date_stays_unresolved_without_reference_year() -> None:
    span = NormalizedSpan(
        0, 10, 0, 10, "September 30", "September 30", (SpanKind.DATE,)
    )
    result = annotate_missing_year_dates(
        NormalizationResult("September 30", "September 30", (span,)),
        NormalizationContext(),
    )

    assert result.spans == (span,)
    assert result.resolved_text == "September 30"


def test_weekday_prefixed_named_date_remains_unresolved() -> None:
    text = "Thursday, September 30"
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_missing_year_dates(
        NormalizationResult(text, text, (span,)),
        NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert result.spans == (span,)
    assert result.resolved_text == text


def test_missing_year_uses_reference_year_even_when_date_has_passed() -> None:
    text = "September 30"
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_missing_year_dates(
        NormalizationResult(text, text, (span,)),
        NormalizationContext(reference_datetime=datetime(2026, 10, 20)),
    )

    assert result.spans[0].resolved_value == "2026-09-30"
    assert result.resolved_text == "2026-09-30"


def test_missing_year_uses_timezone_local_reference_year() -> None:
    text = "September 30"
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_missing_year_dates(
        NormalizationResult(text, text, (span,)),
        NormalizationContext(
            reference_datetime=datetime(2026, 12, 31, 23, tzinfo=UTC),
            timezone="Asia/Kolkata",
        ),
    )

    assert result.spans[0].resolved_value == "2027-09-30"
    assert result.resolved_text == "2027-09-30"


@pytest.mark.parametrize(
    "context",
    (None, NormalizationContext(reference_datetime=datetime(2026, 9, 19))),
)
def test_explicit_year_resolves_without_or_against_context(context) -> None:
    text = "September 30 2027"
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_missing_year_dates(
        NormalizationResult(text, text, (span,)),
        context,
    )

    assert result.spans[0].resolved_value == "2027-09-30"
    assert result.resolved_text == "2027-09-30"


@pytest.mark.parametrize(
    ("text", "reference", "expected"),
    (
        ("February 29", datetime(2027, 2, 28), None),
        ("February 29", datetime(2028, 2, 28), "2028-02-29"),
        ("April 31", datetime(2026, 4, 1), None),
    ),
)
def test_missing_year_date_rejects_invalid_calendar_values(
    text, reference, expected
) -> None:
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_missing_year_dates(
        NormalizationResult(text, text, (span,)),
        NormalizationContext(reference_datetime=reference),
    )

    assert result.spans[0].resolved_value == expected
    assert result.resolved_text == (expected or text)


def test_non_date_span_is_not_enriched_as_a_calendar_date() -> None:
    text = "September 30"
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.WORD,))
    result = annotate_missing_year_dates(
        NormalizationResult(text, text, (span,)),
        NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert result == NormalizationResult(text, text, (span,))


def test_relative_date_enriches_an_existing_compatible_date_span() -> None:
    existing = NormalizedSpan(
        source_start=0,
        source_end=8,
        normalized_start=0,
        normalized_end=8,
        source_text="tomorrow",
        normalized_text="tomorrow",
        kinds=(SpanKind.DATE,),
    )
    result = annotate_temporal(
        "tomorrow",
        NormalizationResult("tomorrow", "tomorrow", (existing,)),
        NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert len(result.spans) == 1
    assert result.spans[0].resolved_value == "2026-09-20"
    assert result.resolved_text == "2026-09-20"


def test_relative_date_does_not_override_an_incompatible_overlap() -> None:
    existing = NormalizedSpan(
        source_start=0,
        source_end=8,
        normalized_start=0,
        normalized_end=3,
        source_text="tomorrow",
        normalized_text="TMR",
        kinds=(SpanKind.WORD,),
    )
    result = annotate_temporal(
        "tomorrow",
        NormalizationResult("TMR", "TMR", (existing,)),
        NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert result.spans == (existing,)
    assert result.resolved_text == "TMR"
