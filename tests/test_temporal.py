from datetime import UTC, date, datetime, timedelta

import pytest

from premove_itn import (
    DateOrder,
    NormalizationContext,
    NormalizationResult,
    NormalizedSpan,
    SpanKind,
)
from premove_itn.temporal import (
    annotate_missing_year_dates,
    annotate_numeric_dates,
    annotate_temporal,
)


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


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("the day after tomorrow", "2026-09-21"),
        ("the day before yesterday", "2026-09-17"),
    ),
)
def test_article_prefixed_relative_dates_are_one_span(text, expected) -> None:
    result = annotate_temporal(
        text,
        NormalizationResult(text, text, ()),
        NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert tuple(span.source_text for span in result.spans) == (text,)
    assert result.spans[0].resolved_value == expected
    assert result.resolved_text == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("in two days", "2026-09-21"),
        ("two days from now", "2026-09-21"),
        ("two days ago", "2026-09-17"),
        ("in one week", "2026-09-26"),
        ("in two weeks", "2026-10-03"),
        ("one week from today", "2026-09-26"),
        ("a week ago", "2026-09-12"),
        ("in 2 days", "2026-09-21"),
        ("10 days from today", "2026-09-29"),
    ),
)
def test_bounded_relative_day_and_week_offsets_resolve(text, expected) -> None:
    result = annotate_temporal(
        text,
        NormalizationResult(text, text, ()),
        NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert tuple(span.source_text for span in result.spans) == (text,)
    assert result.spans[0].resolved_value == expected
    assert result.resolved_text == expected


def test_relative_offset_does_not_add_a_nested_today_span() -> None:
    text = "one week from today"
    result = annotate_temporal(
        text,
        NormalizationResult(text, text, ()),
        NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert len(result.spans) == 1
    assert result.spans[0].source_text == text


def test_bounded_relative_offset_stays_unresolved_without_reference_datetime() -> None:
    text = "two days ago"
    result = annotate_temporal(
        text,
        NormalizationResult(text, text, ()),
        NormalizationContext(),
    )

    assert result.spans[0].resolved_value is None
    assert result.resolved_text == text


@pytest.mark.parametrize("text", ("in 0 days", "in 11 days", "in 999999 days"))
def test_relative_offset_rejects_numeric_counts_outside_the_bound(text) -> None:
    unchanged = NormalizationResult(text, text, ())

    assert (
        annotate_temporal(
            text,
            unchanged,
            NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
        )
        is unchanged
    )


@pytest.mark.parametrize(
    ("weekday", "weekday_index"),
    (
        ("Monday", 0),
        ("Tuesday", 1),
        ("Wednesday", 2),
        ("Thursday", 3),
        ("Friday", 4),
        ("Saturday", 5),
        ("Sunday", 6),
    ),
)
def test_weekday_relative_dates_use_calendar_week(weekday, weekday_index) -> None:
    reference = date(2026, 9, 23)
    monday = reference - timedelta(days=reference.weekday())
    for modifier, week_offset in (("last", -7), ("this", 0), ("next", 7)):
        text = f"{modifier} {weekday}"
        result = annotate_temporal(
            text,
            NormalizationResult(text, text, ()),
            NormalizationContext(
                reference_datetime=datetime.combine(reference, datetime.min.time())
            ),
        )

        expected = monday + timedelta(days=weekday_index + week_offset)
        assert len(result.spans) == 1
        assert result.spans[0].resolved_value == expected.isoformat()
        assert result.resolved_text == expected.isoformat()


def test_weekday_relative_matching_is_case_insensitive() -> None:
    text = "NEXT monday"
    result = annotate_temporal(
        text,
        NormalizationResult(text, text, ()),
        NormalizationContext(reference_datetime=datetime(2026, 9, 23)),
    )

    assert result.spans[0].source_text == text
    assert result.spans[0].resolved_value == "2026-09-28"


def test_weekday_relative_date_stays_unresolved_without_reference_datetime() -> None:
    text = "next Monday"
    result = annotate_temporal(
        text,
        NormalizationResult(text, text, ()),
        NormalizationContext(),
    )

    assert result.spans[0].resolved_value is None
    assert result.resolved_text == text


def test_weekday_relative_date_uses_timezone_local_reference_date() -> None:
    result = annotate_temporal(
        "this Monday",
        NormalizationResult("this Monday", "this Monday", ()),
        NormalizationContext(
            reference_datetime=datetime(2026, 9, 20, 23, tzinfo=UTC),
            timezone="Asia/Kolkata",
        ),
    )

    assert result.spans[0].resolved_value == "2026-09-21"


def test_interval_relative_expression_remains_deferred() -> None:
    text = "next quarter"
    unchanged = NormalizationResult(text, text, ())

    assert (
        annotate_temporal(
            text,
            unchanged,
            NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
        )
        is unchanged
    )


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


def test_weekday_prefixed_named_date_validates_against_reference_year() -> None:
    text = "Thursday, September 30"
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_missing_year_dates(
        NormalizationResult(text, text, (span,)),
        NormalizationContext(reference_datetime=datetime(2027, 1, 1)),
    )

    assert result.spans[0].resolved_value == "2027-09-30"
    assert result.resolved_text == "2027-09-30"


def test_weekday_prefixed_named_date_remains_unresolved_when_contradictory() -> None:
    text = "Thursday, September 30"
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_missing_year_dates(
        NormalizationResult(text, text, (span,)),
        NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert result.spans == (span,)
    assert result.resolved_text == text


def test_weekday_prefixed_named_date_uses_timezone_local_reference_year() -> None:
    text = "Thursday, September 30"
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
    "text", ("Thursday, September 30 2027", "Thursday, September 30, 2027")
)
def test_weekday_prefixed_named_date_with_explicit_year_resolves_without_context(
    text,
) -> None:
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_missing_year_dates(
        NormalizationResult(text, text, (span,)),
        None,
    )

    assert result.spans[0].resolved_value == "2027-09-30"
    assert result.resolved_text == "2027-09-30"


def test_weekday_prefixed_named_date_with_explicit_year_rejects_contradiction() -> None:
    text = "Thursday, September 30, 2026"
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_missing_year_dates(
        NormalizationResult(text, text, (span,)),
        None,
    )

    assert result.spans == (span,)
    assert result.resolved_text == text


def test_weekday_prefixed_named_date_needs_reference_year_when_missing() -> None:
    text = "Thursday, September 30"
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_missing_year_dates(
        NormalizationResult(text, text, (span,)),
        NormalizationContext(),
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
    ("text", "context", "expected"),
    (
        ("24/09/2026", NormalizationContext(), "2026-09-24"),
        ("24-09-2026", NormalizationContext(), "2026-09-24"),
        ("24.09.2026", NormalizationContext(), "2026-09-24"),
        ("24 09 2026", NormalizationContext(), "2026-09-24"),
        (
            "24/09/2026",
            NormalizationContext(date_order=DateOrder.MDY),
            "2026-09-24",
        ),
    ),
)
def test_structurally_unambiguous_numeric_dates_resolve_without_order(
    text, context, expected
) -> None:
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_numeric_dates(
        NormalizationResult(text, text, (span,)),
        context,
    )

    assert result.spans[0].resolved_value == expected
    assert result.resolved_text == expected


@pytest.mark.parametrize(
    ("text", "date_order", "expected"),
    (
        ("03/04/2026", DateOrder.DMY, "2026-04-03"),
        ("03/04/2026", DateOrder.MDY, "2026-03-04"),
        ("03/2026/04", DateOrder.DYM, "2026-04-03"),
        ("03/2026/04", DateOrder.MYD, "2026-03-04"),
        ("2026/03/04", DateOrder.YDM, "2026-04-03"),
        ("2026/03/04", DateOrder.YMD, "2026-03-04"),
    ),
)
def test_ambiguous_numeric_dates_use_date_order(text, date_order, expected) -> None:
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_numeric_dates(
        NormalizationResult(text, text, (span,)),
        NormalizationContext(date_order=date_order),
    )

    assert result.spans[0].resolved_value == expected
    assert result.resolved_text == expected


def test_ambiguous_numeric_date_stays_unresolved_without_order() -> None:
    text = "03/04/2026"
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_numeric_dates(
        NormalizationResult(text, text, (span,)),
        NormalizationContext(),
    )

    assert result.spans == (span,)
    assert result.resolved_text == text


@pytest.mark.parametrize(
    ("text", "context", "expected"),
    (
        (
            "24/09",
            NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
            "2026-09-24",
        ),
        (
            "03/04",
            NormalizationContext(
                reference_datetime=datetime(2026, 9, 19), date_order=DateOrder.DMY
            ),
            "2026-04-03",
        ),
        (
            "03/04",
            NormalizationContext(
                reference_datetime=datetime(2026, 9, 19), date_order=DateOrder.DYM
            ),
            "2026-04-03",
        ),
        (
            "03/04",
            NormalizationContext(
                reference_datetime=datetime(2026, 9, 19), date_order=DateOrder.YDM
            ),
            "2026-04-03",
        ),
        (
            "03/04",
            NormalizationContext(
                reference_datetime=datetime(2026, 9, 19), date_order=DateOrder.MDY
            ),
            "2026-03-04",
        ),
        (
            "03/04",
            NormalizationContext(
                reference_datetime=datetime(2026, 9, 19), date_order=DateOrder.MYD
            ),
            "2026-03-04",
        ),
        (
            "03/04",
            NormalizationContext(
                reference_datetime=datetime(2026, 9, 19), date_order=DateOrder.YMD
            ),
            "2026-03-04",
        ),
    ),
)
def test_numeric_dates_without_year_use_reference_year(text, context, expected) -> None:
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_numeric_dates(
        NormalizationResult(text, text, (span,)),
        context,
    )

    assert result.spans[0].resolved_value == expected
    assert result.resolved_text == expected


def test_numeric_date_without_year_stays_unresolved_without_reference() -> None:
    text = "24/09"
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_numeric_dates(
        NormalizationResult(text, text, (span,)),
        NormalizationContext(date_order=DateOrder.DMY),
    )

    assert result.spans == (span,)
    assert result.resolved_text == text


@pytest.mark.parametrize("text", ("03/04/26", "31/02/2026", "03/04/2026."))
def test_numeric_dates_with_unsupported_or_invalid_values_stay_unresolved(text) -> None:
    span = NormalizedSpan(0, len(text), 0, len(text), text, text, (SpanKind.DATE,))
    result = annotate_numeric_dates(
        NormalizationResult(text, text, (span,)),
        NormalizationContext(date_order=DateOrder.DMY),
    )

    assert result.spans == (span,)
    assert result.resolved_text == text


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


def test_relative_offset_composes_with_a_contained_number_edit() -> None:
    source = "in two days"
    number_span = NormalizedSpan(
        source_start=3,
        source_end=6,
        normalized_start=3,
        normalized_end=4,
        source_text="two",
        normalized_text="2",
        kinds=(SpanKind.CARDINAL,),
    )
    result = annotate_temporal(
        source,
        NormalizationResult("in 2 days", "in 2 days", (number_span,)),
        NormalizationContext(reference_datetime=datetime(2026, 9, 19)),
    )

    assert result.text == "in 2 days"
    assert result.resolved_text == "2026-09-21"
    assert result.spans[0].source_text == source
    assert result.spans[0].normalized_text == "in 2 days"
    assert result.spans[0].kinds == (SpanKind.DATE,)
    assert result.spans[0].resolved_value == "2026-09-21"
    assert result.spans[1] == number_span


def test_temporal_resolvers_compose_multiple_date_forms() -> None:
    source = "tomorrow September 30 03/04/2026 next Monday"
    spans = (
        NormalizedSpan(
            source_start=9,
            source_end=21,
            normalized_start=9,
            normalized_end=21,
            source_text="September 30",
            normalized_text="September 30",
            kinds=(SpanKind.DATE,),
        ),
        NormalizedSpan(
            source_start=22,
            source_end=32,
            normalized_start=22,
            normalized_end=32,
            source_text="03/04/2026",
            normalized_text="03/04/2026",
            kinds=(SpanKind.DATE,),
        ),
    )
    context = NormalizationContext(
        reference_datetime=datetime(2026, 9, 23),
        date_order=DateOrder.DMY,
    )
    result = annotate_temporal(
        source, NormalizationResult(source, source, spans), context
    )
    result = annotate_missing_year_dates(result, context)
    result = annotate_numeric_dates(result, context)

    assert tuple(span.resolved_value for span in result.spans) == (
        "2026-09-24",
        "2026-09-30",
        "2026-04-03",
        "2026-09-28",
    )
    assert result.resolved_text == ("2026-09-24 2026-09-30 2026-04-03 2026-09-28")


def test_resolved_text_is_rendered_in_normalized_position_order() -> None:
    source = "September 30 October 1"
    october_span = NormalizedSpan(
        source_start=13,
        source_end=22,
        normalized_start=13,
        normalized_end=22,
        source_text="October 1",
        normalized_text="October 1",
        kinds=(SpanKind.DATE,),
    )
    september_span = NormalizedSpan(
        source_start=0,
        source_end=12,
        normalized_start=0,
        normalized_end=12,
        source_text="September 30",
        normalized_text="September 30",
        kinds=(SpanKind.DATE,),
    )
    result = annotate_missing_year_dates(
        NormalizationResult(source, source, (october_span, september_span)),
        NormalizationContext(reference_datetime=datetime(2026, 9, 23)),
    )

    assert result.spans[0].source_text == "October 1"
    assert result.spans[1].source_text == "September 30"
    assert result.resolved_text == "2026-09-30 2026-10-01"


def test_temporal_composition_uses_timezone_local_year_boundary() -> None:
    source = "today tomorrow September 30"
    span = NormalizedSpan(
        source_start=15,
        source_end=27,
        normalized_start=15,
        normalized_end=27,
        source_text="September 30",
        normalized_text="September 30",
        kinds=(SpanKind.DATE,),
    )
    context = NormalizationContext(
        reference_datetime=datetime(2026, 12, 31, 23, tzinfo=UTC),
        timezone="Asia/Kolkata",
    )
    result = annotate_temporal(
        source,
        NormalizationResult(source, source, (span,)),
        context,
    )
    result = annotate_missing_year_dates(result, context)

    assert result.resolved_text == "2027-01-01 2027-01-02 2027-09-30"


def test_temporal_resolvers_compose_after_a_length_changing_date_edit() -> None:
    source = "tomorrow september thirtieth next Monday"
    result = NormalizationResult(
        text="tomorrow september 30 next Monday",
        resolved_text="tomorrow september 30 next Monday",
        spans=(
            NormalizedSpan(
                source_start=9,
                source_end=28,
                normalized_start=9,
                normalized_end=21,
                source_text="september thirtieth",
                normalized_text="september 30",
                kinds=(SpanKind.DATE,),
            ),
        ),
    )
    context = NormalizationContext(reference_datetime=datetime(2026, 9, 23))

    result = annotate_temporal(source, result, context)
    result = annotate_missing_year_dates(result, context)

    assert tuple(
        (span.source_text, span.normalized_start, span.normalized_end)
        for span in result.spans
    ) == (
        ("tomorrow", 0, 8),
        ("september thirtieth", 9, 21),
        ("next Monday", 22, 33),
    )
    assert result.resolved_text == "2026-09-24 2026-09-30 2026-09-28"


def test_temporal_resolution_preserves_an_existing_resolved_date() -> None:
    existing = NormalizedSpan(
        source_start=0,
        source_end=8,
        normalized_start=0,
        normalized_end=9,
        source_text="tomorrow",
        normalized_text="tomorrow!",
        kinds=(SpanKind.DATE,),
        resolved_value="2030-01-01",
    )
    result = annotate_temporal(
        "tomorrow",
        NormalizationResult("tomorrow!", "tomorrow!", (existing,)),
        NormalizationContext(reference_datetime=datetime(2026, 9, 23)),
    )

    assert result.spans == (existing,)
    assert result.resolved_text == "2030-01-01"
