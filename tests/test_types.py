from dataclasses import FrozenInstanceError

import pytest

from premove_itn.labels import SpanKind
from premove_itn.types import (
    NormalizationResult,
    NormalizedEdit,
    TaggedSpan,
    TextSpan,
    WordPrediction,
    WordToken,
)


def test_source_coordinates_are_immutable_half_open_spans() -> None:
    token = WordToken(text="four", start=11, end=15)
    span = TextSpan(start=11, end=22)

    assert token.text == "four"
    assert (token.start, token.end) == (11, 15)
    assert (span.start, span.end) == (11, 22)
    with pytest.raises(FrozenInstanceError):
        token.start = 12  # type: ignore[misc]


@pytest.mark.parametrize(
    ("constructor", "kwargs"),
    [
        (WordToken, {"text": "four", "start": -1, "end": 3}),
        (WordToken, {"text": "", "start": 0, "end": 0}),
        (WordToken, {"text": "four", "start": 5, "end": 4}),
        (WordToken, {"text": "four", "start": 0, "end": 3}),
        (TextSpan, {"start": -1, "end": 1}),
        (TextSpan, {"start": 2, "end": 2}),
        (TextSpan, {"start": 3, "end": 2}),
    ],
)
def test_source_coordinates_reject_invalid_offsets(
    constructor: type[WordToken] | type[TextSpan], kwargs: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        constructor(**kwargs)  # type: ignore[arg-type]


def test_word_prediction_keeps_the_source_token_and_model_evidence() -> None:
    token = WordToken(text="four", start=11, end=15)

    prediction = WordPrediction(token=token, label="B-TIME", score=0.875)

    assert prediction == WordPrediction(token=token, label="B-TIME", score=0.875)


@pytest.mark.parametrize("score", [-0.01, 1.01, float("inf"), float("nan")])
def test_word_prediction_rejects_invalid_scores(score: float) -> None:
    token = WordToken(text="four", start=11, end=15)

    with pytest.raises(ValueError):
        WordPrediction(token=token, label="B-TIME", score=score)


def test_word_prediction_rejects_unknown_labels() -> None:
    token = WordToken(text="four", start=11, end=15)

    with pytest.raises(ValueError):
        WordPrediction(token=token, label="B-UNKNOWN", score=0.9)
    with pytest.raises(ValueError):
        WordPrediction(token=token, label="B-WORD", score=0.9)


def test_tagged_span_preserves_classifier_provenance() -> None:
    tagged_span = TaggedSpan(
        kind=SpanKind.TIME,
        source_text="four thirty",
        span=TextSpan(start=11, end=22),
        score=0.91,
    )

    assert tagged_span.source_text == "four thirty"
    assert tagged_span.span == TextSpan(start=11, end=22)
    assert tagged_span.kind is SpanKind.TIME


def test_normalized_edit_preserves_source_and_replacement_provenance() -> None:
    edit = NormalizedEdit(
        kind=SpanKind.TIME,
        source_text="four thirty",
        source_span=TextSpan(start=11, end=22),
        normalized_text="04:30",
        score=0.91,
    )

    assert edit.source_text == "four thirty"
    assert edit.source_span == TextSpan(start=11, end=22)
    assert edit.normalized_text == "04:30"


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan")])
def test_span_and_edit_reject_invalid_scores(score: float) -> None:
    source_span = TextSpan(start=11, end=22)

    with pytest.raises(ValueError):
        TaggedSpan(SpanKind.TIME, "four thirty", source_span, score)
    with pytest.raises(ValueError):
        NormalizedEdit(SpanKind.TIME, "four thirty", source_span, "04:30", score)


def test_span_and_edit_reject_source_text_that_does_not_match_span_length() -> None:
    source_span = TextSpan(start=11, end=22)

    with pytest.raises(ValueError):
        TaggedSpan(SpanKind.TIME, "", source_span, 0.9)
    with pytest.raises(ValueError):
        NormalizedEdit(SpanKind.TIME, "four", source_span, "4", 0.9)


def test_span_and_edit_reject_unknown_kinds() -> None:
    source_span = TextSpan(start=11, end=22)

    with pytest.raises(ValueError):
        TaggedSpan("UNKNOWN", "four thirty", source_span, 0.9)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NormalizedEdit(  # type: ignore[arg-type]
            "UNKNOWN", "four thirty", source_span, "04:30", 0.9
        )


def test_normalization_result_preserves_ordered_original_source_provenance() -> None:
    edit = NormalizedEdit(
        kind=SpanKind.TIME,
        source_text="four thirty",
        source_span=TextSpan(start=11, end=22),
        normalized_text="04:30",
        score=0.91,
    )

    result = NormalizationResult(
        source_text="meet me at four thirty",
        normalized_text="meet me at 04:30",
        edits=(edit,),
    )

    assert result.source_text[11:22] == result.edits[0].source_text
    assert result.normalized_text == "meet me at 04:30"


def test_normalization_result_rejects_incorrect_source_provenance() -> None:
    edit = NormalizedEdit(
        SpanKind.TIME,
        "five thirty",
        TextSpan(11, 22),
        "05:30",
        0.9,
    )

    with pytest.raises(ValueError):
        NormalizationResult("meet me at four thirty", "meet me at 05:30", (edit,))


def test_normalization_result_rejects_edits_outside_the_original_text() -> None:
    edit = NormalizedEdit(
        SpanKind.TIME,
        "four thirty",
        TextSpan(30, 41),
        "04:30",
        0.9,
    )

    with pytest.raises(ValueError):
        NormalizationResult("meet me at four thirty", "meet me at 04:30", (edit,))


def test_normalization_result_rejects_overlapping_or_unordered_edits() -> None:
    first = NormalizedEdit(SpanKind.CARDINAL, "one", TextSpan(0, 3), "1", 0.9)
    overlapping = NormalizedEdit(SpanKind.WORD, "e two", TextSpan(2, 7), "e 2", 0.8)

    with pytest.raises(ValueError):
        NormalizationResult("one two", "1 2", (first, overlapping))
    with pytest.raises(ValueError):
        NormalizationResult("one two", "1 2", (overlapping, first))


def test_normalization_result_rejects_text_inconsistent_with_edits() -> None:
    edit = NormalizedEdit(
        SpanKind.TIME,
        "four thirty",
        TextSpan(11, 22),
        "04:30",
        0.9,
    )

    with pytest.raises(ValueError):
        NormalizationResult("meet me at four thirty", "meet me at four thirty", (edit,))
    with pytest.raises(ValueError):
        NormalizationResult("unchanged", "changed", ())
