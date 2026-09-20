import pytest

pytest.importorskip("torch")

import torch

from premove_itn import Candidate, SpanKind
from premove_itn.candidates import build_candidate_graph
from premove_itn.decoder import (
    apply_candidate_replacements,
    decode_candidates,
    render_candidate_replacements,
)


def _candidate(start: int, end: int, text: str, replacement: str) -> Candidate:
    return Candidate(
        token_start=0,
        token_end=1,
        char_start=start,
        char_end=end,
        text=text,
        replacement=replacement,
        kinds=(SpanKind.CARDINAL,),
    )


def test_decode_candidates_applies_highest_scoring_legal_path() -> None:
    text = "booking id seven three"
    candidates = (
        _candidate(11, 16, "seven", "7"),
        _candidate(17, 22, "three", "3"),
        _candidate(11, 22, "seven three", "73"),
    )

    decoded = decode_candidates(text, candidates, torch.tensor([0.8, 0.6, 2.1]))

    assert decoded.text == "booking id 73"
    assert decoded.selected_candidates == (candidates[2],)
    assert decoded.score == pytest.approx(2.1)


def test_decode_candidates_keeps_source_when_all_edits_score_negative() -> None:
    text = "seven"
    candidate = _candidate(0, 5, text, "7")

    decoded = decode_candidates(text, (candidate,), torch.tensor([-1.0]))

    assert decoded.text == text
    assert decoded.selected_candidates == ()
    assert decoded.score == 0


def test_apply_candidate_replacements_rejects_overlap() -> None:
    first = _candidate(0, 5, "seven", "7")
    second = _candidate(3, 8, "en thr", "x")

    with pytest.raises(ValueError, match="overlap"):
        apply_candidate_replacements("seven three", (first, second))


def test_apply_candidate_replacements_preserves_gold_implicit_spacing() -> None:
    text = "the rate five percent"
    candidates = build_candidate_graph(text)
    five = next(
        candidate
        for candidate in candidates
        if candidate.text == "five" and candidate.replacement == "5"
    )
    percent = next(
        candidate
        for candidate in candidates
        if candidate.text == "percent" and candidate.replacement == "%"
    )

    assert apply_candidate_replacements(text, (five, percent)) == "the rate 5%"


def test_render_candidate_replacements_returns_exact_normalized_intervals() -> None:
    text = "pay twenty dollars at four thirty"
    money = _candidate(4, 18, "twenty dollars", "$20")
    time = _candidate(22, 33, "four thirty", "04:30")

    rendered = render_candidate_replacements(text, (time, money))

    assert rendered.text == "pay $20 at 04:30"
    assert tuple(span.candidate for span in rendered.spans) == (money, time)
    assert tuple(
        rendered.text[span.normalized_start : span.normalized_end]
        for span in rendered.spans
    ) == ("$20", "04:30")
    assert tuple(
        (span.normalized_start, span.normalized_end) for span in rendered.spans
    ) == ((4, 7), (11, 16))


def test_render_candidate_replacements_tracks_adjacent_candidates() -> None:
    text = "one two"
    one = _candidate(0, 3, "one", "1")
    two = _candidate(4, 7, "two", "2")

    rendered = render_candidate_replacements(text, (one, two))

    assert rendered.text == "1 2"
    assert tuple(
        (span.normalized_start, span.normalized_end) for span in rendered.spans
    ) == ((0, 1), (2, 3))


def test_render_candidate_replacements_tracks_implicit_space_deletion() -> None:
    text = "the rate five percent"
    candidates = build_candidate_graph(text)
    five = next(
        candidate
        for candidate in candidates
        if candidate.text == "five" and candidate.replacement == "5"
    )
    percent = next(
        candidate
        for candidate in candidates
        if candidate.text == "percent" and candidate.replacement == "%"
    )

    rendered = render_candidate_replacements(text, (five, percent))

    assert rendered.text == "the rate 5%"
    assert tuple(
        rendered.text[span.normalized_start : span.normalized_end]
        for span in rendered.spans
    ) == ("5", "%")
    assert tuple(
        (span.normalized_start, span.normalized_end) for span in rendered.spans
    ) == ((9, 10), (10, 11))


def test_render_candidate_replacements_preserves_candidate_kinds() -> None:
    candidate = Candidate(
        token_start=0,
        token_end=3,
        char_start=0,
        char_end=11,
        text="one oh five",
        replacement="105",
        kinds=(SpanKind.CARDINAL, SpanKind.DIGIT_SEQUENCE),
    )

    rendered = render_candidate_replacements("one oh five", (candidate,))

    assert rendered.spans[0].candidate.kinds == (
        SpanKind.CARDINAL,
        SpanKind.DIGIT_SEQUENCE,
    )


def test_render_candidate_replacements_handles_no_candidates() -> None:
    rendered = render_candidate_replacements("leave this unchanged", ())

    assert rendered.text == "leave this unchanged"
    assert rendered.spans == ()


def test_decode_candidates_rejects_wrong_score_shape() -> None:
    candidate = _candidate(0, 5, "seven", "7")

    with pytest.raises(ValueError, match="equal length"):
        decode_candidates("seven", (candidate,), torch.empty(0))
