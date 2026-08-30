import pytest

pytest.importorskip("torch")

import torch

from premove_itn import Candidate, SpanKind
from premove_itn.decoder import apply_candidate_replacements, decode_candidates


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


def test_decode_candidates_rejects_wrong_score_shape() -> None:
    candidate = _candidate(0, 5, "seven", "7")

    with pytest.raises(ValueError, match="equal length"):
        decode_candidates("seven", (candidate,), torch.empty(0))
