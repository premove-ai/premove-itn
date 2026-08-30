"""Exact maximum-score decoding over deterministic candidate intervals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from premove_itn.candidates import Candidate
from premove_itn.structured_loss import max_path_indices


@dataclass(frozen=True, slots=True)
class DecodedPath:
    """One highest-scoring path and its rendered source replacement."""

    text: str
    selected_candidates: tuple[Candidate, ...]
    score: float


def apply_candidate_replacements(
    text: str,
    candidates: Sequence[Candidate],
) -> str:
    """Apply non-overlapping candidate replacements in source order."""
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.char_start))
    pieces: list[str] = []
    cursor = 0
    for candidate in ordered:
        if candidate.char_start < cursor:
            raise ValueError("selected candidates overlap")
        if (
            candidate.char_start < 0
            or candidate.char_end > len(text)
            or candidate.char_end <= candidate.char_start
        ):
            raise ValueError("selected candidate span is outside the source")
        if text[candidate.char_start : candidate.char_end] != candidate.text:
            raise ValueError("selected candidate text does not match the source")
        pieces.append(text[cursor : candidate.char_start])
        pieces.append(candidate.replacement)
        cursor = candidate.char_end
    pieces.append(text[cursor:])
    return "".join(pieces)


def decode_candidates(
    text: str,
    candidates: Sequence[Candidate],
    candidate_scores: torch.Tensor,
) -> DecodedPath:
    """Decode one sentence with exact max-sum interval dynamic programming."""
    if candidate_scores.ndim != 1:
        raise ValueError("candidate scores must be one-dimensional")
    if candidate_scores.shape[0] != len(candidates):
        raise ValueError("candidate scores and candidates must have equal length")

    candidate_char_spans = torch.tensor(
        [(candidate.char_start, candidate.char_end) for candidate in candidates],
        dtype=torch.long,
        device=candidate_scores.device,
    ).reshape(-1, 2)
    selected_indices = max_path_indices(
        candidate_scores,
        candidate_char_spans,
        source_char_count=len(text),
    )
    selected_candidates = tuple(candidates[index] for index in selected_indices)
    score_values = tuple(
        float(score) for score in candidate_scores.detach().cpu().tolist()
    )
    return DecodedPath(
        text=apply_candidate_replacements(text, selected_candidates),
        selected_candidates=selected_candidates,
        score=sum(score_values[index] for index in selected_indices),
    )
