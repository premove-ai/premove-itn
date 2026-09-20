"""Exact maximum-score decoding over deterministic candidate intervals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from premove_itn.candidates import Candidate, is_implicit_space_transition
from premove_itn.structured_loss import max_path_indices


@dataclass(frozen=True, slots=True)
class DecodedPath:
    """One highest-scoring path and its rendered source replacement."""

    text: str
    selected_candidates: tuple[Candidate, ...]
    score: float


@dataclass(frozen=True, slots=True)
class RenderedSpan:
    """One selected candidate and its half-open normalized-text interval."""

    candidate: Candidate
    normalized_start: int
    normalized_end: int


@dataclass(frozen=True, slots=True)
class RenderedReplacements:
    """Text and exact normalized intervals from one replacement render."""

    text: str
    spans: tuple[RenderedSpan, ...]


def render_candidate_replacements(
    text: str,
    candidates: Sequence[Candidate],
) -> RenderedReplacements:
    """Render non-overlapping candidates and retain their output intervals."""
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.char_start))
    pieces: list[str] = []
    spans: list[RenderedSpan] = []
    cursor = 0
    rendered_length = 0
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
        source_gap_end = candidate.char_start
        rendered_prefix = "".join(pieces)
        if candidate.char_start == cursor + 1 and is_implicit_space_transition(
            text,
            cursor,
            rendered_prefix + candidate.replacement,
            len(rendered_prefix),
        ):
            source_gap_end -= 1
        source_gap = text[cursor:source_gap_end]
        pieces.append(source_gap)
        rendered_length += len(source_gap)
        normalized_start = rendered_length
        pieces.append(candidate.replacement)
        rendered_length += len(candidate.replacement)
        spans.append(
            RenderedSpan(
                candidate=candidate,
                normalized_start=normalized_start,
                normalized_end=rendered_length,
            )
        )
        cursor = candidate.char_end
    pieces.append(text[cursor:])
    return RenderedReplacements(text="".join(pieces), spans=tuple(spans))


def apply_candidate_replacements(
    text: str,
    candidates: Sequence[Candidate],
) -> str:
    """Apply non-overlapping candidate replacements in source order."""
    return render_candidate_replacements(text, candidates).text


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

    candidate_char_spans = tuple(
        (candidate.char_start, candidate.char_end) for candidate in candidates
    )
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
