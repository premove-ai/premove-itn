"""Log-space path partitioning for candidate interval graphs."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def interval_log_partition(
    candidate_scores: torch.Tensor,
    candidate_token_spans: torch.Tensor,
    sentence_token_count: int,
    *,
    candidate_mask: torch.Tensor | Sequence[bool] | None = None,
    keep_mask: torch.Tensor | Sequence[bool] | None = None,
) -> torch.Tensor:
    """Return the log partition over complete non-overlapping token paths.

    A path covers every source token with either an unchanged one-token KEEP
    transition or one selected candidate interval. Candidate scores are added
    along a path, while KEEP transitions have score zero. ``candidate_mask``
    restricts which candidate transitions are legal. ``keep_mask`` restricts
    which source tokens may use KEEP; this is needed for a gold path partition
    when a target edit must cover a source position.
    """
    _validate_inputs(
        candidate_scores,
        candidate_token_spans,
        sentence_token_count,
        candidate_mask=candidate_mask,
        keep_mask=keep_mask,
    )
    candidate_count = candidate_scores.shape[0]
    enabled_candidates = _bool_mask(candidate_mask, candidate_count, "candidate")
    enabled_keep = _bool_mask(keep_mask, sentence_token_count, "KEEP")

    spans = candidate_token_spans.detach().cpu().tolist()
    candidates_by_start: list[list[tuple[int, int]]] = [
        [] for _ in range(sentence_token_count)
    ]
    for index, (start, end) in enumerate(spans):
        if enabled_candidates[index]:
            candidates_by_start[start].append((index, end))

    negative_infinity = candidate_scores.new_full((), float("-inf"))
    forward = [negative_infinity for _ in range(sentence_token_count + 1)]
    forward[0] = candidate_scores.new_zeros(())

    for start in range(sentence_token_count):
        current = forward[start]
        if enabled_keep[start]:
            forward[start + 1] = torch.logaddexp(forward[start + 1], current)
        for candidate_index, end in candidates_by_start[start]:
            transition = current + candidate_scores[candidate_index]
            forward[end] = torch.logaddexp(forward[end], transition)

    return forward[sentence_token_count]


def structured_negative_log_likelihood(
    candidate_scores: torch.Tensor,
    candidate_token_spans: torch.Tensor,
    sentence_token_count: int,
    gold_candidate_mask: torch.Tensor | Sequence[bool],
    *,
    gold_keep_mask: torch.Tensor | Sequence[bool] | None = None,
) -> torch.Tensor:
    """Return ``log Z(all paths) - log Z(gold-compatible paths)``.

    ``gold_candidate_mask`` selects candidate transitions that occur on at
    least one correct derivation. ``gold_keep_mask`` selects unchanged source
    tokens that are allowed on a correct derivation. It defaults to all KEEP
    transitions for convenience, but callers with edits must constrain it from
    the source-target gold alignment.
    """
    all_log_partition = interval_log_partition(
        candidate_scores,
        candidate_token_spans,
        sentence_token_count,
    )
    gold_log_partition = interval_log_partition(
        candidate_scores,
        candidate_token_spans,
        sentence_token_count,
        candidate_mask=gold_candidate_mask,
        keep_mask=gold_keep_mask,
    )
    if not torch.isfinite(gold_log_partition):
        raise ValueError("gold-compatible paths do not cover the sentence")
    return all_log_partition - gold_log_partition


def _bool_mask(
    mask: torch.Tensor | Sequence[bool] | None,
    expected_size: int,
    name: str,
) -> list[bool]:
    if mask is None:
        return [True] * expected_size
    values = torch.as_tensor(mask, dtype=torch.bool).detach().cpu()
    if values.ndim != 1 or values.shape[0] != expected_size:
        raise ValueError(f"{name} mask must have length {expected_size}")
    return values.tolist()


def _validate_inputs(
    candidate_scores: torch.Tensor,
    candidate_token_spans: torch.Tensor,
    sentence_token_count: int,
    *,
    candidate_mask: torch.Tensor | Sequence[bool] | None,
    keep_mask: torch.Tensor | Sequence[bool] | None,
) -> None:
    if candidate_scores.ndim != 1:
        raise ValueError("candidate scores must be a one-dimensional tensor")
    if candidate_token_spans.ndim != 2 or candidate_token_spans.shape[1] != 2:
        raise ValueError("candidate token spans must have shape [candidate, 2]")
    if candidate_token_spans.shape[0] != candidate_scores.shape[0]:
        raise ValueError("candidate scores and spans must have equal length")
    if sentence_token_count < 0:
        raise ValueError("sentence token count must be non-negative")
    if candidate_token_spans.numel():
        starts = candidate_token_spans[:, 0]
        ends = candidate_token_spans[:, 1]
        if torch.any(starts < 0) or torch.any(ends > sentence_token_count):
            raise ValueError("candidate token span is outside the sentence")
        if torch.any(ends <= starts):
            raise ValueError("candidate token spans must be non-empty")
    _bool_mask(candidate_mask, candidate_scores.shape[0], "candidate")
    _bool_mask(keep_mask, sentence_token_count, "KEEP")
