"""Log-space path partitioning for deterministic candidate graphs."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import torch

from premove_itn.candidates import AlignmentState, Candidate, GoldGraph


def all_paths_log_partition(
    candidate_scores: torch.Tensor,
    candidate_char_spans: Sequence[tuple[int, int]] | torch.Tensor,
    source_char_count: int,
) -> torch.Tensor:
    """Return the log partition over all complete source-character paths.

    A path covers every source character with either a one-character KEEP
    transition or one selected candidate interval. KEEP transitions have score
    zero. Candidate spans are source character offsets, not encoder-token
    offsets.
    """
    spans = _validate_source_inputs(
        candidate_scores, candidate_char_spans, source_char_count
    )
    candidates_by_start: list[list[tuple[int, int]]] = [
        [] for _ in range(source_char_count)
    ]
    for index, (start, end) in enumerate(spans):
        candidates_by_start[start].append((index, end))

    negative_infinity = candidate_scores.new_full((), float("-inf"))
    forward = [negative_infinity for _ in range(source_char_count + 1)]
    forward[0] = candidate_scores.new_zeros(())
    for start in range(source_char_count):
        current = forward[start]
        forward[start + 1] = torch.logaddexp(forward[start + 1], current)
        for candidate_index, end in candidates_by_start[start]:
            transition = current + candidate_scores[candidate_index]
            forward[end] = torch.logaddexp(forward[end], transition)
    return forward[source_char_count]


def max_path_indices(
    candidate_scores: torch.Tensor,
    candidate_char_spans: Sequence[tuple[int, int]] | torch.Tensor,
    source_char_count: int,
) -> tuple[int, ...]:
    """Return candidate indices on the highest-scoring complete source path.

    This uses the same source-character interval graph as
    :func:`all_paths_log_partition`, replacing log-sum-exp with max and keeping
    predecessor pointers for backtracking. KEEP transitions have score zero.
    """
    spans = _validate_source_inputs(
        candidate_scores, candidate_char_spans, source_char_count
    )
    scores = tuple(float(score) for score in candidate_scores.detach().cpu().tolist())
    if not all(isfinite(score) for score in scores):
        raise ValueError("candidate scores must be finite")

    candidates_by_start: list[list[tuple[int, int]]] = [
        [] for _ in range(source_char_count)
    ]
    for index, (start, end) in enumerate(spans):
        candidates_by_start[start].append((index, end))

    best = [float("-inf")] * (source_char_count + 1)
    predecessors: list[int | None] = [None] * (source_char_count + 1)
    selected_candidates: list[int | None] = [None] * (source_char_count + 1)
    best[0] = 0.0
    for start in range(source_char_count):
        current = best[start]
        if not isfinite(current):
            continue

        keep_target = start + 1
        if current > best[keep_target]:
            best[keep_target] = current
            predecessors[keep_target] = start
            selected_candidates[keep_target] = None

        for candidate_index, end in candidates_by_start[start]:
            candidate_total = current + scores[candidate_index]
            if candidate_total > best[end]:
                best[end] = candidate_total
                predecessors[end] = start
                selected_candidates[end] = candidate_index

    if not isfinite(best[source_char_count]):
        raise ValueError("source graph does not contain a complete path")

    path: list[int] = []
    position = source_char_count
    while position:
        predecessor = predecessors[position]
        if predecessor is None:
            raise RuntimeError("max-path backpointer is incomplete")
        candidate_index = selected_candidates[position]
        if candidate_index is not None:
            path.append(candidate_index)
        position = predecessor
    return tuple(reversed(path))


def gold_paths_log_partition(
    candidate_scores: torch.Tensor,
    candidates: Sequence[Candidate],
    gold_graph: GoldGraph,
) -> torch.Tensor:
    """Return the log partition over the exact paths in ``gold_graph``.

    Gold states retain both source and target positions. Candidate and KEEP
    transitions are therefore followed exactly as recovered by the oracle;
    independent source-only masks are not sufficient because they can create
    target-invalid crossover paths.
    """
    if candidate_scores.ndim != 1:
        raise ValueError("candidate scores must be a one-dimensional tensor")
    if candidate_scores.shape[0] != len(candidates):
        raise ValueError("candidate scores and candidates must have equal length")
    if not gold_graph.states:
        raise ValueError("gold graph must contain at least one state")

    candidate_indices = {candidate: index for index, candidate in enumerate(candidates)}
    if len(candidate_indices) != len(candidates):
        raise ValueError("candidates must be unique")
    states = tuple(sorted(gold_graph.states))
    state_set = set(states)
    transitions_by_source: dict[
        AlignmentState, list[tuple[AlignmentState, int | None]]
    ] = {state: [] for state in states}
    for source, target in gold_graph.keep_transitions:
        _validate_gold_edge(source, target, state_set)
        transitions_by_source[source].append((target, None))
    for transition in gold_graph.candidate_transitions:
        _validate_gold_edge(transition.source, transition.target, state_set)
        try:
            candidate_index = candidate_indices[transition.candidate]
        except KeyError as error:
            raise ValueError(
                "gold graph contains a candidate outside the batch"
            ) from error
        transitions_by_source[transition.source].append(
            (transition.target, candidate_index)
        )

    negative_infinity = candidate_scores.new_full((), float("-inf"))
    forward = {state: negative_infinity for state in states}
    forward[states[0]] = candidate_scores.new_zeros(())
    for source in states:
        current = forward[source]
        for target, candidate_index in transitions_by_source[source]:
            transition_score = (
                current
                if candidate_index is None
                else current + candidate_scores[candidate_index]
            )
            forward[target] = torch.logaddexp(forward[target], transition_score)
    return forward[states[-1]]


def structured_negative_log_likelihood(
    candidate_scores: torch.Tensor,
    candidates: Sequence[Candidate],
    gold_graph: GoldGraph,
    source_char_count: int,
) -> torch.Tensor:
    """Return ``log Z(all paths) - log Z(exact gold paths)``."""
    candidate_char_spans = tuple(
        (candidate.char_start, candidate.char_end) for candidate in candidates
    )
    all_log_partition = all_paths_log_partition(
        candidate_scores,
        candidate_char_spans,
        source_char_count,
    )
    gold_log_partition = gold_paths_log_partition(
        candidate_scores,
        candidates,
        gold_graph,
    )
    return all_log_partition - gold_log_partition


def _validate_source_inputs(
    candidate_scores: torch.Tensor,
    candidate_char_spans: Sequence[tuple[int, int]] | torch.Tensor,
    source_char_count: int,
) -> tuple[tuple[int, int], ...]:
    if candidate_scores.ndim != 1:
        raise ValueError("candidate scores must be a one-dimensional tensor")
    if isinstance(candidate_char_spans, torch.Tensor):
        if candidate_char_spans.ndim != 2 or candidate_char_spans.shape[1] != 2:
            raise ValueError("candidate character spans must have shape [candidate, 2]")
        spans = tuple(
            (int(start), int(end))
            for start, end in candidate_char_spans.detach().cpu().tolist()
        )
    else:
        try:
            spans = tuple((int(start), int(end)) for start, end in candidate_char_spans)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "candidate character spans must contain start/end pairs"
            ) from error
    if len(spans) != candidate_scores.shape[0]:
        raise ValueError("candidate scores and spans must have equal length")
    if source_char_count < 0:
        raise ValueError("source character count must be non-negative")
    for start, end in spans:
        if start < 0 or end > source_char_count:
            raise ValueError("candidate character span is outside the source")
        if end <= start:
            raise ValueError("candidate character spans must be non-empty")
    return spans


def _validate_gold_edge(
    source: AlignmentState,
    target: AlignmentState,
    state_set: set[AlignmentState],
) -> None:
    if source not in state_set or target not in state_set:
        raise ValueError("gold graph transition references an unknown state")
    if target.source_position <= source.source_position:
        raise ValueError("gold graph transitions must advance the source")
