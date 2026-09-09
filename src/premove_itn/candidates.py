"""Deterministic sentence candidate enumeration."""

from __future__ import annotations

import re
from dataclasses import dataclass

from premove_itn.labels import SPAN_KINDS, SpanKind

from . import _rust

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")
NO_SPACE_BEFORE = frozenset(".,;:!?%)]}")


def is_implicit_space_transition(
    source_text: str,
    source_position: int,
    target_text: str,
    target_position: int,
) -> bool:
    """Return whether one source separator may attach a target symbol."""
    return (
        0 <= source_position < len(source_text)
        and source_text[source_position].isspace()
        and source_position + 1 < len(source_text)
        and not source_text[source_position + 1].isspace()
        and 0 <= target_position < len(target_text)
        and target_text[target_position] in NO_SPACE_BEFORE
        and (target_position == 0 or not target_text[target_position - 1].isspace())
    )


@dataclass(frozen=True, slots=True)
class Candidate:
    """One distinct edit for half-open source-token and character spans."""

    token_start: int
    token_end: int
    char_start: int
    char_end: int
    text: str
    replacement: str
    kinds: tuple[SpanKind, ...]


@dataclass(frozen=True, slots=True)
class _SourceSpan:
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True, order=True, slots=True)
class AlignmentState:
    """One aligned source and expected-output character position."""

    source_position: int
    target_position: int


@dataclass(frozen=True, slots=True)
class CandidateTransition:
    """One candidate edge that participates in a complete gold derivation."""

    source: AlignmentState
    target: AlignmentState
    candidate: Candidate


@dataclass(frozen=True, slots=True)
class GoldGraph:
    """Packed states and candidate edges for every correct derivation."""

    states: tuple[AlignmentState, ...]
    candidate_transitions: tuple[CandidateTransition, ...]
    keep_transitions: tuple[tuple[AlignmentState, AlignmentState], ...] = ()


def build_candidate_graph(text: str) -> tuple[Candidate, ...]:
    """Enumerate and deduplicate all Rust realizations for all token spans."""
    token_bounds = tuple(
        (match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)
    )
    spans = tuple(
        _SourceSpan(
            token_start=token_start,
            token_end=token_end,
            char_start=token_bounds[token_start][0],
            char_end=token_bounds[token_end - 1][1],
            text=text[token_bounds[token_start][0] : token_bounds[token_end - 1][1]],
        )
        for token_start in range(len(token_bounds))
        for token_end in range(token_start + 1, len(token_bounds) + 1)
    )
    unique_texts = tuple(dict.fromkeys(span.text for span in spans))
    text_indices = {span_text: index for index, span_text in enumerate(unique_texts)}
    realization_tables = _rust.realize_candidate_batch(
        unique_texts,
        tuple(kind.value for kind in SPAN_KINDS),
    )
    decoded_tables = tuple(
        tuple(
            (
                replacement,
                tuple(
                    kind
                    for index, kind in enumerate(SPAN_KINDS)
                    if kind_mask & (1 << index)
                ),
            )
            for replacement, kind_mask in table
        )
        for table in realization_tables
    )
    candidates = [
        Candidate(
            token_start=span.token_start,
            token_end=span.token_end,
            char_start=span.char_start,
            char_end=span.char_end,
            text=span.text,
            replacement=replacement,
            kinds=kinds,
        )
        for span in spans
        for replacement, kinds in decoded_tables[text_indices[span.text]]
    ]
    candidates.sort(
        key=lambda candidate: (
            candidate.token_start,
            candidate.token_end,
            candidate.replacement,
        )
    )
    return tuple(candidates)


def build_gold_graph(
    text: str,
    expected_text: str,
    candidates: tuple[Candidate, ...] | None = None,
) -> GoldGraph | None:
    """Recover all candidate transitions on complete derivations of the target."""
    if text == expected_text:
        states = tuple(
            AlignmentState(position, position) for position in range(len(text) + 1)
        )
        return GoldGraph(
            states,
            (),
            tuple(zip(states[:-1], states[1:], strict=True)),
        )

    candidates_by_start: dict[int, list[Candidate]] = {}
    for candidate in (
        candidates if candidates is not None else build_candidate_graph(text)
    ):
        candidates_by_start.setdefault(candidate.char_start, []).append(candidate)

    start = AlignmentState(0, 0)
    end = AlignmentState(len(text), len(expected_text))
    pending = [start]
    forward_states: set[AlignmentState] = set()
    predecessors: dict[AlignmentState, list[AlignmentState]] = {}
    candidate_edges: list[CandidateTransition] = []
    keep_edges: list[tuple[AlignmentState, AlignmentState]] = []
    while pending:
        state = pending.pop()
        if state in forward_states:
            continue
        forward_states.add(state)

        if state.source_position == len(text):
            continue

        if (
            state.target_position < len(expected_text)
            and text[state.source_position] == expected_text[state.target_position]
        ):
            target = AlignmentState(
                state.source_position + 1,
                state.target_position + 1,
            )
            predecessors.setdefault(target, []).append(state)
            keep_edges.append((state, target))
            pending.append(target)

        # Spoken punctuation words are separate source tokens, but their
        # written symbols attach to the preceding token (for example,
        # ``five percent`` -> ``5%``).  Treat only that source separator as
        # an implicit normalization; all other characters still need an
        # exact unchanged-character match or a candidate edit.
        if is_implicit_space_transition(
            text,
            state.source_position,
            expected_text,
            state.target_position,
        ):
            target = AlignmentState(state.source_position + 1, state.target_position)
            predecessors.setdefault(target, []).append(state)
            keep_edges.append((state, target))
            pending.append(target)

        for candidate in candidates_by_start.get(state.source_position, ()):
            if not expected_text.startswith(
                candidate.replacement, state.target_position
            ):
                continue
            target = AlignmentState(
                candidate.char_end,
                state.target_position + len(candidate.replacement),
            )
            predecessors.setdefault(target, []).append(state)
            candidate_edges.append(CandidateTransition(state, target, candidate))
            pending.append(target)

    if end not in forward_states:
        return None

    can_reach_end = {end}
    pending = [end]
    while pending:
        state = pending.pop()
        for predecessor in predecessors.get(state, ()):
            if predecessor not in can_reach_end:
                can_reach_end.add(predecessor)
                pending.append(predecessor)

    gold_edges = tuple(
        edge
        for edge in candidate_edges
        if edge.source in can_reach_end and edge.target in can_reach_end
    )
    gold_keep_edges = tuple(
        edge
        for edge in keep_edges
        if edge[0] in can_reach_end and edge[1] in can_reach_end
    )
    return GoldGraph(tuple(sorted(can_reach_end)), gold_edges, gold_keep_edges)


def target_is_reachable(text: str, expected_text: str) -> bool:
    """Return whether candidates and unchanged characters can form the target."""
    return build_gold_graph(text, expected_text) is not None
