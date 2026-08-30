"""Deterministic sentence candidate enumeration."""

from __future__ import annotations

import re
from dataclasses import dataclass

from premove_itn.labels import SPAN_KINDS, SpanKind

from . import _rust

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")
NO_SPACE_BEFORE = frozenset(".,;:!?%)]}")


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
    tokens = tuple(TOKEN_PATTERN.finditer(text))
    grouped: dict[tuple[int, int, str], set[SpanKind]] = {}

    for token_start in range(len(tokens)):
        char_start = tokens[token_start].start()
        for token_end in range(token_start + 1, len(tokens) + 1):
            char_end = tokens[token_end - 1].end()
            span_text = text[char_start:char_end]
            for kind in SPAN_KINDS:
                for replacement in _rust.realize_options(kind.value, span_text):
                    if replacement == span_text:
                        continue
                    key = (token_start, token_end, replacement)
                    grouped.setdefault(key, set()).add(kind)

    return tuple(
        Candidate(
            token_start=token_start,
            token_end=token_end,
            char_start=tokens[token_start].start(),
            char_end=tokens[token_end - 1].end(),
            text=text[tokens[token_start].start() : tokens[token_end - 1].end()],
            replacement=replacement,
            kinds=tuple(kind for kind in SPAN_KINDS if kind in kinds),
        )
        for (token_start, token_end, replacement), kinds in sorted(grouped.items())
    )


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
        if (
            text[state.source_position].isspace()
            and state.source_position + 1 < len(text)
            and not text[state.source_position + 1].isspace()
            and state.target_position < len(expected_text)
            and expected_text[state.target_position] in NO_SPACE_BEFORE
            and (
                state.target_position == 0
                or not expected_text[state.target_position - 1].isspace()
            )
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
