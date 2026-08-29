"""Deterministic sentence candidate enumeration."""

from __future__ import annotations

import re
from dataclasses import dataclass

from premove_itn.labels import SPAN_KINDS, SpanKind

from . import _rust

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


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


def target_is_reachable(text: str, expected_text: str) -> bool:
    """Return whether candidates and unchanged characters can form the target."""
    candidates_by_start: dict[int, list[tuple[int, str]]] = {}
    for candidate in build_candidate_graph(text):
        candidates_by_start.setdefault(candidate.char_start, []).append(
            (candidate.char_end, candidate.replacement)
        )

    pending = [(0, 0)]
    visited: set[tuple[int, int]] = set()
    while pending:
        source_position, target_position = pending.pop()
        state = (source_position, target_position)
        if state in visited:
            continue
        visited.add(state)

        if source_position == len(text):
            if target_position == len(expected_text):
                return True
            continue

        if (
            target_position < len(expected_text)
            and text[source_position] == expected_text[target_position]
        ):
            pending.append((source_position + 1, target_position + 1))

        pending.extend(
            (char_end, target_position + len(replacement))
            for char_end, replacement in candidates_by_start.get(source_position, ())
            if expected_text.startswith(replacement, target_position)
        )

    return False
