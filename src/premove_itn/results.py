"""Public immutable results for structured normalization."""

from __future__ import annotations

from dataclasses import dataclass

from .labels import SpanKind


@dataclass(frozen=True, slots=True)
class NormalizedSpan:
    """One selected normalization with source and normalized coordinates."""

    source_start: int
    source_end: int
    normalized_start: int
    normalized_end: int
    source_text: str
    normalized_text: str
    kinds: tuple[SpanKind, ...]
    resolved_value: str | None = None


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Readable, resolved, and structured views of one normalization."""

    text: str
    resolved_text: str
    spans: tuple[NormalizedSpan, ...]
