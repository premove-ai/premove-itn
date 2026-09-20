"""Deterministic contextual temporal annotations."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .context import NormalizationContext
from .labels import SpanKind
from .results import NormalizationResult, NormalizedSpan

_RELATIVE_DATE_PATTERN = re.compile(
    r"(?<!\w)(day after tomorrow|day before yesterday|today|tomorrow|yesterday)(?!\w)",
    re.IGNORECASE,
)
_RELATIVE_DATE_OFFSETS = {
    "today": 0,
    "tomorrow": 1,
    "yesterday": -1,
    "day after tomorrow": 2,
    "day before yesterday": -2,
}


def _reference_date(context: NormalizationContext | None) -> date | None:
    if context is None or context.reference_datetime is None:
        return None
    reference = context.reference_datetime
    if context.timezone is not None:
        try:
            timezone = ZoneInfo(context.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone {context.timezone!r}") from exc
        if reference.tzinfo is not None:
            return reference.astimezone(timezone).date()
        return reference.date()
    if reference.tzinfo is not None:
        return reference.astimezone(reference.tzinfo).date()
    return reference.date()


def _map_source_boundary(
    position: int,
    source_length: int,
    result: NormalizationResult,
) -> int | None:
    """Map an unchanged source boundary into the rendered text."""
    previous_source = 0
    previous_normalized = 0
    for span in result.spans:
        if position < span.source_start:
            return _map_segment(
                position,
                previous_source,
                previous_normalized,
                span.source_start,
                span.normalized_start,
            )
        if position == span.source_start:
            return span.normalized_start
        if position < span.source_end:
            return None
        previous_source = span.source_end
        previous_normalized = span.normalized_end
        if position == span.source_end:
            return previous_normalized
    return _map_segment(
        position,
        previous_source,
        previous_normalized,
        source_length,
        len(result.text),
    )


def _map_segment(
    position: int,
    source_start: int,
    normalized_start: int,
    source_end: int,
    normalized_end: int,
) -> int | None:
    source_length = source_end - source_start
    normalized_length = normalized_end - normalized_start
    if source_length == normalized_length:
        return normalized_start + (position - source_start)
    if source_length == 1 and normalized_length == 0:
        return normalized_start
    return None


def _source_range_to_normalized(
    source_start: int,
    source_end: int,
    source_length: int,
    result: NormalizationResult,
) -> tuple[int, int] | None:
    if any(
        span.source_start < source_end and source_start < span.source_end
        for span in result.spans
    ):
        return None
    normalized_start = _map_source_boundary(source_start, source_length, result)
    normalized_end = _map_source_boundary(source_end, source_length, result)
    if normalized_start is None or normalized_end is None:
        return None
    if normalized_end < normalized_start:
        return None
    return normalized_start, normalized_end


def _render_resolved_text(result: NormalizationResult) -> str:
    resolved_spans = tuple(
        span for span in result.spans if span.resolved_value is not None
    )
    if not resolved_spans:
        return result.text
    pieces: list[str] = []
    cursor = 0
    for span in resolved_spans:
        if span.normalized_start < cursor:
            raise ValueError("resolved spans overlap")
        pieces.append(result.text[cursor : span.normalized_start])
        pieces.append(span.resolved_value)
        cursor = span.normalized_end
    pieces.append(result.text[cursor:])
    return "".join(pieces)


def annotate_temporal(
    source: str,
    result: NormalizationResult,
    context: NormalizationContext | None,
) -> NormalizationResult:
    """Annotate supported relative dates and resolve them when possible."""
    matches = tuple(_RELATIVE_DATE_PATTERN.finditer(source))
    if not matches:
        return result
    reference_date = _reference_date(context)
    spans = list(result.spans)
    for match in matches:
        source_start, source_end = match.span()
        normalized_range = _source_range_to_normalized(
            source_start,
            source_end,
            len(source),
            result,
        )
        if normalized_range is None:
            continue
        normalized_start, normalized_end = normalized_range
        resolved_value = None
        if reference_date is not None:
            offset = _RELATIVE_DATE_OFFSETS[match.group(0).lower()]
            resolved_value = (reference_date + timedelta(days=offset)).isoformat()
        annotation = NormalizedSpan(
            source_start=source_start,
            source_end=source_end,
            normalized_start=normalized_start,
            normalized_end=normalized_end,
            source_text=source[source_start:source_end],
            normalized_text=result.text[normalized_start:normalized_end],
            kinds=(SpanKind.DATE,),
            resolved_value=resolved_value,
        )
        overlapping = [
            index
            for index, span in enumerate(spans)
            if span.source_start < source_end and source_start < span.source_end
        ]
        if not overlapping:
            spans.append(annotation)
        elif len(overlapping) == 1:
            index = overlapping[0]
            existing = spans[index]
            if (
                existing.source_start == source_start
                and existing.source_end == source_end
                and SpanKind.DATE in existing.kinds
                and resolved_value is not None
            ):
                spans[index] = replace(existing, resolved_value=resolved_value)

    ordered_spans = tuple(sorted(spans, key=lambda span: span.source_start))
    enriched = NormalizationResult(
        text=result.text,
        resolved_text=result.resolved_text,
        spans=ordered_spans,
    )
    return replace(enriched, resolved_text=_render_resolved_text(enriched))
