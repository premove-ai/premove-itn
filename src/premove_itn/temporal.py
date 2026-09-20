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
_MONTHS = {
    name: index
    for index, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_WEEKDAYS = frozenset(
    (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )
)
_DATE_TOKEN_PATTERN = re.compile(r"[a-z]+|\d+", re.IGNORECASE)


def _reference_date(context: NormalizationContext | None) -> date | None:
    if context is None or context.reference_datetime is None:
        return None
    reference = context.reference_datetime
    if context.timezone is not None:
        try:
            timezone = ZoneInfo(context.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone {context.timezone!r}") from exc
        if reference.utcoffset() is not None:
            return reference.astimezone(timezone).date()
        return reference.date()
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


def _parse_named_date(text: str) -> tuple[int, int, int | None] | None:
    tokens = [token.lower() for token in _DATE_TOKEN_PATTERN.findall(text)]
    tokens = [
        token
        for token in tokens
        if token not in _WEEKDAYS and token not in {"of", "the", "st", "nd", "rd", "th"}
    ]
    month_positions = [index for index, token in enumerate(tokens) if token in _MONTHS]
    if len(month_positions) != 1:
        return None
    month_position = month_positions[0]
    if month_position == 0:
        day_position = 1
    elif month_position == 1:
        day_position = 0
    else:
        return None
    if len(tokens) not in (2, 3) or not tokens[day_position].isdigit():
        return None
    day = int(tokens[day_position])
    if not 1 <= day <= 31:
        return None
    year = None
    if len(tokens) == 3:
        year_token = tokens[2]
        if not year_token.isdigit() or len(year_token) != 4:
            return None
        year = int(year_token)
    return day, _MONTHS[tokens[month_position]], year


def annotate_missing_year_dates(
    result: NormalizationResult,
    context: NormalizationContext | None,
) -> NormalizationResult:
    """Resolve selected named-month DATE spans with an explicit or missing year."""
    spans = list(result.spans)
    reference_date = None
    reference_date_checked = False
    changed = False
    for index, span in enumerate(spans):
        if SpanKind.DATE not in span.kinds or span.resolved_value is not None:
            continue
        parsed = _parse_named_date(span.normalized_text)
        if parsed is None:
            continue
        day, month, explicit_year = parsed
        if explicit_year is None:
            if not reference_date_checked:
                reference_date = _reference_date(context)
                reference_date_checked = True
            if reference_date is None:
                continue
            year = reference_date.year
        else:
            year = explicit_year
        try:
            resolved_value = date(year, month, day).isoformat()
        except ValueError:
            continue
        spans[index] = replace(span, resolved_value=resolved_value)
        changed = True
    if not changed:
        return result
    enriched = replace(result, spans=tuple(spans))
    return replace(enriched, resolved_text=_render_resolved_text(enriched))


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
        overlapping = [
            index
            for index, span in enumerate(spans)
            if span.source_start < source_end and source_start < span.source_end
        ]
        if overlapping:
            if len(overlapping) == 1:
                index = overlapping[0]
                existing = spans[index]
                if (
                    existing.source_start == source_start
                    and existing.source_end == source_end
                    and SpanKind.DATE in existing.kinds
                    and reference_date is not None
                ):
                    offset = _RELATIVE_DATE_OFFSETS[match.group(0).lower()]
                    spans[index] = replace(
                        existing,
                        resolved_value=(
                            reference_date + timedelta(days=offset)
                        ).isoformat(),
                    )
            continue
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
        spans.append(annotation)

    ordered_spans = tuple(sorted(spans, key=lambda span: span.source_start))
    enriched = NormalizationResult(
        text=result.text,
        resolved_text=result.resolved_text,
        spans=ordered_spans,
    )
    return replace(enriched, resolved_text=_render_resolved_text(enriched))
