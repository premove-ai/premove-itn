"""Deterministic contextual temporal annotations."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .context import DateOrder, NormalizationContext
from .labels import SpanKind
from .results import NormalizationResult, NormalizedSpan

_RELATIVE_DATE_PATTERN = re.compile(
    r"(?<!\w)(day after tomorrow|day before yesterday|today|tomorrow|yesterday)(?!\w)",
    re.IGNORECASE,
)
_RELATIVE_COUNT_PATTERN = (
    r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)"
)
_RELATIVE_OFFSET_PATTERN = re.compile(
    rf"(?<!\w)(?:"
    rf"in\s+(?P<in_count>{_RELATIVE_COUNT_PATTERN})\s+(?P<in_unit>day|days|week|weeks)"
    rf"|(?P<from_count>{_RELATIVE_COUNT_PATTERN})\s+"
    rf"(?P<from_unit>day|days|week|weeks)\s+from\s+(?:now|today)"
    rf"|(?P<ago_count>{_RELATIVE_COUNT_PATTERN})\s+"
    rf"(?P<ago_unit>day|days|week|weeks)\s+ago"
    rf")(?!\w)",
    re.IGNORECASE,
)
_RELATIVE_DATE_OFFSETS = {
    "today": 0,
    "tomorrow": 1,
    "yesterday": -1,
    "day after tomorrow": 2,
    "day before yesterday": -2,
}
_RELATIVE_COUNT_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
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
_DATE_TOKEN_PATTERN = re.compile(r"[a-z]+|\d+", re.IGNORECASE)
_NUMERIC_DATE_PATTERN = re.compile(
    r"\s*(\d+)\s*(?:([/.-])\s*(\d+)(?:\s*\2\s*(\d+))?|\s+(\d+)(?:\s+(\d+))?)\s*"
)


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


def _relative_offset(match: re.Match[str]) -> int:
    """Return the signed day offset represented by one relative-date match."""
    if match.lastgroup is None:
        return _RELATIVE_DATE_OFFSETS[match.group(0).lower()]

    count_text = next(
        value
        for value in (
            match.group("in_count"),
            match.group("from_count"),
            match.group("ago_count"),
        )
        if value is not None
    ).lower()
    count = (
        int(count_text) if count_text.isdigit() else _RELATIVE_COUNT_WORDS[count_text]
    )
    unit = next(
        value
        for value in (
            match.group("in_unit"),
            match.group("from_unit"),
            match.group("ago_unit"),
        )
        if value is not None
    ).lower()
    days = count * (7 if unit.startswith("week") else 1)
    if match.group("ago_count") is not None:
        return -days
    return days


def _relative_matches(source: str) -> tuple[re.Match[str], ...]:
    """Return relative-date matches without exposing nested ``today`` spans."""
    offset_matches = tuple(_RELATIVE_OFFSET_PATTERN.finditer(source))
    date_matches = tuple(
        match
        for match in _RELATIVE_DATE_PATTERN.finditer(source)
        if not any(
            offset.start() <= match.start() and match.end() <= offset.end()
            for offset in offset_matches
        )
    )
    return tuple(
        sorted((*offset_matches, *date_matches), key=lambda match: match.start())
    )


def _parse_named_date(text: str) -> tuple[int, int, int | None] | None:
    tokens = [token.lower() for token in _DATE_TOKEN_PATTERN.findall(text)]
    tokens = [
        token for token in tokens if token not in {"of", "the", "st", "nd", "rd", "th"}
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


def _parse_numeric_date(text: str) -> tuple[tuple[int, ...], tuple[str, ...]] | None:
    match = _NUMERIC_DATE_PATTERN.fullmatch(text)
    if match is None:
        return None
    if match.group(2) is not None:
        tokens = (match.group(1), match.group(3), match.group(4))
    else:
        tokens = (match.group(1), match.group(5), match.group(6))
    raw_tokens = tuple(token for token in tokens if token is not None)
    return tuple(int(token) for token in raw_tokens), raw_tokens


def _valid_numeric_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _resolve_numeric_date(
    values: tuple[int, ...],
    raw_values: tuple[str, ...],
    context: NormalizationContext | None,
    reference_date: date | None,
) -> str | None:
    if len(values) == 2:
        if any(len(value) == 4 for value in raw_values) or reference_date is None:
            return None
        day_month_candidates: dict[str, str] = {}
        for order, day_index, month_index in (
            ("DM", 0, 1),
            ("MD", 1, 0),
        ):
            resolved = _valid_numeric_date(
                reference_date.year,
                values[month_index],
                values[day_index],
            )
            if resolved is not None:
                day_month_candidates[resolved] = order
        if len(day_month_candidates) == 1:
            return next(iter(day_month_candidates))
        if (
            len(day_month_candidates) != 2
            or context is None
            or context.date_order is None
        ):
            return None
        preferred_order = (
            "DM"
            if context.date_order.index("D") < context.date_order.index("M")
            else "MD"
        )
        return next(
            (
                resolved
                for resolved, order in day_month_candidates.items()
                if order == preferred_order
            ),
            None,
        )

    if len(values) != 3:
        return None
    year_indices = [index for index, value in enumerate(raw_values) if len(value) == 4]
    if len(year_indices) != 1:
        return None
    year_index = year_indices[0]
    remaining_indices = tuple(index for index in range(3) if index != year_index)
    candidates: dict[str, set[DateOrder]] = {}
    for day_index, month_index in (
        (remaining_indices[0], remaining_indices[1]),
        (remaining_indices[1], remaining_indices[0]),
    ):
        order_values = [""] * 3
        order_values[year_index] = "Y"
        order_values[day_index] = "D"
        order_values[month_index] = "M"
        order = DateOrder("".join(order_values))
        resolved = _valid_numeric_date(
            values[year_index],
            values[month_index],
            values[day_index],
        )
        if resolved is not None:
            candidates.setdefault(resolved, set()).add(order)
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) != 2 or context is None or context.date_order is None:
        return None
    return next(
        (
            resolved
            for resolved, orders in candidates.items()
            if context.date_order in orders
        ),
        None,
    )


def annotate_numeric_dates(
    result: NormalizationResult,
    context: NormalizationContext | None,
) -> NormalizationResult:
    """Resolve selected numeric DATE spans when their interpretation is safe."""
    spans = list(result.spans)
    reference_date = None
    reference_date_checked = False
    changed = False
    for index, span in enumerate(spans):
        if SpanKind.DATE not in span.kinds or span.resolved_value is not None:
            continue
        parsed = _parse_numeric_date(span.normalized_text)
        if parsed is None:
            continue
        values, raw_values = parsed
        if len(values) == 2 and not reference_date_checked:
            reference_date = _reference_date(context)
            reference_date_checked = True
        resolved_value = _resolve_numeric_date(
            values,
            raw_values,
            context,
            reference_date,
        )
        if resolved_value is None:
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
    """Annotate supported relative dates and resolve them when possible.

    Weekday-relative expressions are intentionally left for a later stage
    because their calendar policy is not part of this resolver contract.
    """
    matches = _relative_matches(source)
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
                    offset = _relative_offset(match)
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
            offset = _relative_offset(match)
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
