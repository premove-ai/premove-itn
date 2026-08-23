from enum import StrEnum

from premove_itn.labels import SpanKind


class GoogleClassAction(StrEnum):
    CONTEXT = "context"
    SPAN = "span"
    QUARANTINE = "quarantine"


GOOGLE_SPAN_KIND_MAP: dict[str, SpanKind] = {
    "CARDINAL": SpanKind.CARDINAL,
    "DATE": SpanKind.DATE,
    "DECIMAL": SpanKind.DECIMAL,
    "DIGIT": SpanKind.DIGIT_SEQUENCE,
    "ELECTRONIC": SpanKind.ELECTRONIC,
    "MEASURE": SpanKind.MEASUREMENT,
    "MONEY": SpanKind.MONEY,
    "ORDINAL": SpanKind.ORDINAL,
    "TELEPHONE": SpanKind.PHONE,
    "TIME": SpanKind.TIME,
}

GOOGLE_CONTEXT_CLASSES = frozenset({"PLAIN", "PUNCT"})

GOOGLE_QUARANTINE_CLASSES = frozenset({"ADDRESS", "FRACTION", "LETTERS", "VERBATIM"})


def google_class_action(source_class: str) -> GoogleClassAction:
    normalized_class = source_class.strip().upper()
    if normalized_class in GOOGLE_SPAN_KIND_MAP:
        return GoogleClassAction.SPAN
    if normalized_class in GOOGLE_CONTEXT_CLASSES:
        return GoogleClassAction.CONTEXT
    if normalized_class in GOOGLE_QUARANTINE_CLASSES:
        return GoogleClassAction.QUARANTINE
    raise ValueError(f"unknown Google TN class: {source_class!r}")


def google_span_kind(source_class: str) -> SpanKind | None:
    if google_class_action(source_class) is not GoogleClassAction.SPAN:
        return None
    return GOOGLE_SPAN_KIND_MAP[source_class.strip().upper()]
