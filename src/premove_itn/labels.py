from enum import StrEnum


class SpanKind(StrEnum):
    DIGIT_SEQUENCE = "DIGIT_SEQUENCE"
    CARDINAL = "CARDINAL"
    TIME = "TIME"
    DATE = "DATE"
    MONEY = "MONEY"
    DECIMAL = "DECIMAL"
    PHONE = "PHONE"
    ELECTRONIC = "ELECTRONIC"
    MEASUREMENT = "MEASUREMENT"
    ORDINAL = "ORDINAL"
    PUNCTUATION = "PUNCTUATION"
    WHITELIST = "WHITELIST"
    WORD = "WORD"


SPAN_KINDS: tuple[SpanKind, ...] = tuple(SpanKind)

BIO_LABELS: tuple[str, ...] = (
    "O",
    *(f"{prefix}-{kind}" for kind in SPAN_KINDS for prefix in ("B", "I")),
)

LABEL_TO_ID: dict[str, int] = {label: index for index, label in enumerate(BIO_LABELS)}
ID_TO_LABEL: dict[int, str] = {index: label for label, index in LABEL_TO_ID.items()}
