from __future__ import annotations

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
