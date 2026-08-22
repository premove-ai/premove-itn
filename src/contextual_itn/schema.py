from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class SemanticClass(StrEnum):
    CARDINAL = "CARDINAL"
    ORDINAL = "ORDINAL"
    DIGIT_SEQUENCE = "DIGIT_SEQUENCE"
    DECIMAL = "DECIMAL"
    FRACTION = "FRACTION"
    MONEY = "MONEY"
    PERCENTAGE = "PERCENTAGE"
    DATE = "DATE"
    TIME = "TIME"
    DURATION = "DURATION"
    MEASUREMENT = "MEASUREMENT"
    PHONE = "PHONE"
    ALPHANUMERIC = "ALPHANUMERIC"
    ELECTRONIC = "ELECTRONIC"


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    kind: SemanticClass
    value: str


@dataclass(frozen=True, slots=True)
class Example:
    text: str
    tokens: tuple[str, ...]
    labels: tuple[str, ...]
    spans: tuple[Span, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
