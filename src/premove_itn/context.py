"""Caller-supplied facts for deterministic contextual normalization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class DateOrder(StrEnum):
    """Preferred field order for otherwise ambiguous numeric dates."""

    DMY = "DMY"
    MDY = "MDY"
    YMD = "YMD"


@dataclass(frozen=True, slots=True)
class NormalizationContext:
    """Explicit temporal facts available during one normalization."""

    reference_datetime: datetime | None = None
    timezone: str | None = None
    locale: str | None = None
    date_order: DateOrder | None = None

    def __post_init__(self) -> None:
        if self.reference_datetime is not None and not isinstance(
            self.reference_datetime, datetime
        ):
            raise TypeError("reference_datetime must be a datetime or None")
        if self.timezone is not None and not isinstance(self.timezone, str):
            raise TypeError("timezone must be a string or None")
        if self.timezone == "":
            raise ValueError("timezone must not be empty")
        if self.locale is not None and not isinstance(self.locale, str):
            raise TypeError("locale must be a string or None")
        if self.locale is not None and not self.locale.strip():
            raise ValueError("locale must not be empty")
        if self.date_order is not None and not isinstance(self.date_order, DateOrder):
            raise TypeError("date_order must be a DateOrder or None")
