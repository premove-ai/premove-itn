import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum

from premove_itn.dataset.google_tn.candidates import (
    GoogleTnCandidateResult,
    GoogleTnSpanCandidate,
)
from premove_itn.labels import SpanKind

GoogleTnRealizer = Callable[[str, str], str | None]


class GoogleTnRejectionReason(StrEnum):
    REALIZER_REJECTED = "realizer_rejected"
    WRITTEN_MISMATCH = "written_mismatch"


class GoogleTnMatchKind(StrEnum):
    EXACT = "exact"
    CANONICAL = "canonical"


@dataclass(frozen=True, slots=True)
class GoogleTnTrustedCandidate:
    source_name: str
    sentence_number: int
    line_number: int
    source_class: str
    kind: SpanKind
    spoken: str
    google_written: str
    realized: str
    match_kind: GoogleTnMatchKind


@dataclass(frozen=True, slots=True)
class GoogleTnCandidateRejection:
    candidate: GoogleTnSpanCandidate
    reason: GoogleTnRejectionReason
    realized: str | None


@dataclass(frozen=True, slots=True)
class GoogleTnValidationResult:
    extraction: GoogleTnCandidateResult
    trusted_candidates: tuple[GoogleTnTrustedCandidate, ...]
    rejected_candidates: tuple[GoogleTnCandidateRejection, ...]

    @property
    def is_accepted(self) -> bool:
        return not self.extraction.quarantined_rows and not self.rejected_candidates


def validate_google_tn_candidates(
    extraction: GoogleTnCandidateResult, realizer: GoogleTnRealizer
) -> GoogleTnValidationResult:
    trusted_candidates: list[GoogleTnTrustedCandidate] = []
    rejected_candidates: list[GoogleTnCandidateRejection] = []

    for candidate in extraction.span_candidates:
        realized = realizer(candidate.kind.value, candidate.spoken)
        if realized is None:
            rejected_candidates.append(
                GoogleTnCandidateRejection(
                    candidate, GoogleTnRejectionReason.REALIZER_REJECTED, None
                )
            )
            continue

        match_kind = _google_written_match_kind(
            candidate.kind, candidate.written, realized
        )
        if match_kind is not None:
            trusted_candidates.append(
                GoogleTnTrustedCandidate(
                    candidate.source_name,
                    candidate.sentence_number,
                    candidate.line_number,
                    candidate.source_class,
                    candidate.kind,
                    candidate.spoken,
                    candidate.written,
                    realized,
                    match_kind,
                )
            )
        else:
            rejected_candidates.append(
                GoogleTnCandidateRejection(
                    candidate, GoogleTnRejectionReason.WRITTEN_MISMATCH, realized
                )
            )

    return GoogleTnValidationResult(
        extraction, tuple(trusted_candidates), tuple(rejected_candidates)
    )


def _canonical_date_written(text: str) -> str:
    return " ".join(text.casefold().replace(",", "").split())


_MONTH_NUMBERS = {
    alias: month
    for month, aliases in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for alias in aliases
}


def _calendar_date(year: str, month: int, day: str) -> date | None:
    if not (len(year) == 4 and year.isascii() and year.isdigit()):
        return None
    if not (day.isascii() and day.isdigit() and 1 <= len(day) <= 2):
        return None
    try:
        return date(int(year), month, int(day))
    except ValueError:
        return None


def _canonical_date(text: str) -> date | None:
    numeric = re.fullmatch(
        r"\s*(?P<year>\d{4})(?P<separator>[-/])(?P<month>\d{1,2})(?P=separator)(?P<day>\d{1,2})\s*",
        text,
    )
    if numeric is not None:
        try:
            return date(
                int(numeric.group("year")),
                int(numeric.group("month")),
                int(numeric.group("day")),
            )
        except ValueError:
            return None

    fields = text.replace(",", " ").split()
    if len(fields) != 3:
        return None
    first_month = _MONTH_NUMBERS.get(fields[0].casefold().removesuffix("."))
    second_month = _MONTH_NUMBERS.get(fields[1].casefold().removesuffix("."))
    if first_month is not None:
        return _calendar_date(fields[2], first_month, fields[1])
    if second_month is not None:
        return _calendar_date(fields[2], second_month, fields[0])
    return None


def _canonical_grouped_integer(text: str) -> str | None:
    text = text.strip(" \t\r\n")
    if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)", text):
        return None
    return text.replace(",", "")


def _canonical_grouped_decimal(text: str) -> str | None:
    match = re.fullmatch(
        r"(?P<integer>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+))(?P<fraction>\.\d+)", text
    )
    if match is None:
        return None
    integer = _canonical_grouped_integer(match.group("integer"))
    if integer is None:
        return None
    return integer + match.group("fraction")


def _canonical_grouped_number(text: str) -> str | None:
    if "." in text:
        return _canonical_grouped_decimal(text)
    return _canonical_grouped_integer(text)


_MONEY_MAGNITUDES = {
    "k": Decimal("1000"),
    "thousand": Decimal("1000"),
    "m": Decimal("1000000"),
    "million": Decimal("1000000"),
    "bn": Decimal("1000000000"),
    "billion": Decimal("1000000000"),
}


def _canonical_grouped_money(text: str) -> tuple[str, Decimal, str] | None:
    match = re.fullmatch(
        r"(?P<prefix>\D*?)(?P<amount>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
        r"(?:\s*(?P<magnitude>thousand|million|billion|bn|k|m))?(?P<suffix>\D*)",
        text.strip(" \t\r\n"),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    amount = match.group("amount")
    canonical_amount = _canonical_grouped_number(amount)
    if canonical_amount is None:
        return None
    value = Decimal(canonical_amount)
    magnitude = match.group("magnitude")
    if magnitude is not None:
        value *= _MONEY_MAGNITUDES[magnitude.casefold()]
    return match.group("prefix"), value, match.group("suffix")


def _canonical_measurement(text: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"\s*(?P<amount>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(?P<unit>\S+)\s*",
        text,
    )
    if match is None:
        return None
    amount = _canonical_grouped_number(match.group("amount"))
    if amount is None:
        return None
    return amount, match.group("unit")


def _canonical_time(text: str) -> tuple[int, int, str | None] | None:
    match = re.fullmatch(
        r"\s*(?P<hour>\d{1,2})(?:\s*[:.]\s*(?P<minute>\d{2}))?\s*"
        r"(?P<meridiem>[aApP](?:\s*\.\s*)?\s*[mM]\.?)?"
        r"(?:\s+(?P<timezone>[A-Za-z]{2,5}))?\s*",
        text,
    )
    if match is None:
        return None

    hour = int(match.group("hour"))
    minute_text = match.group("minute")
    meridiem_text = match.group("meridiem")
    timezone_text = match.group("timezone")
    timezone = timezone_text.casefold() if timezone_text is not None else None
    if meridiem_text is None:
        if minute_text is None or hour > 23:
            return None
        minute = int(minute_text)
        return (hour, minute, timezone) if minute < 60 else None

    meridiem = re.sub(r"[\s.]", "", meridiem_text).casefold()
    if meridiem not in {"am", "pm"} or not 1 <= hour <= 12:
        return None
    minute = int(minute_text) if minute_text is not None else 0
    if minute >= 60:
        return None
    if meridiem == "am":
        hour = 0 if hour == 12 else hour
    elif hour != 12:
        hour += 12
    return hour, minute, timezone


def _same_canonical(
    left: str, right: str, canonicalize: Callable[[str], object | None]
) -> bool:
    left_value = canonicalize(left)
    return left_value is not None and left_value == canonicalize(right)


def _google_written_match_kind(
    kind: SpanKind, google_written: str, realized: str
) -> GoogleTnMatchKind | None:
    if google_written == realized:
        return GoogleTnMatchKind.EXACT
    if kind is SpanKind.DATE and _canonical_date_written(
        google_written
    ) == _canonical_date_written(realized):
        return GoogleTnMatchKind.CANONICAL
    if kind is SpanKind.DATE and _same_canonical(
        google_written, realized, _canonical_date
    ):
        return GoogleTnMatchKind.CANONICAL
    if kind is SpanKind.CARDINAL and _same_canonical(
        google_written, realized, _canonical_grouped_integer
    ):
        return GoogleTnMatchKind.CANONICAL
    if kind is SpanKind.DECIMAL and _same_canonical(
        google_written, realized, _canonical_grouped_decimal
    ):
        return GoogleTnMatchKind.CANONICAL
    if kind is SpanKind.MONEY and _same_canonical(
        google_written, realized, _canonical_grouped_money
    ):
        return GoogleTnMatchKind.CANONICAL
    if kind is SpanKind.MEASUREMENT and _same_canonical(
        google_written, realized, _canonical_measurement
    ):
        return GoogleTnMatchKind.CANONICAL
    if kind is SpanKind.TIME and _same_canonical(
        google_written, realized, _canonical_time
    ):
        return GoogleTnMatchKind.CANONICAL
    return None
