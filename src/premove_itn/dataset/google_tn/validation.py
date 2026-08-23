from collections.abc import Callable
from dataclasses import dataclass
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


def _google_written_match_kind(
    kind: SpanKind, google_written: str, realized: str
) -> GoogleTnMatchKind | None:
    if google_written == realized:
        return GoogleTnMatchKind.EXACT
    if kind is SpanKind.DATE and _canonical_date_written(
        google_written
    ) == _canonical_date_written(realized):
        return GoogleTnMatchKind.CANONICAL
    return None
