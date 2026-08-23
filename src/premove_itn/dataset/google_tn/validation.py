from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from premove_itn.dataset.google_tn.candidates import (
    GoogleTnCandidateResult,
    GoogleTnSpanCandidate,
)

GoogleTnRealizer = Callable[[str, str], str | None]


class GoogleTnRejectionReason(StrEnum):
    REALIZER_REJECTED = "realizer_rejected"
    WRITTEN_MISMATCH = "written_mismatch"


@dataclass(frozen=True, slots=True)
class GoogleTnCandidateRejection:
    candidate: GoogleTnSpanCandidate
    reason: GoogleTnRejectionReason
    realized: str | None


@dataclass(frozen=True, slots=True)
class GoogleTnValidationResult:
    extraction: GoogleTnCandidateResult
    trusted_candidates: tuple[GoogleTnSpanCandidate, ...]
    rejected_candidates: tuple[GoogleTnCandidateRejection, ...]

    @property
    def is_accepted(self) -> bool:
        return not self.extraction.quarantined_rows and not self.rejected_candidates


def validate_google_tn_candidates(
    extraction: GoogleTnCandidateResult, realizer: GoogleTnRealizer
) -> GoogleTnValidationResult:
    trusted_candidates: list[GoogleTnSpanCandidate] = []
    rejected_candidates: list[GoogleTnCandidateRejection] = []

    for candidate in extraction.span_candidates:
        realized = realizer(candidate.kind.value, candidate.spoken)
        if realized == candidate.written:
            trusted_candidates.append(candidate)
        else:
            reason = (
                GoogleTnRejectionReason.REALIZER_REJECTED
                if realized is None
                else GoogleTnRejectionReason.WRITTEN_MISMATCH
            )
            rejected_candidates.append(
                GoogleTnCandidateRejection(candidate, reason, realized)
            )

    return GoogleTnValidationResult(
        extraction, tuple(trusted_candidates), tuple(rejected_candidates)
    )
