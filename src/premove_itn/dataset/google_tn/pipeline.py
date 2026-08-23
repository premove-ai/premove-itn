from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from premove_itn.dataset.google_tn.candidates import extract_google_tn_candidates
from premove_itn.dataset.google_tn.compile import (
    assemble_google_tn_sentence,
    compile_google_tn_record,
)
from premove_itn.dataset.google_tn.parser import GoogleTnRow, GoogleTnSentence
from premove_itn.dataset.google_tn.validation import (
    GoogleTnCandidateRejection,
    GoogleTnRealizer,
    GoogleTnTrustedCandidate,
    validate_google_tn_candidates,
)
from premove_itn.dataset.records import TrainingRecord


@dataclass(frozen=True, slots=True)
class GoogleTnSentenceOutcome:
    sentence: GoogleTnSentence
    record: TrainingRecord | None
    trusted_candidates: tuple[GoogleTnTrustedCandidate, ...]
    quarantined_rows: tuple[GoogleTnRow, ...]
    candidate_rejections: tuple[GoogleTnCandidateRejection, ...]

    def __post_init__(self) -> None:
        has_rejection = bool(self.quarantined_rows or self.candidate_rejections)
        if (self.record is not None) == has_rejection:
            raise ValueError(
                "record exists if and only if the sentence has no rejection details"
            )

    @property
    def is_accepted(self) -> bool:
        return self.record is not None


def process_google_tn_sentence(
    sentence: GoogleTnSentence, realizer: GoogleTnRealizer
) -> GoogleTnSentenceOutcome:
    extraction = extract_google_tn_candidates(sentence)
    validation = validate_google_tn_candidates(extraction, realizer)
    record = None
    if validation.is_accepted:
        assembled = assemble_google_tn_sentence(validation)
        record = compile_google_tn_record(assembled)

    return GoogleTnSentenceOutcome(
        sentence,
        record,
        validation.trusted_candidates,
        extraction.quarantined_rows,
        validation.rejected_candidates,
    )


def iter_google_tn_outcomes(
    sentences: Iterable[GoogleTnSentence], realizer: GoogleTnRealizer
) -> Iterator[GoogleTnSentenceOutcome]:
    for sentence in sentences:
        yield process_google_tn_sentence(sentence, realizer)
