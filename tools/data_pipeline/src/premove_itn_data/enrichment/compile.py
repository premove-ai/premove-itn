from __future__ import annotations

from premove_itn_data.enrichment.generate import EnrichmentCandidate
from premove_itn_data.records import (
    TrainingRecord,
    TrainingRecordCompileError,
    TrainingSpan,
    compile_training_record,
)


class EnrichmentCompileError(ValueError):
    """Raised when an enrichment candidate cannot form a training record."""


def compile_enrichment_candidate(candidate: EnrichmentCandidate) -> TrainingRecord:
    """Validate one enrichment candidate and derive its training record."""
    if not candidate.text:
        raise EnrichmentCompileError("candidate text must be non-empty")
    if not candidate.context_provenance:
        raise EnrichmentCompileError("candidate context provenance must be non-empty")

    spans: list[TrainingSpan] = []
    for span in candidate.spans:
        if not span.source or not span.replacement:
            raise EnrichmentCompileError("candidate span values must be non-empty")
        if not span.context_provenance or not span.donor_provenance:
            raise EnrichmentCompileError("candidate span provenance must be non-empty")
        spans.append(
            TrainingSpan(
                kind=span.kind,
                start=span.start,
                end=span.end,
                source=span.source,
                replacement=span.replacement,
            )
        )

    try:
        return compile_training_record(candidate.text, tuple(spans))
    except TrainingRecordCompileError as error:
        raise EnrichmentCompileError(str(error)) from error
