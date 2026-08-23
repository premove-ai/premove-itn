from dataclasses import dataclass

from premove_itn.dataset.google_tn import (
    GoogleClassAction,
    google_class_action,
    google_span_kind,
)
from premove_itn.dataset.google_tn_parser import GoogleTnRow, GoogleTnSentence
from premove_itn.labels import SpanKind


@dataclass(frozen=True, slots=True)
class GoogleTnSpanCandidate:
    source_name: str
    sentence_number: int
    line_number: int
    source_class: str
    kind: SpanKind
    written: str
    spoken: str


@dataclass(frozen=True, slots=True)
class GoogleTnCandidateResult:
    sentence: GoogleTnSentence
    span_candidates: tuple[GoogleTnSpanCandidate, ...]
    context_rows: tuple[GoogleTnRow, ...]
    quarantined_rows: tuple[GoogleTnRow, ...]


def extract_google_tn_candidates(
    sentence: GoogleTnSentence,
) -> GoogleTnCandidateResult:
    span_candidates: list[GoogleTnSpanCandidate] = []
    context_rows: list[GoogleTnRow] = []
    quarantined_rows: list[GoogleTnRow] = []

    for row in sentence.rows:
        action = google_class_action(row.source_class)
        if action is GoogleClassAction.CONTEXT:
            context_rows.append(row)
        elif action is GoogleClassAction.SPAN:
            kind = google_span_kind(row.source_class)
            assert kind is not None
            span_candidates.append(
                GoogleTnSpanCandidate(
                    row.source_name,
                    sentence.sentence_number,
                    row.line_number,
                    row.source_class,
                    kind,
                    row.written,
                    row.spoken,
                )
            )
        elif action is GoogleClassAction.QUARANTINE:
            quarantined_rows.append(row)

    return GoogleTnCandidateResult(
        sentence,
        tuple(span_candidates),
        tuple(context_rows),
        tuple(quarantined_rows),
    )
