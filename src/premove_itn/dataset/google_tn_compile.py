from dataclasses import dataclass

from premove_itn.dataset.google_tn import (
    GoogleClassAction,
    google_class_action,
    google_span_kind,
)
from premove_itn.dataset.google_tn_validation import GoogleTnValidationResult
from premove_itn.labels import SpanKind


class GoogleTnCompileError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GoogleTnAssembledSpan:
    kind: SpanKind
    start: int
    end: int
    source: str
    replacement: str


@dataclass(frozen=True, slots=True)
class GoogleTnAssembledSentence:
    text: str
    expected_text: str
    spans: tuple[GoogleTnAssembledSpan, ...]


def assemble_google_tn_sentence(
    validation: GoogleTnValidationResult,
) -> GoogleTnAssembledSentence:
    if not validation.is_accepted:
        raise GoogleTnCompileError("cannot compile an unaccepted Google TN sentence")

    source_parts: list[str] = []
    target_parts: list[str] = []
    spans: list[GoogleTnAssembledSpan] = []
    consumed_candidate_ids: set[int] = set()
    source_offset = 0

    for row in validation.extraction.sentence.rows:
        action = google_class_action(row.source_class)
        if action is GoogleClassAction.CONTEXT:
            source_piece = row.written
            target_piece = row.written
        elif action is GoogleClassAction.SPAN:
            matches = tuple(
                candidate
                for candidate in validation.trusted_candidates
                if candidate.source_name == row.source_name
                and candidate.sentence_number
                == validation.extraction.sentence.sentence_number
                and candidate.line_number == row.line_number
                and candidate.source_class == row.source_class
                and candidate.kind is google_span_kind(row.source_class)
                and candidate.written == row.written
                and candidate.spoken == row.spoken
            )
            if len(matches) != 1:
                raise GoogleTnCompileError(
                    f"span row at line {row.line_number} must have exactly one "
                    "trusted candidate"
                )
            candidate = matches[0]
            consumed_candidate_ids.add(id(candidate))
            source_piece = candidate.spoken
            target_piece = candidate.written
        else:
            raise GoogleTnCompileError("accepted sentence contains a quarantined row")

        if source_parts:
            source_offset += 1
        start = source_offset
        source_offset += len(source_piece)
        source_parts.append(source_piece)
        target_parts.append(target_piece)

        if action is GoogleClassAction.SPAN:
            spans.append(
                GoogleTnAssembledSpan(
                    candidate.kind,
                    start,
                    source_offset,
                    source_piece,
                    target_piece,
                )
            )

    if len(consumed_candidate_ids) != len(validation.trusted_candidates):
        raise GoogleTnCompileError(
            "every trusted candidate must correspond to one sentence row"
        )

    return GoogleTnAssembledSentence(
        " ".join(source_parts), " ".join(target_parts), tuple(spans)
    )
