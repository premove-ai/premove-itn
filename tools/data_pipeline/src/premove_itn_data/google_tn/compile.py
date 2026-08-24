from dataclasses import dataclass

from premove_itn_data.google_tn.policy import (
    GoogleClassAction,
    google_class_action,
    google_span_kind,
)
from premove_itn_data.google_tn.validation import GoogleTnValidationResult
from premove_itn_data.records import (
    TrainingRecord,
    TrainingRecordCompileError,
    TrainingSpan,
    compile_training_record,
)

NO_SPACE_BEFORE = frozenset({".", ",", ";", ":", "!", "?", "%", ")", "]", "}"})
NO_SPACE_AFTER = frozenset({"(", "[", "{"})


class GoogleTnCompileError(ValueError):
    pass


def _needs_space(previous: str | None, current: str) -> bool:
    if previous is None:
        return False
    if current in NO_SPACE_BEFORE:
        return False
    return previous not in NO_SPACE_AFTER


@dataclass(frozen=True, slots=True)
class GoogleTnAssembledSentence:
    text: str
    expected_text: str
    spans: tuple[TrainingSpan, ...]


def assemble_google_tn_sentence(
    validation: GoogleTnValidationResult,
) -> GoogleTnAssembledSentence:
    if not validation.is_accepted:
        raise GoogleTnCompileError("cannot compile an unaccepted Google TN sentence")

    source_parts: list[str] = []
    target_parts: list[str] = []
    spans: list[TrainingSpan] = []
    consumed_candidate_ids: set[int] = set()
    source_offset = 0
    previous_source_piece: str | None = None
    previous_target_piece: str | None = None

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
                and candidate.google_written == row.written
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
            target_piece = candidate.realized
        else:
            raise GoogleTnCompileError("accepted sentence contains a quarantined row")

        if _needs_space(previous_source_piece, source_piece):
            source_parts.append(" ")
            source_offset += 1
        start = source_offset
        source_offset += len(source_piece)
        source_parts.append(source_piece)

        if _needs_space(previous_target_piece, target_piece):
            target_parts.append(" ")
        target_parts.append(target_piece)
        previous_source_piece = source_piece
        previous_target_piece = target_piece

        if action is GoogleClassAction.SPAN:
            spans.append(
                TrainingSpan(
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
        "".join(source_parts), "".join(target_parts), tuple(spans)
    )


def compile_google_tn_record(
    assembled: GoogleTnAssembledSentence,
) -> TrainingRecord:
    try:
        return compile_training_record(
            assembled.text,
            assembled.spans,
            expected_text=assembled.expected_text,
        )
    except TrainingRecordCompileError as error:
        raise GoogleTnCompileError(str(error)) from error
