from dataclasses import dataclass

from premove_itn.dataset.google_tn import (
    GoogleClassAction,
    google_class_action,
    google_span_kind,
)
from premove_itn.dataset.google_tn_validation import GoogleTnValidationResult
from premove_itn.labels import SpanKind
from premove_itn.tokenize import tokenize
from premove_itn.types import WordToken


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


@dataclass(frozen=True, slots=True)
class GoogleTnTrainingRecord:
    text: str
    expected_text: str
    spans: tuple[GoogleTnAssembledSpan, ...]
    tokens: tuple[WordToken, ...]
    bio_labels: tuple[str, ...]


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


def compile_google_tn_record(
    assembled: GoogleTnAssembledSentence,
) -> GoogleTnTrainingRecord:
    target_parts: list[str] = []
    cursor = 0
    for span in assembled.spans:
        if span.start < cursor:
            raise GoogleTnCompileError("spans must be ordered and non-overlapping")
        if span.start < 0 or span.end <= span.start or span.end > len(assembled.text):
            raise GoogleTnCompileError(
                "span offsets must identify non-empty source text"
            )
        if assembled.text[span.start : span.end] != span.source:
            raise GoogleTnCompileError("span source must match the exact source slice")
        target_parts.append(assembled.text[cursor : span.start])
        target_parts.append(span.replacement)
        cursor = span.end

    target_parts.append(assembled.text[cursor:])
    if "".join(target_parts) != assembled.expected_text:
        raise GoogleTnCompileError(
            "expected text must equal the result of applying span replacements"
        )

    tokens = tokenize(assembled.text)
    labels = ["O"] * len(tokens)

    for span in assembled.spans:
        covered = tuple(
            index
            for index, token in enumerate(tokens)
            if token.start >= span.start and token.end <= span.end
        )
        if not covered:
            if any(
                token.end > span.start and token.start < span.end for token in tokens
            ):
                raise GoogleTnCompileError(
                    "span offsets must align to token boundaries"
                )
            raise GoogleTnCompileError("each span must cover at least one token")
        if (
            tokens[covered[0]].start != span.start
            or tokens[covered[-1]].end != span.end
        ):
            raise GoogleTnCompileError("span offsets must align to token boundaries")
        for offset, index in enumerate(covered):
            prefix = "B" if offset == 0 else "I"
            labels[index] = f"{prefix}-{span.kind.value}"

    return GoogleTnTrainingRecord(
        assembled.text,
        assembled.expected_text,
        assembled.spans,
        tokens,
        tuple(labels),
    )
