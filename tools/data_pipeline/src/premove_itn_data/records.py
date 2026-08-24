from dataclasses import dataclass

from premove_itn.labels import SpanKind
from premove_itn.tokenize import tokenize
from premove_itn.types import WordToken


class TrainingRecordCompileError(ValueError):
    """Raised when spans cannot form a valid training record."""


@dataclass(frozen=True, slots=True)
class TrainingSpan:
    kind: SpanKind
    start: int
    end: int
    source: str
    replacement: str


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    text: str
    expected_text: str
    spans: tuple[TrainingSpan, ...]
    tokens: tuple[WordToken, ...]
    bio_labels: tuple[str, ...]


def compile_training_record(
    text: str,
    spans: tuple[TrainingSpan, ...],
    *,
    expected_text: str | None = None,
) -> TrainingRecord:
    """Validate spans and derive expected text, tokens, and BIO labels."""
    target_parts: list[str] = []
    cursor = 0
    for span in spans:
        if not isinstance(span.kind, SpanKind):
            raise TrainingRecordCompileError(f"unknown span kind: {span.kind!r}")
        if span.start < cursor:
            raise TrainingRecordCompileError(
                "spans must be ordered and non-overlapping"
            )
        if span.start < 0 or span.end <= span.start or span.end > len(text):
            raise TrainingRecordCompileError(
                "span offsets must identify non-empty source text"
            )
        if text[span.start : span.end] != span.source:
            raise TrainingRecordCompileError(
                "span source must match the exact source slice"
            )
        target_parts.append(text[cursor : span.start])
        target_parts.append(span.replacement)
        cursor = span.end

    target_parts.append(text[cursor:])
    compiled_expected_text = "".join(target_parts)
    if expected_text is not None and compiled_expected_text != expected_text:
        raise TrainingRecordCompileError(
            "expected text must equal the result of applying span replacements"
        )

    tokens = tokenize(text)
    labels = ["O"] * len(tokens)
    for span in spans:
        covered = tuple(
            index
            for index, token in enumerate(tokens)
            if token.start >= span.start and token.end <= span.end
        )
        if not covered:
            if any(
                token.end > span.start and token.start < span.end for token in tokens
            ):
                raise TrainingRecordCompileError(
                    "span offsets must align to token boundaries"
                )
            raise TrainingRecordCompileError("each span must cover at least one token")
        if (
            tokens[covered[0]].start != span.start
            or tokens[covered[-1]].end != span.end
        ):
            raise TrainingRecordCompileError(
                "span offsets must align to token boundaries"
            )
        for offset, index in enumerate(covered):
            prefix = "B" if offset == 0 else "I"
            labels[index] = f"{prefix}-{span.kind.value}"

    return TrainingRecord(
        text,
        compiled_expected_text,
        spans,
        tokens,
        tuple(labels),
    )
