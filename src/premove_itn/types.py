from __future__ import annotations

from dataclasses import dataclass

from premove_itn.labels import LABEL_TO_ID, SpanKind


def _validate_nonempty_offsets(start: int, end: int) -> None:
    if start < 0:
        raise ValueError("start must be non-negative")
    if end <= start:
        raise ValueError("end must be greater than start")


def _validate_score(score: float) -> None:
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be between 0 and 1")


def _validate_source_text(source_text: str, source_span: TextSpan) -> None:
    if len(source_text) != source_span.end - source_span.start:
        raise ValueError("source text length must match its source span")


def _validate_kind(kind: SpanKind) -> None:
    if not isinstance(kind, SpanKind):
        raise ValueError(f"unknown span kind: {kind!r}")


@dataclass(frozen=True, slots=True)
class WordToken:
    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        _validate_nonempty_offsets(self.start, self.end)
        if len(self.text) != self.end - self.start:
            raise ValueError("token text length must match its source span")


@dataclass(frozen=True, slots=True)
class TextSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        _validate_nonempty_offsets(self.start, self.end)


@dataclass(frozen=True, slots=True)
class WordPrediction:
    token: WordToken
    label: str
    score: float

    def __post_init__(self) -> None:
        if self.label not in LABEL_TO_ID:
            raise ValueError(f"unknown BIO label: {self.label!r}")
        _validate_score(self.score)


@dataclass(frozen=True, slots=True)
class TaggedSpan:
    kind: SpanKind
    source_text: str
    span: TextSpan
    score: float

    def __post_init__(self) -> None:
        _validate_kind(self.kind)
        _validate_source_text(self.source_text, self.span)
        _validate_score(self.score)


@dataclass(frozen=True, slots=True)
class NormalizedEdit:
    kind: SpanKind
    source_text: str
    source_span: TextSpan
    normalized_text: str
    score: float

    def __post_init__(self) -> None:
        _validate_kind(self.kind)
        _validate_source_text(self.source_text, self.source_span)
        _validate_score(self.score)


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    source_text: str
    normalized_text: str
    edits: tuple[NormalizedEdit, ...]

    def __post_init__(self) -> None:
        cursor = 0
        normalized_parts: list[str] = []

        for edit in self.edits:
            span = edit.source_span
            if span.start < cursor:
                raise ValueError("edits must be ordered and non-overlapping")
            if span.end > len(self.source_text):
                raise ValueError("edit span exceeds the original source text")
            if self.source_text[span.start : span.end] != edit.source_text:
                raise ValueError(
                    "edit source text must match the original source slice"
                )

            normalized_parts.append(self.source_text[cursor : span.start])
            normalized_parts.append(edit.normalized_text)
            cursor = span.end

        normalized_parts.append(self.source_text[cursor:])
        if "".join(normalized_parts) != self.normalized_text:
            raise ValueError("normalized text must equal the result of applying edits")
