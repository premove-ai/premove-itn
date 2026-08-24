from dataclasses import dataclass

from premove_itn.labels import SpanKind
from premove_itn.types import WordToken


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
