"""Deterministic inverse text normalization primitives."""

from premove_itn.labels import SpanKind

from . import _rust


def realize(kind: SpanKind, text: str) -> str | None:
    """Apply one explicit deterministic realizer to the complete input."""
    return _rust.realize(kind.value, text)


def normalize_sentence(text: str) -> str:
    """Run the upstream deterministic English ITN sentence normalizer."""
    return _rust.baseline_normalize_sentence(text)


def tn_normalize(text: str) -> str:
    """Run the upstream English text-normalization implementation."""
    return _rust.tn_normalize(text)


__all__ = [
    "SpanKind",
    "normalize_sentence",
    "realize",
    "tn_normalize",
]
