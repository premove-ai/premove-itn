"""Deterministic inverse text normalization primitives."""

from premove_itn.labels import SpanKind

from . import _rust


def realize(kind: SpanKind, text: str) -> str | None:
    """Apply one explicit deterministic realizer to the complete input."""
    return _rust.realize(kind.value, text)


def realize_options(kind: SpanKind, text: str) -> list[str]:
    """Return deterministic semantic interpretations for one complete input."""
    return _rust.realize_options(kind.value, text)


def representations_equivalent(kind: SpanKind, canonical: str, observed: str) -> bool:
    """Compare CARDINAL, DATE, DECIMAL, DIGIT_SEQUENCE, MEASUREMENT, MONEY,
    ORDINAL, PHONE, or TIME values.

    Ignore rendering policy while comparing semantic values.
    """
    return _rust.representations_equivalent(kind.value, canonical, observed)


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
    "realize_options",
    "representations_equivalent",
    "tn_normalize",
]
