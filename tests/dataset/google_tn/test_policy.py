import pytest

from premove_itn.dataset.google_tn.policy import (
    GOOGLE_QUARANTINE_CLASSES,
    GOOGLE_SPAN_KIND_MAP,
    GoogleClassAction,
    google_class_action,
    google_span_kind,
)
from premove_itn.labels import SpanKind


def test_google_span_map_is_explicit_and_complete() -> None:
    assert GOOGLE_SPAN_KIND_MAP == {
        "CARDINAL": SpanKind.CARDINAL,
        "DATE": SpanKind.DATE,
        "DECIMAL": SpanKind.DECIMAL,
        "DIGIT": SpanKind.DIGIT_SEQUENCE,
        "ELECTRONIC": SpanKind.ELECTRONIC,
        "MEASURE": SpanKind.MEASUREMENT,
        "MONEY": SpanKind.MONEY,
        "ORDINAL": SpanKind.ORDINAL,
        "TELEPHONE": SpanKind.PHONE,
        "TIME": SpanKind.TIME,
    }


@pytest.mark.parametrize("source_class", ["PLAIN", " plain "])
def test_plain_is_context_not_a_span(source_class: str) -> None:
    assert google_class_action(source_class) is GoogleClassAction.CONTEXT
    assert google_span_kind(source_class) is None


@pytest.mark.parametrize("source_class", sorted(GOOGLE_QUARANTINE_CLASSES))
def test_unsupported_classes_are_quarantined(source_class: str) -> None:
    assert google_class_action(source_class) is GoogleClassAction.QUARANTINE
    assert google_span_kind(source_class) is None


def test_google_punctuation_is_passthrough_context() -> None:
    assert google_class_action("PUNCT") is GoogleClassAction.CONTEXT
    assert google_span_kind("PUNCT") is None


def test_unknown_google_class_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown Google TN class"):
        google_class_action("NEW_CLASS")


def test_source_class_matching_is_case_and_whitespace_tolerant() -> None:
    assert google_span_kind("  time ") is SpanKind.TIME
