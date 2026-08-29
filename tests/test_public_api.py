import premove_itn
from premove_itn import SpanKind, realize, realize_options, representations_equivalent


def test_package_exports_only_deterministic_primitives() -> None:
    assert premove_itn.__all__ == [
        "SpanKind",
        "normalize_sentence",
        "realize",
        "realize_options",
        "representations_equivalent",
        "tn_normalize",
    ]


def test_realize_routes_an_explicit_kind_to_rust() -> None:
    assert realize(SpanKind.TIME, "four thirty") == "04:30"
    assert realize(SpanKind.DIGIT_SEQUENCE, "zero zero seven") == "007"


def test_realize_options_returns_only_semantic_cardinal_alternatives() -> None:
    assert realize_options(SpanKind.CARDINAL, "two") == ["2"]
    assert realize_options(SpanKind.CARDINAL, "seven eighty eight") == ["95", "788"]


def test_representations_equivalent_is_separate_from_realization() -> None:
    assert representations_equivalent(SpanKind.CARDINAL, "12345", "12,345")
    assert representations_equivalent(SpanKind.DATE, "4 march 2014", "2014-03-04")
    assert representations_equivalent(SpanKind.TIME, "04:30 p.m.", "4.30 PM")
    assert representations_equivalent(SpanKind.MONEY, "$1000000", "$1M")
    assert not representations_equivalent(SpanKind.CARDINAL, "12345", "12346")
