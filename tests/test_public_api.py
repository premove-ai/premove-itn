import premove_itn
from premove_itn import SpanKind, realize, realize_options


def test_package_exports_only_deterministic_primitives() -> None:
    assert premove_itn.__all__ == [
        "SpanKind",
        "normalize_sentence",
        "realize",
        "realize_options",
        "tn_normalize",
    ]


def test_realize_routes_an_explicit_kind_to_rust() -> None:
    assert realize(SpanKind.TIME, "four thirty") == "04:30"
    assert realize(SpanKind.DIGIT_SEQUENCE, "zero zero seven") == "007"


def test_realize_options_returns_only_semantic_cardinal_alternatives() -> None:
    assert realize_options(SpanKind.CARDINAL, "two") == ["2"]
    assert realize_options(SpanKind.CARDINAL, "seven eighty eight") == ["95", "788"]
