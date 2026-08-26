import premove_itn
from premove_itn import SpanKind, realize


def test_package_exports_only_deterministic_primitives() -> None:
    assert premove_itn.__all__ == [
        "SpanKind",
        "normalize_sentence",
        "realize",
        "tn_normalize",
    ]


def test_realize_routes_an_explicit_kind_to_rust() -> None:
    assert realize(SpanKind.TIME, "four thirty") == "04:30"
    assert realize(SpanKind.DIGIT_SEQUENCE, "zero zero seven") == "007"
