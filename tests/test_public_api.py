import premove_itn
from premove_itn import (
    NormalizationResult,
    NormalizedEdit,
    SpanKind,
    TaggedSpan,
    TextSpan,
    WordPrediction,
    WordToken,
)


def test_package_exports_only_stable_public_contracts() -> None:
    assert premove_itn.__all__ == [
        "NormalizationResult",
        "NormalizedEdit",
        "SpanKind",
        "TaggedSpan",
        "TextSpan",
        "WordPrediction",
        "WordToken",
    ]
    assert SpanKind.TIME == "TIME"
    assert WordToken.__module__ == "premove_itn.types"
    assert TextSpan.__module__ == "premove_itn.types"
    assert WordPrediction.__module__ == "premove_itn.types"
    assert TaggedSpan.__module__ == "premove_itn.types"
    assert NormalizedEdit.__module__ == "premove_itn.types"
    assert NormalizationResult.__module__ == "premove_itn.types"
