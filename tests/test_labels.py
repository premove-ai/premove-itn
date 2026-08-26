from premove_itn.labels import SPAN_KINDS, SpanKind


def test_span_kinds_match_the_supported_realizers() -> None:
    assert SPAN_KINDS == (
        SpanKind.DIGIT_SEQUENCE,
        SpanKind.CARDINAL,
        SpanKind.TIME,
        SpanKind.DATE,
        SpanKind.MONEY,
        SpanKind.DECIMAL,
        SpanKind.PHONE,
        SpanKind.ELECTRONIC,
        SpanKind.MEASUREMENT,
        SpanKind.ORDINAL,
        SpanKind.PUNCTUATION,
        SpanKind.WHITELIST,
        SpanKind.WORD,
    )
