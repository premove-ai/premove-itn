from premove_itn.labels import (
    BIO_LABELS,
    ID_TO_LABEL,
    LABEL_TO_ID,
    SPAN_KINDS,
    SpanKind,
)


def test_span_kinds_have_a_stable_model_order() -> None:
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


def test_bio_labels_have_a_stable_model_order() -> None:
    assert BIO_LABELS == (
        "O",
        "B-DIGIT_SEQUENCE",
        "I-DIGIT_SEQUENCE",
        "B-CARDINAL",
        "I-CARDINAL",
        "B-TIME",
        "I-TIME",
        "B-DATE",
        "I-DATE",
        "B-MONEY",
        "I-MONEY",
        "B-DECIMAL",
        "I-DECIMAL",
        "B-PHONE",
        "I-PHONE",
        "B-ELECTRONIC",
        "I-ELECTRONIC",
        "B-MEASUREMENT",
        "I-MEASUREMENT",
        "B-ORDINAL",
        "I-ORDINAL",
        "B-PUNCTUATION",
        "I-PUNCTUATION",
        "B-WHITELIST",
        "I-WHITELIST",
        "B-WORD",
        "I-WORD",
    )


def test_label_id_mappings_preserve_saved_model_meaning() -> None:
    assert LABEL_TO_ID["O"] == 0
    assert LABEL_TO_ID["B-DIGIT_SEQUENCE"] == 1
    assert LABEL_TO_ID["I-DIGIT_SEQUENCE"] == 2
    assert LABEL_TO_ID["B-CARDINAL"] == 3
    assert LABEL_TO_ID["B-WORD"] == 25
    assert LABEL_TO_ID["I-WORD"] == 26
    assert ID_TO_LABEL[0] == "O"
    assert ID_TO_LABEL[26] == "I-WORD"
    assert len(LABEL_TO_ID) == len(ID_TO_LABEL) == 27
