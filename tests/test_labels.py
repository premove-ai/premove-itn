import pytest

from premove_itn.labels import (
    MODEL_V1_LABEL_CONTRACT,
    SPAN_KINDS,
    SpanKind,
)


def test_span_kinds_have_a_stable_realizer_order() -> None:
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


def test_model_v1_label_contract_has_a_stable_numeric_meaning() -> None:
    assert MODEL_V1_LABEL_CONTRACT.version == 1
    assert MODEL_V1_LABEL_CONTRACT.kinds == (
        SpanKind.CARDINAL,
        SpanKind.DATE,
        SpanKind.DECIMAL,
        SpanKind.DIGIT_SEQUENCE,
        SpanKind.ELECTRONIC,
        SpanKind.MEASUREMENT,
        SpanKind.MONEY,
        SpanKind.ORDINAL,
        SpanKind.PHONE,
        SpanKind.TIME,
    )
    assert MODEL_V1_LABEL_CONTRACT.bio_labels == (
        "O",
        "B-CARDINAL",
        "I-CARDINAL",
        "B-DATE",
        "I-DATE",
        "B-DECIMAL",
        "I-DECIMAL",
        "B-DIGIT_SEQUENCE",
        "I-DIGIT_SEQUENCE",
        "B-ELECTRONIC",
        "I-ELECTRONIC",
        "B-MEASUREMENT",
        "I-MEASUREMENT",
        "B-MONEY",
        "I-MONEY",
        "B-ORDINAL",
        "I-ORDINAL",
        "B-PHONE",
        "I-PHONE",
        "B-TIME",
        "I-TIME",
    )
    assert MODEL_V1_LABEL_CONTRACT.label_to_id["B-CARDINAL"] == 1
    assert MODEL_V1_LABEL_CONTRACT.label_to_id["B-DIGIT_SEQUENCE"] == 7
    assert MODEL_V1_LABEL_CONTRACT.id_to_label[1] == "B-CARDINAL"
    assert len(MODEL_V1_LABEL_CONTRACT.bio_labels) == 21


def test_checkpoint_id2label_is_the_validated_numeric_authority() -> None:
    labels = MODEL_V1_LABEL_CONTRACT.checkpoint_labels(
        id2label={
            str(label_id): label
            for label_id, label in MODEL_V1_LABEL_CONTRACT.id_to_label.items()
        },
        label2id=dict(MODEL_V1_LABEL_CONTRACT.label_to_id),
    )

    assert labels[1] == "B-CARDINAL"
    assert labels[7] == "B-DIGIT_SEQUENCE"


def test_checkpoint_rejects_the_legacy_13_kind_numeric_order() -> None:
    legacy_labels = (
        "O",
        *(f"{prefix}-{kind.value}" for kind in SPAN_KINDS for prefix in ("B", "I")),
    )

    with pytest.raises(ValueError, match="checkpoint id2label does not match"):
        MODEL_V1_LABEL_CONTRACT.checkpoint_labels(
            id2label=dict(enumerate(legacy_labels))
        )


def test_checkpoint_rejects_an_inconsistent_reverse_map() -> None:
    wrong_label2id = dict(MODEL_V1_LABEL_CONTRACT.label_to_id)
    wrong_label2id["B-CARDINAL"] = 7

    with pytest.raises(ValueError, match="checkpoint label2id does not match"):
        MODEL_V1_LABEL_CONTRACT.checkpoint_labels(
            id2label=MODEL_V1_LABEL_CONTRACT.id_to_label,
            label2id=wrong_label2id,
        )
