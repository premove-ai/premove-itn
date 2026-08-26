from premove_itn_training.labels import BIO_LABELS, ID_TO_LABEL, LABEL_TO_ID

from premove_itn.labels import MODEL_V1_LABEL_CONTRACT


def test_model_v1_label_contract_is_explicit_and_stable() -> None:
    assert BIO_LABELS == (
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
    assert {label: index for index, label in enumerate(BIO_LABELS)} == LABEL_TO_ID
    assert {index: label for label, index in LABEL_TO_ID.items()} == ID_TO_LABEL
    assert BIO_LABELS is MODEL_V1_LABEL_CONTRACT.bio_labels
    assert dict(MODEL_V1_LABEL_CONTRACT.label_to_id) == LABEL_TO_ID
    assert dict(MODEL_V1_LABEL_CONTRACT.id_to_label) == ID_TO_LABEL
