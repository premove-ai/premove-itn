"""The frozen Model V1 label contract."""

TRAINED_KINDS: tuple[str, ...] = (
    "CARDINAL",
    "DATE",
    "DECIMAL",
    "DIGIT_SEQUENCE",
    "ELECTRONIC",
    "MEASUREMENT",
    "MONEY",
    "ORDINAL",
    "PHONE",
    "TIME",
)

BIO_LABELS: tuple[str, ...] = (
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

LABEL_TO_ID: dict[str, int] = {
    label: label_id for label_id, label in enumerate(BIO_LABELS)
}
ID_TO_LABEL: dict[int, str] = {
    label_id: label for label, label_id in LABEL_TO_ID.items()
}
