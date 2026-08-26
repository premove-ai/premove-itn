"""Training-facing views of the shared runtime Model V1 label contract."""

from premove_itn.labels import MODEL_V1_LABEL_CONTRACT

TRAINED_KINDS: tuple[str, ...] = tuple(
    kind.value for kind in MODEL_V1_LABEL_CONTRACT.kinds
)
BIO_LABELS: tuple[str, ...] = MODEL_V1_LABEL_CONTRACT.bio_labels
LABEL_TO_ID: dict[str, int] = dict(MODEL_V1_LABEL_CONTRACT.label_to_id)
ID_TO_LABEL: dict[int, str] = dict(MODEL_V1_LABEL_CONTRACT.id_to_label)
