from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType


class SpanKind(StrEnum):
    DIGIT_SEQUENCE = "DIGIT_SEQUENCE"
    CARDINAL = "CARDINAL"
    TIME = "TIME"
    DATE = "DATE"
    MONEY = "MONEY"
    DECIMAL = "DECIMAL"
    PHONE = "PHONE"
    ELECTRONIC = "ELECTRONIC"
    MEASUREMENT = "MEASUREMENT"
    ORDINAL = "ORDINAL"
    PUNCTUATION = "PUNCTUATION"
    WHITELIST = "WHITELIST"
    WORD = "WORD"


SPAN_KINDS: tuple[SpanKind, ...] = tuple(SpanKind)


@dataclass(frozen=True, slots=True)
class ModelV1LabelContract:
    """Versioned numeric label meaning for the Model V1 checkpoint."""

    version: int
    kinds: tuple[SpanKind, ...]
    bio_labels: tuple[str, ...] = field(init=False)
    label_to_id: Mapping[str, int] = field(init=False, repr=False)
    id_to_label: Mapping[int, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("label contract version must be positive")
        if (
            not self.kinds
            or any(not isinstance(kind, SpanKind) for kind in self.kinds)
            or len(set(self.kinds)) != len(self.kinds)
        ):
            raise ValueError(
                "label contract kinds must be non-empty unique SpanKind values"
            )

        labels = (
            "O",
            *(f"{prefix}-{kind.value}" for kind in self.kinds for prefix in ("B", "I")),
        )
        label_to_id = {label: label_id for label_id, label in enumerate(labels)}
        id_to_label = {label_id: label for label, label_id in label_to_id.items()}
        object.__setattr__(self, "bio_labels", labels)
        object.__setattr__(self, "label_to_id", MappingProxyType(label_to_id))
        object.__setattr__(self, "id_to_label", MappingProxyType(id_to_label))

    def checkpoint_labels(
        self,
        *,
        id2label: Mapping[int | str, str],
        label2id: Mapping[str, int] | None = None,
    ) -> tuple[str, ...]:
        """Validate checkpoint maps and return its authoritative ID ordering."""
        normalized_id2label: dict[int, str] = {}
        for raw_label_id, label in id2label.items():
            if isinstance(raw_label_id, bool):
                raise ValueError("checkpoint id2label keys must be integer IDs")
            if isinstance(raw_label_id, int):
                label_id = raw_label_id
            elif isinstance(raw_label_id, str) and raw_label_id.isdecimal():
                label_id = int(raw_label_id)
            else:
                raise ValueError("checkpoint id2label keys must be integer IDs")
            if not isinstance(label, str) or label_id in normalized_id2label:
                raise ValueError("checkpoint id2label must map unique IDs to labels")
            normalized_id2label[label_id] = label

        expected_ids = set(range(len(self.bio_labels)))
        if set(normalized_id2label) != expected_ids:
            raise ValueError(
                f"checkpoint id2label does not match Model V1 label contract "
                f"v{self.version}"
            )
        checkpoint_labels = tuple(
            normalized_id2label[label_id] for label_id in range(len(self.bio_labels))
        )
        if checkpoint_labels != self.bio_labels:
            raise ValueError(
                f"checkpoint id2label does not match Model V1 label contract "
                f"v{self.version}"
            )

        if label2id is not None:
            normalized_label2id: dict[str, int] = {}
            for label, label_id in label2id.items():
                if (
                    not isinstance(label, str)
                    or not isinstance(label_id, int)
                    or isinstance(label_id, bool)
                ):
                    raise ValueError(
                        "checkpoint label2id must map labels to integer IDs"
                    )
                normalized_label2id[label] = label_id
            if normalized_label2id != dict(self.label_to_id):
                raise ValueError(
                    f"checkpoint label2id does not match Model V1 label contract "
                    f"v{self.version}"
                )
        return checkpoint_labels


MODEL_V1_LABEL_CONTRACT = ModelV1LabelContract(
    version=1,
    kinds=(
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
    ),
)
