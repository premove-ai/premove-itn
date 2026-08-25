from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from premove_itn_training.dataset import TrainingExample

IGNORE_INDEX = -100
_MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "token_type_ids")


class UnsafeTruncationError(ValueError):
    def __init__(
        self,
        *,
        max_length: int,
        longest: int,
        offenders: Sequence[tuple[str, int]],
    ) -> None:
        preview = ", ".join(
            f"{identity[:12]} ({length})" for identity, length in offenders[:5]
        )
        super().__init__(
            f"{len(offenders)} records exceed max_length={max_length}; "
            f"longest={longest}; first offenders: {preview}"
        )
        self.max_length = max_length
        self.longest = longest
        self.offenders = tuple(offenders)


@dataclass(frozen=True, slots=True)
class LengthAudit:
    records: int
    minimum: int
    p50: int
    p90: int
    p95: int
    p99: int
    maximum: int
    records_with_unknown_subwords: int
    unknown_subwords: int


def _percentile(sorted_lengths: Sequence[int], percentile: float) -> int:
    if not sorted_lengths:
        return 0
    index = round((len(sorted_lengths) - 1) * percentile)
    return sorted_lengths[index]


def _length_audit(
    lengths: Sequence[int],
    *,
    records_with_unknown_subwords: int,
    unknown_subwords: int,
) -> LengthAudit:
    ordered = sorted(lengths)
    if not ordered:
        return LengthAudit(0, 0, 0, 0, 0, 0, 0, 0, 0)
    return LengthAudit(
        records=len(ordered),
        minimum=ordered[0],
        p50=_percentile(ordered, 0.50),
        p90=_percentile(ordered, 0.90),
        p95=_percentile(ordered, 0.95),
        p99=_percentile(ordered, 0.99),
        maximum=ordered[-1],
        records_with_unknown_subwords=records_with_unknown_subwords,
        unknown_subwords=unknown_subwords,
    )


def align_word_labels(
    word_ids: Sequence[int | None], word_label_ids: Sequence[int]
) -> list[int]:
    """Apply loss only to the first model subword for each source word."""
    aligned: list[int] = []
    observed: set[int] = set()
    previous_word_id = -1
    for word_id in word_ids:
        if word_id is None:
            aligned.append(IGNORE_INDEX)
            continue
        if word_id < 0 or word_id >= len(word_label_ids):
            raise ValueError(f"tokenizer returned invalid word_id {word_id}")
        if word_id < previous_word_id:
            raise ValueError("tokenizer word_ids must be non-decreasing")
        if word_id in observed:
            aligned.append(IGNORE_INDEX)
        else:
            aligned.append(word_label_ids[word_id])
            observed.add(word_id)
        previous_word_id = word_id

    expected = set(range(len(word_label_ids)))
    if observed != expected:
        missing = sorted(expected - observed)
        raise ValueError(f"tokenizer omitted source word IDs: {missing}")
    return aligned


class EncodedDataset(Sequence[dict[str, list[int]]]):
    def __init__(
        self,
        examples: Sequence[TrainingExample],
        tokenizer: Any,
        *,
        max_length: int,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Model V1 requires a fast tokenizer with word_ids()")

        items: list[dict[str, list[int]]] = []
        lengths: list[int] = []
        offenders: list[tuple[str, int]] = []
        unknown_subwords = 0
        records_with_unknown_subwords = 0
        unknown_token_id = getattr(tokenizer, "unk_token_id", None)
        for example in examples:
            encoded = tokenizer(
                list(example.tokens),
                is_split_into_words=True,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
            input_ids = list(encoded["input_ids"])
            length = len(input_ids)
            lengths.append(length)
            record_unknown_subwords = (
                input_ids.count(unknown_token_id)
                if isinstance(unknown_token_id, int)
                else 0
            )
            unknown_subwords += record_unknown_subwords
            records_with_unknown_subwords += bool(record_unknown_subwords)
            if length > max_length:
                offenders.append((example.identity, length))
                continue
            word_ids = encoded.word_ids()
            if len(word_ids) != length:
                raise ValueError("tokenizer input_ids and word_ids lengths must match")
            item = {
                key: list(encoded[key])
                for key in _MODEL_INPUT_KEYS
                if key in encoded
            }
            item["labels"] = align_word_labels(word_ids, example.label_ids)
            if any(len(values) != length for values in item.values()):
                raise ValueError("encoded model fields must have equal lengths")
            items.append(item)

        self.audit = _length_audit(
            lengths,
            records_with_unknown_subwords=records_with_unknown_subwords,
            unknown_subwords=unknown_subwords,
        )
        if offenders:
            raise UnsafeTruncationError(
                max_length=max_length,
                longest=self.audit.maximum,
                offenders=offenders,
            )
        self._items = tuple(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self._items[index]


class LabelPaddingCollator:
    """Dynamically pad model inputs and use -100 for label padding."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> Any:
        import torch

        labels = [list(feature["labels"]) for feature in features]
        model_features = [
            {key: list(value) for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            return_tensors="pt",
        )
        sequence_length = int(batch["input_ids"].shape[1])
        padding_side = getattr(self.tokenizer, "padding_side", "right")
        padded_labels = []
        for label_ids in labels:
            padding = [IGNORE_INDEX] * (sequence_length - len(label_ids))
            padded_labels.append(
                label_ids + padding if padding_side == "right" else padding + label_ids
            )
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch
