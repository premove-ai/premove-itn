from __future__ import annotations

from typing import Any

import pytest
import torch
from premove_itn_training.alignment import (
    IGNORE_INDEX,
    EncodedDataset,
    LabelPaddingCollator,
    UnsafeTruncationError,
    align_word_labels,
)
from premove_itn_training.dataset import TrainingExample
from premove_itn_training.labels import LABEL_TO_ID


class _Encoding(dict[str, list[int]]):
    def __init__(self, values: dict[str, list[int]], word_ids: list[int | None]):
        super().__init__(values)
        self._word_ids = word_ids

    def word_ids(self) -> list[int | None]:
        return self._word_ids


class _Tokenizer:
    is_fast = True
    padding_side = "right"

    def __init__(self, pieces_by_token: dict[str, int] | None = None) -> None:
        self.pieces_by_token = pieces_by_token or {}

    def __call__(self, tokens: list[str], **kwargs: Any) -> _Encoding:
        assert kwargs == {
            "is_split_into_words": True,
            "add_special_tokens": True,
            "padding": False,
            "truncation": False,
        }
        input_ids = [101]
        word_ids: list[int | None] = [None]
        for word_id, token in enumerate(tokens):
            piece_count = self.pieces_by_token.get(token, 1)
            input_ids.extend([1000 + word_id] * piece_count)
            word_ids.extend([word_id] * piece_count)
        input_ids.append(102)
        word_ids.append(None)
        return _Encoding(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            },
            word_ids,
        )

    def pad(
        self,
        features: list[dict[str, list[int]]],
        *,
        padding: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding is True
        assert return_tensors == "pt"
        maximum = max(len(feature["input_ids"]) for feature in features)
        return {
            "input_ids": torch.tensor(
                [
                    feature["input_ids"]
                    + [0] * (maximum - len(feature["input_ids"]))
                    for feature in features
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    feature["attention_mask"]
                    + [0] * (maximum - len(feature["attention_mask"]))
                    for feature in features
                ]
            ),
        }


def _example(tokens: tuple[str, ...], labels: tuple[str, ...]) -> TrainingExample:
    return TrainingExample(
        identity="a" * 64,
        tokens=tokens,
        label_ids=tuple(LABEL_TO_ID[label] for label in labels),
        context_only=all(label == "O" for label in labels),
        multi_span=False,
    )


def test_alignment_supervises_only_each_words_first_subword() -> None:
    labels = [LABEL_TO_ID["B-TIME"], LABEL_TO_ID["I-TIME"]]

    assert align_word_labels([None, 0, 1, 1, None], labels) == [
        IGNORE_INDEX,
        LABEL_TO_ID["B-TIME"],
        LABEL_TO_ID["I-TIME"],
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]


def test_alignment_rejects_omitted_source_words() -> None:
    with pytest.raises(ValueError, match="omitted source word IDs"):
        align_word_labels([None, 0, None], [0, 0])


def test_encoding_fails_closed_instead_of_truncating() -> None:
    example = _example(("long", "utterance"), ("O", "O"))
    tokenizer = _Tokenizer({"long": 3, "utterance": 2})

    with pytest.raises(UnsafeTruncationError, match="exceed max_length=6"):
        EncodedDataset([example], tokenizer, max_length=6)


def test_batching_pads_inputs_and_labels_independently() -> None:
    tokenizer = _Tokenizer({"thirty": 2})
    first = EncodedDataset(
        [_example(("four", "thirty"), ("B-TIME", "I-TIME"))],
        tokenizer,
        max_length=8,
    )[0]
    second = EncodedDataset(
        [_example(("hello",), ("O",))], tokenizer, max_length=8
    )[0]

    batch = LabelPaddingCollator(tokenizer)([first, second])

    assert batch["input_ids"].shape == batch["attention_mask"].shape == (2, 5)
    assert batch["labels"].tolist()[0] == first["labels"]
    assert batch["labels"].tolist()[1] == second["labels"] + [
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]
