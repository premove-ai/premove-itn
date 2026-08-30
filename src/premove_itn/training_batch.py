"""Training-example preparation over candidate and gold graphs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from premove_itn.candidate_scorer import CandidateBatch, collate_candidate_batch
from premove_itn.candidates import (
    Candidate,
    GoldGraph,
    build_candidate_graph,
    build_gold_graph,
)
from premove_itn.model_inputs import (
    EncodedCandidates,
    OffsetTokenizer,
    encode_candidates,
)
from premove_itn.structured_loss import structured_negative_log_likelihood

DEFAULT_TRAIN_BATCH_SIZE = 8


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One encoded sentence with its complete candidate and gold graphs."""

    text: str
    expected_text: str
    candidates: tuple[Candidate, ...]
    gold_graph: GoldGraph
    encoded: EncodedCandidates


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Padded model inputs plus sentence-level structured supervision."""

    candidate_batch: CandidateBatch
    examples: tuple[TrainingExample, ...]

    def to(self, device: torch.device | str) -> TrainingBatch:
        """Move model tensors while keeping graph metadata on the host."""
        return TrainingBatch(
            candidate_batch=self.candidate_batch.to(device),
            examples=self.examples,
        )


class CandidateScoringModel(Protocol):
    """Model interface needed to reduce a batch to structured loss."""

    def __call__(self, batch: CandidateBatch) -> torch.Tensor: ...


def prepare_training_example(
    text: str,
    expected_text: str,
    tokenizer: OffsetTokenizer,
) -> TrainingExample:
    """Build one candidate-encoded example and its exact gold graph."""
    candidates = build_candidate_graph(text)
    gold_graph = build_gold_graph(text, expected_text, candidates)
    if gold_graph is None:
        raise ValueError("expected text is not reachable by the candidate graph")
    encoded = encode_candidates(text, candidates, tokenizer)
    return TrainingExample(text, expected_text, candidates, gold_graph, encoded)


def prepare_training_batch(
    records: Sequence[tuple[str, str]],
    tokenizer: OffsetTokenizer,
    *,
    pad_token_id: int,
) -> TrainingBatch:
    """Prepare and collate a non-empty batch of text/target records."""
    if not records:
        raise ValueError("training batch must contain at least one record")
    examples = tuple(
        prepare_training_example(text, expected_text, tokenizer)
        for text, expected_text in records
    )
    return _collate_training_examples(examples, pad_token_id=pad_token_id)


def prepare_training_batches(
    records: Sequence[tuple[str, str]],
    tokenizer: OffsetTokenizer,
    *,
    pad_token_id: int,
    batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    length_bucketed: bool = True,
) -> tuple[TrainingBatch, ...]:
    """Prepare deterministic, optionally length-bucketed training batches."""
    if not records:
        raise ValueError("training records must contain at least one record")
    if batch_size <= 0:
        raise ValueError("training batch size must be positive")

    examples = tuple(
        prepare_training_example(text, expected_text, tokenizer)
        for text, expected_text in records
    )
    if length_bucketed:
        examples = tuple(
            sorted(examples, key=lambda example: len(example.encoded.input_ids))
        )
    return tuple(
        _collate_training_examples(
            examples[start : start + batch_size],
            pad_token_id=pad_token_id,
        )
        for start in range(0, len(examples), batch_size)
    )


def _collate_training_examples(
    examples: Sequence[TrainingExample],
    *,
    pad_token_id: int,
) -> TrainingBatch:
    """Collate already prepared examples into one model batch."""
    candidate_batch = collate_candidate_batch(
        tuple((example.encoded, example.candidates) for example in examples),
        pad_token_id=pad_token_id,
    )
    return TrainingBatch(candidate_batch, examples)


def structured_batch_loss(
    scorer: CandidateScoringModel,
    batch: TrainingBatch,
) -> torch.Tensor:
    """Score a batch once and average its per-sentence structured losses."""
    scores = scorer(batch.candidate_batch)
    candidate_count = batch.candidate_batch.candidate_offsets[-1]
    if scores.ndim != 1 or scores.shape[0] != candidate_count:
        raise ValueError("scorer must return one scalar per flattened candidate")
    if len(batch.examples) + 1 != len(batch.candidate_batch.candidate_offsets):
        raise ValueError("training examples and candidate offsets must agree")

    losses: list[torch.Tensor] = []
    for sentence_index, example in enumerate(batch.examples):
        start, end = batch.candidate_batch.candidate_offsets[
            sentence_index : sentence_index + 2
        ]
        losses.append(
            structured_negative_log_likelihood(
                scores[start:end],
                example.candidates,
                example.gold_graph,
                source_char_count=len(example.text),
            )
        )
    return torch.stack(losses).mean()
