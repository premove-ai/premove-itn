import pytest

pytest.importorskip("torch")

import torch

from premove_itn.structured_loss import (
    interval_log_partition,
    structured_negative_log_likelihood,
)


def test_interval_log_partition_sums_all_complete_paths() -> None:
    scores = torch.tensor([1.0, 2.0, 3.0])
    spans = torch.tensor([[0, 1], [1, 2], [0, 2]])

    partition = interval_log_partition(scores, spans, sentence_token_count=2)
    expected = torch.logsumexp(torch.tensor([0.0, 1.0, 2.0, 3.0, 3.0]), dim=0)

    assert torch.allclose(partition, expected)


def test_structured_loss_restricts_gold_candidates_and_keep_edges() -> None:
    scores = torch.tensor([1.0, 2.0, 3.0])
    spans = torch.tensor([[0, 1], [1, 2], [0, 2]])

    loss = structured_negative_log_likelihood(
        scores,
        spans,
        sentence_token_count=2,
        gold_candidate_mask=torch.tensor([False, False, True]),
        gold_keep_mask=torch.tensor([False, False]),
    )

    expected = torch.logsumexp(torch.tensor([0.0, 1.0, 2.0, 3.0, 3.0]), dim=0) - 3
    assert torch.allclose(loss, expected)


def test_structured_loss_keeps_multiple_gold_derivations() -> None:
    scores = torch.tensor([1.0, 2.0, 3.0])
    spans = torch.tensor([[0, 1], [1, 2], [0, 2]])

    loss = structured_negative_log_likelihood(
        scores,
        spans,
        sentence_token_count=2,
        gold_candidate_mask=torch.tensor([True, True, True]),
        gold_keep_mask=torch.tensor([False, False]),
    )

    all_paths = torch.logsumexp(torch.tensor([0.0, 1.0, 2.0, 3.0, 3.0]), dim=0)
    gold_paths = torch.logsumexp(torch.tensor([3.0, 3.0]), dim=0)
    assert torch.allclose(loss, all_paths - gold_paths)


def test_interval_log_partition_supports_empty_candidate_sets() -> None:
    scores = torch.empty(0)
    spans = torch.empty((0, 2), dtype=torch.long)

    partition = interval_log_partition(scores, spans, sentence_token_count=3)

    assert partition.item() == 0.0


def test_interval_log_partition_supports_zero_token_sentences() -> None:
    scores = torch.empty(0)
    spans = torch.empty((0, 2), dtype=torch.long)

    partition = interval_log_partition(scores, spans, sentence_token_count=0)

    assert partition.item() == 0.0


def test_structured_loss_backpropagates_through_candidate_scores() -> None:
    scores = torch.tensor([0.5, -0.25], requires_grad=True)
    spans = torch.tensor([[0, 1], [0, 2]])

    loss = structured_negative_log_likelihood(
        scores,
        spans,
        sentence_token_count=2,
        gold_candidate_mask=torch.tensor([False, True]),
        gold_keep_mask=torch.tensor([False, False]),
    )
    loss.backward()

    assert scores.grad is not None
    assert torch.all(torch.isfinite(scores.grad))
    assert torch.any(scores.grad != 0)


def test_structured_loss_rejects_impossible_gold_paths() -> None:
    scores = torch.tensor([1.0])
    spans = torch.tensor([[0, 1]])

    with pytest.raises(ValueError, match="do not cover"):
        structured_negative_log_likelihood(
            scores,
            spans,
            sentence_token_count=2,
            gold_candidate_mask=torch.tensor([True]),
            gold_keep_mask=torch.tensor([False, False]),
        )


def test_interval_log_partition_rejects_mismatched_metadata() -> None:
    with pytest.raises(ValueError, match="equal length"):
        interval_log_partition(
            torch.tensor([1.0]),
            torch.tensor([[0, 1], [1, 2]]),
            sentence_token_count=2,
        )
