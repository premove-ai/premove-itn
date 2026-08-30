import pytest

pytest.importorskip("torch")

import torch

from premove_itn import AlignmentState, Candidate, CandidateTransition, SpanKind
from premove_itn.candidates import GoldGraph
from premove_itn.structured_loss import (
    all_paths_log_partition,
    gold_paths_log_partition,
    structured_negative_log_likelihood,
)


def _candidate(
    token_start: int,
    token_end: int,
    char_start: int,
    char_end: int,
    text: str,
    replacement: str,
) -> Candidate:
    return Candidate(
        token_start,
        token_end,
        char_start,
        char_end,
        text,
        replacement,
        (SpanKind.WORD,),
    )


def test_all_paths_log_partition_sums_overlapping_candidates() -> None:
    scores = torch.tensor([1.0, 2.0, 3.0])
    spans = torch.tensor([[0, 1], [1, 2], [0, 2]])

    partition = all_paths_log_partition(scores, spans, source_char_count=2)
    expected = torch.logsumexp(torch.tensor([0.0, 1.0, 2.0, 3.0, 3.0]), dim=0)

    assert torch.allclose(partition, expected)


def test_gold_partition_preserves_source_target_states() -> None:
    first_short = _candidate(0, 1, 0, 1, "A", "1")
    first_long = _candidate(0, 1, 0, 1, "A", "12")
    second_short = _candidate(1, 2, 1, 2, "B", "3")
    second_long = _candidate(1, 2, 1, 2, "B", "23")
    candidates = (first_short, first_long, second_short, second_long)
    start = AlignmentState(0, 0)
    after_short = AlignmentState(1, 1)
    after_long = AlignmentState(1, 2)
    end = AlignmentState(2, 3)
    graph = GoldGraph(
        states=(start, after_short, after_long, end),
        candidate_transitions=(
            CandidateTransition(start, after_short, first_short),
            CandidateTransition(after_short, end, second_long),
            CandidateTransition(start, after_long, first_long),
            CandidateTransition(after_long, end, second_short),
        ),
    )
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0])

    partition = gold_paths_log_partition(scores, candidates, graph)
    expected = torch.logsumexp(torch.tensor([1.0 + 4.0, 2.0 + 3.0]), dim=0)

    assert torch.allclose(partition, expected)


def test_structured_loss_uses_source_character_spans() -> None:
    candidate = _candidate(9, 10, 1, 3, "ab", "x")
    graph = GoldGraph(
        states=(AlignmentState(0, 0), AlignmentState(1, 1), AlignmentState(3, 1)),
        candidate_transitions=(
            CandidateTransition(AlignmentState(1, 1), AlignmentState(3, 1), candidate),
        ),
        keep_transitions=((AlignmentState(0, 0), AlignmentState(1, 1)),),
    )

    loss = structured_negative_log_likelihood(
        torch.tensor([2.0]),
        (candidate,),
        graph,
        source_char_count=3,
    )

    expected = torch.logaddexp(torch.tensor(0.0), torch.tensor(2.0)) - 2
    assert torch.allclose(loss, expected)


def test_structured_loss_keeps_multiple_exact_gold_derivations() -> None:
    first = _candidate(0, 1, 0, 1, "A", "1")
    second = _candidate(1, 2, 1, 2, "B", "23")
    whole = _candidate(0, 2, 0, 2, "AB", "123")
    candidates = (first, second, whole)
    start = AlignmentState(0, 0)
    middle = AlignmentState(1, 1)
    end = AlignmentState(2, 3)
    graph = GoldGraph(
        states=(start, middle, end),
        candidate_transitions=(
            CandidateTransition(start, middle, first),
            CandidateTransition(middle, end, second),
            CandidateTransition(start, end, whole),
        ),
    )
    scores = torch.tensor([1.0, 2.0, 3.0])

    partition = gold_paths_log_partition(scores, candidates, graph)

    assert torch.allclose(
        partition, torch.logaddexp(torch.tensor(3.0), torch.tensor(3.0))
    )


def test_keep_only_sentence_has_zero_partition_and_loss() -> None:
    states = tuple(AlignmentState(position, position) for position in range(4))
    graph = GoldGraph(
        states=states,
        candidate_transitions=(),
        keep_transitions=tuple(zip(states[:-1], states[1:], strict=True)),
    )
    scores = torch.empty(0)
    spans = torch.empty((0, 2), dtype=torch.long)

    assert all_paths_log_partition(scores, spans, 3).item() == 0
    assert gold_paths_log_partition(scores, (), graph).item() == 0
    assert structured_negative_log_likelihood(scores, (), graph, 3).item() == 0


def test_structured_loss_has_finite_gradients_for_sparse_gold_graph() -> None:
    first = _candidate(0, 1, 0, 1, "A", "1")
    second = _candidate(1, 2, 1, 2, "B", "2")
    whole = _candidate(0, 2, 0, 2, "AB", "12")
    candidates = (first, second, whole)
    start = AlignmentState(0, 0)
    end = AlignmentState(2, 2)
    graph = GoldGraph(
        states=(start, end),
        candidate_transitions=(CandidateTransition(start, end, whole),),
    )
    scores = torch.tensor([0.5, -0.25, 1.0], requires_grad=True)

    loss = structured_negative_log_likelihood(scores, candidates, graph, 2)
    loss.backward()

    assert scores.grad is not None
    assert torch.all(torch.isfinite(scores.grad))


def test_gold_partition_rejects_unknown_candidate() -> None:
    graph_candidate = _candidate(0, 1, 0, 1, "A", "1")
    batch_candidate = _candidate(0, 1, 0, 1, "A", "2")
    graph = GoldGraph(
        states=(AlignmentState(0, 0), AlignmentState(1, 1)),
        candidate_transitions=(
            CandidateTransition(
                AlignmentState(0, 0), AlignmentState(1, 1), graph_candidate
            ),
        ),
    )

    with pytest.raises(ValueError, match="outside the batch"):
        gold_paths_log_partition(torch.tensor([1.0]), (batch_candidate,), graph)


def test_all_paths_rejects_mismatched_metadata() -> None:
    with pytest.raises(ValueError, match="equal length"):
        all_paths_log_partition(
            torch.tensor([1.0]),
            torch.tensor([[0, 1], [1, 2]]),
            source_char_count=2,
        )
