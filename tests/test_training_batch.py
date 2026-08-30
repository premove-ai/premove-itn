from dataclasses import replace

import pytest

pytest.importorskip("torch")

import torch

from premove_itn import AlignmentState, Candidate, CandidateTransition, SpanKind
from premove_itn.candidate_scorer import collate_candidate_batch
from premove_itn.candidates import GoldGraph
from premove_itn.model_inputs import EncodedCandidates
from premove_itn.training_batch import (
    TrainingBatch,
    TrainingExample,
    prepare_training_batch,
    prepare_training_batches,
    prepare_training_example,
    structured_batch_loss,
)


def _candidate(
    char_start: int,
    char_end: int,
    text: str,
    replacement: str,
) -> Candidate:
    return Candidate(
        token_start=0,
        token_end=1,
        char_start=char_start,
        char_end=char_end,
        text=text,
        replacement=replacement,
        kinds=(SpanKind.WORD,),
    )


def _example(
    text: str,
    candidate: Candidate | None,
    score_span: tuple[int, int] | None,
) -> TrainingExample:
    if candidate is None:
        states = tuple(
            AlignmentState(position, position) for position in range(len(text) + 1)
        )
        graph = GoldGraph(
            states=states,
            candidate_transitions=(),
            keep_transitions=tuple(zip(states[:-1], states[1:], strict=True)),
        )
        candidates = ()
        encoded = EncodedCandidates(
            input_ids=(1, 2),
            attention_mask=(1, 1),
            candidate_token_spans=(),
            candidate_replacement_ids=(),
        )
    else:
        assert score_span is not None
        start = AlignmentState(0, 0)
        end = AlignmentState(len(text), len(candidate.replacement))
        graph = GoldGraph(
            states=(start, end),
            candidate_transitions=(CandidateTransition(start, end, candidate),),
        )
        candidates = (candidate,)
        encoded = EncodedCandidates(
            input_ids=(1, 2, 3),
            attention_mask=(1, 1, 1),
            candidate_token_spans=(score_span,),
            candidate_replacement_ids=((7,),),
        )
    return TrainingExample(
        text, candidate.replacement if candidate else text, candidates, graph, encoded
    )


def test_prepare_training_example_builds_reachable_graph(monkeypatch) -> None:
    candidate = _candidate(0, 1, "a", "1")
    encoded = EncodedCandidates(
        input_ids=(1, 2),
        attention_mask=(1, 1),
        candidate_token_spans=((0, 1),),
        candidate_replacement_ids=((7,),),
    )
    graph = GoldGraph(
        states=(AlignmentState(0, 0), AlignmentState(1, 1)),
        candidate_transitions=(
            CandidateTransition(AlignmentState(0, 0), AlignmentState(1, 1), candidate),
        ),
    )
    monkeypatch.setattr(
        "premove_itn.training_batch.build_candidate_graph",
        lambda text: (candidate,),
    )
    monkeypatch.setattr(
        "premove_itn.training_batch.build_gold_graph",
        lambda text, expected, candidates: graph,
    )
    monkeypatch.setattr(
        "premove_itn.training_batch.encode_candidates",
        lambda text, candidates, tokenizer: encoded,
    )

    example = prepare_training_example("a", "1", object())

    assert example.candidates == (candidate,)
    assert example.gold_graph is graph
    assert example.encoded is encoded


def test_prepare_training_example_rejects_unreachable_target(monkeypatch) -> None:
    monkeypatch.setattr(
        "premove_itn.training_batch.build_candidate_graph", lambda text: ()
    )
    monkeypatch.setattr(
        "premove_itn.training_batch.build_gold_graph",
        lambda text, expected, candidates: None,
    )

    with pytest.raises(ValueError, match="not reachable"):
        prepare_training_example("a", "1", object())


def test_prepare_training_batch_preserves_offsets_and_examples(monkeypatch) -> None:
    first = _example("a", _candidate(0, 1, "a", "1"), (0, 1))
    second = _example("bc", None, None)
    examples = iter((first, second))
    monkeypatch.setattr(
        "premove_itn.training_batch.prepare_training_example",
        lambda text, expected, tokenizer: next(examples),
    )

    batch = prepare_training_batch((("a", "1"), ("bc", "bc")), object(), pad_token_id=0)

    assert batch.examples == (first, second)
    assert batch.candidate_batch.candidate_offsets == (0, 1, 1)


def test_prepare_training_batches_length_buckets_and_chunks(monkeypatch) -> None:
    prepared = {
        text: replace(
            _example(text, None, None),
            encoded=EncodedCandidates(
                input_ids=tuple(range(length)),
                attention_mask=(1,) * length,
                candidate_token_spans=(),
                candidate_replacement_ids=(),
            ),
        )
        for text, length in (("long", 5), ("short", 2), ("medium", 3))
    }
    monkeypatch.setattr(
        "premove_itn.training_batch.prepare_training_example",
        lambda text, expected, tokenizer: prepared[text],
    )

    batches = prepare_training_batches(
        (("long", "long"), ("short", "short"), ("medium", "medium")),
        object(),
        pad_token_id=0,
        batch_size=2,
    )

    assert [[example.text for example in batch.examples] for batch in batches] == [
        ["short", "medium"],
        ["long"],
    ]
    assert [batch.candidate_batch.input_ids.shape[1] for batch in batches] == [3, 5]


def test_prepare_training_batches_can_preserve_input_order(monkeypatch) -> None:
    prepared = {
        text: replace(
            _example(text, None, None),
            encoded=EncodedCandidates(
                input_ids=tuple(range(length)),
                attention_mask=(1,) * length,
                candidate_token_spans=(),
                candidate_replacement_ids=(),
            ),
        )
        for text, length in (("first", 5), ("second", 2))
    }
    monkeypatch.setattr(
        "premove_itn.training_batch.prepare_training_example",
        lambda text, expected, tokenizer: prepared[text],
    )

    batches = prepare_training_batches(
        (("first", "first"), ("second", "second")),
        object(),
        pad_token_id=0,
        batch_size=8,
        length_bucketed=False,
    )

    assert [example.text for example in batches[0].examples] == ["first", "second"]


def test_prepare_training_batches_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="batch size"):
        prepare_training_batches((("a", "a"),), object(), pad_token_id=0, batch_size=0)


def test_structured_batch_loss_slices_scores_by_sentence() -> None:
    first_candidate = _candidate(0, 1, "a", "1")
    first = _example("a", first_candidate, (0, 1))
    second = _example("bc", None, None)
    candidate_batch = collate_candidate_batch(
        ((first.encoded, first.candidates), (second.encoded, second.candidates)),
        pad_token_id=0,
    )
    batch = TrainingBatch(candidate_batch, (first, second))
    scorer_scores = torch.tensor([1.5], requires_grad=True)

    class FixedScorer:
        def __call__(self, batch):
            return scorer_scores

    loss = structured_batch_loss(FixedScorer(), batch)
    expected = (torch.logaddexp(torch.tensor(0.0), scorer_scores) - scorer_scores) / 2

    assert torch.allclose(loss, expected)
    loss.backward()
    assert scorer_scores.grad is not None
    assert torch.all(torch.isfinite(scorer_scores.grad))


def test_structured_batch_loss_rejects_wrong_score_shape() -> None:
    example = _example("a", None, None)
    candidate_batch = collate_candidate_batch(
        ((example.encoded, example.candidates),), pad_token_id=0
    )
    batch = TrainingBatch(candidate_batch, (example,))

    with pytest.raises(ValueError, match="one scalar"):
        structured_batch_loss(lambda _: torch.ones(1), batch)
