from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch
from torch import nn

import premove_itn.training as training_module
from premove_itn import AlignmentState, Candidate, CandidateTransition, SpanKind
from premove_itn.candidate_scorer import collate_candidate_batch
from premove_itn.candidates import GoldGraph
from premove_itn.model_inputs import EncodedCandidates
from premove_itn.training import (
    EpochMetrics,
    TrainingConfig,
    create_optimizer,
    load_checkpoint,
    save_checkpoint,
    train,
    train_epoch,
)
from premove_itn.training_batch import TrainingBatch, TrainingExample


def _training_batch() -> TrainingBatch:
    candidate = Candidate(
        token_start=0,
        token_end=1,
        char_start=0,
        char_end=1,
        text="a",
        replacement="1",
        kinds=(SpanKind.WORD,),
    )
    start = AlignmentState(0, 0)
    end = AlignmentState(1, 1)
    graph = GoldGraph(
        states=(start, end),
        candidate_transitions=(CandidateTransition(start, end, candidate),),
    )
    encoded = EncodedCandidates(
        input_ids=(1, 2),
        attention_mask=(1, 1),
        candidate_token_spans=((0, 1),),
        candidate_replacement_ids=((3,),),
    )
    example = TrainingExample("a", "1", (candidate,), graph, encoded)
    candidate_batch = collate_candidate_batch(
        ((encoded, (candidate,)),),
        pad_token_id=0,
    )
    return TrainingBatch(candidate_batch, (example,))


class ScalarScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, batch) -> torch.Tensor:
        self.calls += 1
        return self.score.expand(batch.candidate_offsets[-1])


def test_create_optimizer_validates_and_uses_adamw() -> None:
    model = ScalarScorer()

    optimizer = create_optimizer(model, TrainingConfig(learning_rate=0.1))

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.1

    with pytest.raises(ValueError, match="learning rate"):
        create_optimizer(model, TrainingConfig(learning_rate=0))


def test_train_epoch_reduces_single_candidate_loss() -> None:
    model = ScalarScorer()
    optimizer = create_optimizer(
        model,
        TrainingConfig(learning_rate=0.5, weight_decay=0),
    )
    batch = _training_batch()

    first = train_epoch(model, (batch,), optimizer, epoch=1, grad_clip_norm=None)
    second = train_epoch(model, (batch,), optimizer, epoch=2, grad_clip_norm=None)

    assert first.steps == 1
    assert first.examples == 1
    assert second.mean_loss < first.mean_loss
    assert model.score.item() > 0


def test_train_recreates_batch_source_and_saves_checkpoint(tmp_path: Path) -> None:
    model = ScalarScorer()
    optimizer = create_optimizer(
        model,
        TrainingConfig(learning_rate=0.1, weight_decay=0),
    )
    batch = _training_batch()
    calls = 0

    def batches():
        nonlocal calls
        calls += 1
        return (batch,)

    checkpoint = tmp_path / "checkpoints" / "step.pt"
    history = train(
        model,
        batches,
        optimizer,
        epochs=2,
        grad_clip_norm=None,
        checkpoint_path=checkpoint,
    )

    assert calls == 2
    assert history == (
        EpochMetrics(1, history[0].mean_loss, 1, 1),
        EpochMetrics(2, history[1].mean_loss, 1, 1),
    )
    assert checkpoint.exists()


def test_checkpoint_round_trip_restores_model_optimizer_and_metrics(
    tmp_path: Path,
) -> None:
    model = ScalarScorer()
    optimizer = create_optimizer(model, TrainingConfig(weight_decay=0))
    optimizer.zero_grad()
    model.score.backward()
    optimizer.step()
    metrics = (EpochMetrics(3, 0.25, 4, 8),)
    checkpoint = tmp_path / "state.pt"

    save_checkpoint(checkpoint, model, optimizer, epoch=3, metrics=metrics)
    restored_model = ScalarScorer()
    restored_optimizer = create_optimizer(
        restored_model, TrainingConfig(weight_decay=0)
    )
    state = load_checkpoint(checkpoint, restored_model, restored_optimizer)

    assert state == type(state)(epoch=3, metrics=metrics)
    assert restored_model.score.item() == model.score.item()
    assert restored_optimizer.state_dict()["state"]


def test_train_epoch_rejects_empty_epoch() -> None:
    model = ScalarScorer()
    optimizer = create_optimizer(model, TrainingConfig(weight_decay=0))

    with pytest.raises(ValueError, match="at least one batch"):
        train_epoch(model, (), optimizer, epoch=1)


def test_train_epoch_accepts_zero_candidate_batch_without_backward() -> None:
    model = ScalarScorer()
    optimizer = create_optimizer(model, TrainingConfig(weight_decay=0))
    states = (AlignmentState(0, 0), AlignmentState(1, 1))
    encoded = EncodedCandidates(
        input_ids=(1, 2),
        attention_mask=(1, 1),
        candidate_token_spans=(),
        candidate_replacement_ids=(),
    )
    example = TrainingExample(
        "a",
        "a",
        (),
        GoldGraph(
            states=states,
            candidate_transitions=(),
            keep_transitions=((states[0], states[1]),),
        ),
        encoded,
    )
    candidate_batch = collate_candidate_batch(((encoded, ()),), pad_token_id=0)
    empty_batch = TrainingBatch(
        candidate_batch=candidate_batch,
        examples=(example,),
    )

    # A zero-candidate batch has no candidate score and therefore no gradient path.
    # The loop should count it without attempting backward().
    metrics = train_epoch(model, (empty_batch,), optimizer, epoch=1)

    assert metrics.mean_loss == 0
    assert metrics.steps == 1
    assert metrics.examples == 1
    assert model.calls == 0


def test_train_epoch_reports_example_weighted_loss(monkeypatch) -> None:
    model = ScalarScorer()
    optimizer = create_optimizer(model, TrainingConfig(weight_decay=0))
    batch = _training_batch()
    larger_batch = TrainingBatch(batch.candidate_batch, (batch.examples[0],) * 2)
    losses = iter((model.score * 0 + 1, model.score * 0 + 3))
    monkeypatch.setattr(
        training_module,
        "structured_batch_loss",
        lambda scorer, current_batch: next(losses),
    )

    metrics = train_epoch(
        model,
        (batch, larger_batch),
        optimizer,
        epoch=1,
        grad_clip_norm=None,
    )

    assert metrics.mean_loss == pytest.approx(7 / 3)
    assert metrics.steps == 2
    assert metrics.examples == 3


class InfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        return value

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.full_like(gradient, float("inf")),)


class InfiniteGradientScorer(ScalarScorer):
    def forward(self, batch) -> torch.Tensor:
        self.calls += 1
        return InfiniteGradient.apply(self.score).expand(batch.candidate_offsets[-1])


def test_train_epoch_rejects_nonfinite_gradients_without_clipping() -> None:
    model = InfiniteGradientScorer()
    optimizer = create_optimizer(model, TrainingConfig(weight_decay=0))

    with pytest.raises(RuntimeError, match="non-finite"):
        train_epoch(
            model,
            (_training_batch(),),
            optimizer,
            epoch=1,
            grad_clip_norm=None,
        )
