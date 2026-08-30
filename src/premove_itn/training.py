"""Minimal optimizer loop for the contextual candidate experiment."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from premove_itn.training_batch import TrainingBatch, structured_batch_loss

DEFAULT_CHECKPOINT_EVERY_STEPS = 500


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimizer settings for the first experiment."""

    learning_rate: float = 2e-5
    weight_decay: float = 0.01


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Aggregate metrics for one training epoch."""

    epoch: int
    mean_loss: float
    steps: int
    examples: int


@dataclass(frozen=True, slots=True)
class CheckpointState:
    """Metadata restored alongside model and optimizer parameters."""

    epoch: int
    metrics: tuple[EpochMetrics, ...]
    step: int = 0
    batch_offset: int = 0
    training_distribution: tuple[tuple[str, int], ...] = ()


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create the AdamW optimizer used by the first training experiment."""
    if config.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight decay must be non-negative")
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def train_epoch(
    model: nn.Module,
    batches: Iterable[TrainingBatch],
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    device: torch.device | str | None = None,
    grad_clip_norm: float | None = 1.0,
    checkpoint_path: str | Path | None = None,
    checkpoint_every_steps: int | None = None,
    previous_metrics: tuple[EpochMetrics, ...] = (),
    training_distribution: Mapping[str, int] | None = None,
    resume_state: CheckpointState | None = None,
    source_batch_offset: int = 0,
) -> EpochMetrics:
    """Run one optimizer epoch over prepared training batches.

    The caller must move ``model`` to ``device`` before creating the optimizer.
    This function moves only each batch's model tensors.

    When configured, checkpoints are written after each ``checkpoint_every_steps``
    optimizer updates. The checkpoint step is the number of updates completed in
    this epoch; batches without candidates do not advance it. A
    ``training_distribution`` is required when checkpointing so saved metadata
    cannot silently omit per-kind exposure. ``resume_state`` requires the same
    deterministic batch order and skips every batch already represented by the
    restored model and optimizer state.
    """
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("gradient clip norm must be positive")
    if checkpoint_every_steps is not None and checkpoint_every_steps <= 0:
        raise ValueError("checkpoint interval must be positive")
    if checkpoint_path is not None and training_distribution is None:
        raise ValueError("training distribution is required when checkpointing")
    if source_batch_offset < 0:
        raise ValueError("source batch offset must be non-negative")
    normalized_distribution = _normalize_training_distribution(training_distribution)

    resume_batch_offset = 0
    if resume_state is not None:
        if resume_state.epoch != epoch or resume_state.batch_offset <= 0:
            raise ValueError("resume checkpoint must be inside the requested epoch")
        if not resume_state.metrics:
            raise ValueError("resume checkpoint must contain partial epoch metrics")
        partial_metrics = resume_state.metrics[-1]
        if partial_metrics.epoch != epoch:
            raise ValueError("resume metrics must match the requested epoch")
        if partial_metrics.steps != resume_state.batch_offset:
            raise ValueError("resume batch offset must match partial epoch steps")
        if previous_metrics and previous_metrics != resume_state.metrics[:-1]:
            raise ValueError("previous metrics do not match the resume checkpoint")
        if (
            normalized_distribution
            and normalized_distribution != resume_state.training_distribution
        ):
            raise ValueError("training distribution does not match the checkpoint")
        previous_metrics = resume_state.metrics[:-1]
        resume_batch_offset = resume_state.batch_offset
        if source_batch_offset > resume_batch_offset:
            raise ValueError("source starts after the resume checkpoint")
        total_loss = partial_metrics.mean_loss * partial_metrics.examples
        step_count = partial_metrics.steps
        optimizer_step_count = resume_state.step
        example_count = partial_metrics.examples
    else:
        if source_batch_offset:
            raise ValueError("source batch offset requires a resume checkpoint")
        total_loss = 0.0
        step_count = 0
        optimizer_step_count = 0
        example_count = 0

    model.train()
    batches_seen = source_batch_offset
    for local_batch_index, batch in enumerate(batches):
        batch_index = source_batch_offset + local_batch_index
        batches_seen = batch_index + 1
        if batch_index < resume_batch_offset:
            continue
        model_batch = batch if device is None else batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        candidate_count = batch.candidate_batch.candidate_offsets[-1]
        if candidate_count == 0:
            step_count += 1
            example_count += len(batch.examples)
            continue
        loss = structured_batch_loss(model, model_batch)
        if not loss.requires_grad:
            raise RuntimeError("training loss is detached from candidate scores")
        loss.backward()
        max_norm = grad_clip_norm if grad_clip_norm is not None else float("inf")
        nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)
        optimizer.step()
        weighted_loss = loss.detach() * len(batch.examples)
        total_loss = total_loss + weighted_loss
        step_count += 1
        optimizer_step_count += 1
        example_count += len(batch.examples)
        if (
            checkpoint_path is not None
            and checkpoint_every_steps is not None
            and optimizer_step_count % checkpoint_every_steps == 0
        ):
            partial_metrics = EpochMetrics(
                epoch=epoch,
                mean_loss=_mean_loss(total_loss, example_count),
                steps=step_count,
                examples=example_count,
            )
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=epoch,
                step=optimizer_step_count,
                batch_offset=step_count,
                metrics=previous_metrics + (partial_metrics,),
                training_distribution=normalized_distribution,
            )

    if batches_seen < resume_batch_offset:
        raise ValueError("batch source ended before the resume offset")
    if step_count == 0:
        raise ValueError("training epoch must contain at least one batch")
    return EpochMetrics(
        epoch=epoch,
        mean_loss=_mean_loss(total_loss, example_count),
        steps=step_count,
        examples=example_count,
    )


def train(
    model: nn.Module,
    batch_source: Callable[[], Iterable[TrainingBatch]],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    device: torch.device | str | None = None,
    grad_clip_norm: float | None = 1.0,
    checkpoint_path: str | Path | None = None,
    checkpoint_every_steps: int | None = DEFAULT_CHECKPOINT_EVERY_STEPS,
    start_epoch: int = 0,
    previous_metrics: tuple[EpochMetrics, ...] = (),
    training_distribution: Mapping[str, int] | None = None,
    resume_state: CheckpointState | None = None,
    resumed_batch_source: Callable[[int], Iterable[TrainingBatch]] | None = None,
) -> tuple[EpochMetrics, ...]:
    """Train for ``epochs`` with periodic and end-of-epoch checkpoints.

    ``training_distribution`` is required when ``checkpoint_path`` is set. A
    mid-epoch ``resume_state`` completes that epoch before later epochs begin.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if start_epoch < 0:
        raise ValueError("start epoch must be non-negative")
    if checkpoint_every_steps is not None and checkpoint_every_steps <= 0:
        raise ValueError("checkpoint interval must be positive")
    if checkpoint_path is not None and training_distribution is None:
        raise ValueError("training distribution is required when checkpointing")
    normalized_distribution = _normalize_training_distribution(training_distribution)
    if resume_state is not None and resume_state.batch_offset > 0:
        if previous_metrics:
            raise ValueError("previous metrics must come from the resume checkpoint")
        history = list(resume_state.metrics[:-1])
        first_epoch = resume_state.epoch
    else:
        if resume_state is not None:
            start_epoch = resume_state.epoch
            previous_metrics = resume_state.metrics
        history = list(previous_metrics)
        first_epoch = start_epoch + 1

    for offset in range(epochs):
        epoch = first_epoch + offset
        current_resume_state = (
            resume_state
            if offset == 0
            and resume_state is not None
            and resume_state.batch_offset > 0
            else None
        )
        source_batch_offset = 0
        if current_resume_state is not None and resumed_batch_source is not None:
            source_batch_offset = current_resume_state.batch_offset
            batches = resumed_batch_source(source_batch_offset)
        else:
            batches = batch_source()
        metrics = train_epoch(
            model,
            batches,
            optimizer,
            epoch=epoch,
            device=device,
            grad_clip_norm=grad_clip_norm,
            checkpoint_path=checkpoint_path,
            checkpoint_every_steps=checkpoint_every_steps,
            previous_metrics=tuple(history),
            training_distribution=normalized_distribution,
            resume_state=current_resume_state,
            source_batch_offset=source_batch_offset,
        )
        history.append(metrics)
        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=metrics.epoch,
                step=0,
                batch_offset=0,
                metrics=tuple(history),
                training_distribution=normalized_distribution,
            )
    return tuple(history)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    step: int = 0,
    batch_offset: int = 0,
    metrics: tuple[EpochMetrics, ...] = (),
    training_distribution: Mapping[str, int] | None = None,
) -> None:
    """Save model, optimizer, and training progress to one checkpoint file."""
    if epoch < 0:
        raise ValueError("checkpoint epoch must be non-negative")
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    if batch_offset < 0:
        raise ValueError("checkpoint batch offset must be non-negative")
    normalized_distribution = _normalize_training_distribution(training_distribution)
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "batch_offset": batch_offset,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": [asdict(item) for item in metrics],
        "training_distribution": dict(normalized_distribution),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=checkpoint_path.parent,
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: torch.device | str = "cpu",
) -> CheckpointState:
    """Restore a checkpoint and return its training metadata."""
    checkpoint = torch.load(Path(path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a mapping")
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = int(checkpoint["epoch"])
        step = int(checkpoint.get("step", 0))
        batch_offset = int(checkpoint.get("batch_offset", 0))
        metrics = tuple(EpochMetrics(**item) for item in checkpoint["metrics"])
        training_distribution = _normalize_training_distribution(
            checkpoint.get("training_distribution", {})
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("checkpoint has an invalid structure") from error
    if epoch < 0:
        raise ValueError("checkpoint epoch must be non-negative")
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    if batch_offset < 0:
        raise ValueError("checkpoint batch offset must be non-negative")
    return CheckpointState(
        epoch=epoch,
        metrics=metrics,
        step=step,
        batch_offset=batch_offset,
        training_distribution=training_distribution,
    )


def _normalize_training_distribution(
    distribution: Mapping[str, int] | tuple[tuple[str, int], ...] | None,
) -> tuple[tuple[str, int], ...]:
    """Validate and sort per-kind training exposure for checkpoint metadata."""
    if distribution is None:
        return ()
    try:
        items = distribution.items()
    except AttributeError:
        items = distribution
    normalized: list[tuple[str, int]] = []
    for kind, count in items:
        if not isinstance(kind, str) or not kind:
            raise ValueError("training distribution kinds must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                "training distribution counts must be non-negative integers"
            )
        normalized.append((kind, count))
    return tuple(sorted(normalized))


def _mean_loss(total_loss: float | torch.Tensor, example_count: int) -> float:
    """Materialize the reporting accumulator only at a reporting boundary."""
    if isinstance(total_loss, torch.Tensor):
        return float(total_loss.cpu()) / example_count
    return total_loss / example_count
