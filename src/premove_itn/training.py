"""Minimal optimizer loop for the contextual candidate experiment."""

from __future__ import annotations

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
) -> EpochMetrics:
    """Run one optimizer epoch over prepared training batches.

    The caller must move ``model`` to ``device`` before creating the optimizer.
    This function moves only each batch's model tensors.

    When configured, checkpoints are written after each ``checkpoint_every_steps``
    optimizer updates. The checkpoint step is the number of updates completed in
    this epoch; batches without candidates do not advance it. A
    ``training_distribution`` is required when checkpointing so saved metadata
    cannot silently omit per-kind exposure.
    """
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("gradient clip norm must be positive")
    if checkpoint_every_steps is not None and checkpoint_every_steps <= 0:
        raise ValueError("checkpoint interval must be positive")
    if checkpoint_path is not None and training_distribution is None:
        raise ValueError("training distribution is required when checkpointing")
    normalized_distribution = _normalize_training_distribution(training_distribution)

    model.train()
    total_loss = 0.0
    step_count = 0
    optimizer_step_count = 0
    example_count = 0
    for batch in batches:
        model_batch = batch if device is None else batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        candidate_count = batch.candidate_batch.candidate_offsets[-1]
        if candidate_count == 0:
            step_count += 1
            example_count += len(batch.examples)
            continue
        loss = structured_batch_loss(model, model_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss is not finite")
        if not loss.requires_grad:
            raise RuntimeError("training loss is detached from candidate scores")
        loss.backward()
        max_norm = grad_clip_norm if grad_clip_norm is not None else float("inf")
        nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)
        optimizer.step()
        total_loss += float(loss.detach()) * len(batch.examples)
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
                mean_loss=total_loss / example_count,
                steps=step_count,
                examples=example_count,
            )
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=epoch,
                step=optimizer_step_count,
                metrics=previous_metrics + (partial_metrics,),
                training_distribution=normalized_distribution,
            )

    if step_count == 0:
        raise ValueError("training epoch must contain at least one batch")
    return EpochMetrics(
        epoch=epoch,
        mean_loss=total_loss / example_count,
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
) -> tuple[EpochMetrics, ...]:
    """Train for ``epochs`` with periodic and end-of-epoch checkpoints.

    ``training_distribution`` is required when ``checkpoint_path`` is set.
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
    history = list(previous_metrics)
    for offset in range(1, epochs + 1):
        metrics = train_epoch(
            model,
            batch_source(),
            optimizer,
            epoch=start_epoch + offset,
            device=device,
            grad_clip_norm=grad_clip_norm,
            checkpoint_path=checkpoint_path,
            checkpoint_every_steps=checkpoint_every_steps,
            previous_metrics=tuple(history),
            training_distribution=normalized_distribution,
        )
        history.append(metrics)
        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=metrics.epoch,
                step=0,
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
    metrics: tuple[EpochMetrics, ...] = (),
    training_distribution: Mapping[str, int] | None = None,
) -> None:
    """Save model, optimizer, and training progress to one checkpoint file."""
    if epoch < 0:
        raise ValueError("checkpoint epoch must be non-negative")
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    normalized_distribution = _normalize_training_distribution(training_distribution)
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": [asdict(item) for item in metrics],
            "training_distribution": dict(normalized_distribution),
        },
        checkpoint_path,
    )


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
    return CheckpointState(
        epoch=epoch,
        metrics=metrics,
        step=step,
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
