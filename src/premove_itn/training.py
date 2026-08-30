"""Minimal optimizer loop for the contextual candidate experiment."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from premove_itn.training_batch import TrainingBatch, structured_batch_loss


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimizer and gradient settings for the first experiment."""

    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    grad_clip_norm: float | None = 1.0


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


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create the AdamW optimizer used by the first training experiment."""
    if config.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight decay must be non-negative")
    if config.grad_clip_norm is not None and config.grad_clip_norm <= 0:
        raise ValueError("gradient clip norm must be positive")
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
) -> EpochMetrics:
    """Run one optimizer epoch over prepared training batches."""
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("gradient clip norm must be positive")

    model.train()
    total_loss = 0.0
    step_count = 0
    example_count = 0
    for batch in batches:
        model_batch = batch if device is None else batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = structured_batch_loss(model, model_batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss is not finite")
        if not loss.requires_grad:
            if batch.candidate_batch.candidate_offsets[-1] != 0:
                raise RuntimeError("training loss is detached from candidate scores")
            total_loss += float(loss.detach())
            step_count += 1
            example_count += len(batch.examples)
            continue
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += float(loss.detach())
        step_count += 1
        example_count += len(batch.examples)

    if step_count == 0:
        raise ValueError("training epoch must contain at least one batch")
    return EpochMetrics(
        epoch=epoch,
        mean_loss=total_loss / step_count,
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
    start_epoch: int = 0,
    previous_metrics: tuple[EpochMetrics, ...] = (),
) -> tuple[EpochMetrics, ...]:
    """Train for ``epochs`` and optionally save a checkpoint after each epoch."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if start_epoch < 0:
        raise ValueError("start epoch must be non-negative")
    history = list(previous_metrics)
    for offset in range(1, epochs + 1):
        metrics = train_epoch(
            model,
            batch_source(),
            optimizer,
            epoch=start_epoch + offset,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )
        history.append(metrics)
        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=metrics.epoch,
                metrics=tuple(history),
            )
    return tuple(history)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    metrics: tuple[EpochMetrics, ...] = (),
) -> None:
    """Save model, optimizer, and epoch metrics to one checkpoint file."""
    if epoch < 0:
        raise ValueError("checkpoint epoch must be non-negative")
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": [asdict(item) for item in metrics],
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
        metrics = tuple(EpochMetrics(**item) for item in checkpoint["metrics"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("checkpoint has an invalid structure") from error
    if epoch < 0:
        raise ValueError("checkpoint epoch must be non-negative")
    return CheckpointState(epoch=epoch, metrics=metrics)
