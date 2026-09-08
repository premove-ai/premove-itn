"""Qualify fused MPS AdamW against the legacy optimizer execution path."""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from itertools import islice

import torch

from premove_itn.candidate_scorer import load_candidate_scorer
from premove_itn.training import (
    EpochMetrics,
    TrainingConfig,
    create_optimizer,
    load_checkpoint,
    save_checkpoint,
    train_epoch,
)
from premove_itn.training_batch import TrainingBatch
from scripts.train_full_google import OUTPUT, SOURCE_CHECKPOINT, batch_source

QUALIFICATION_OUTPUT = OUTPUT.parent / "fused_adamw_qualification"
METRICS = QUALIFICATION_OUTPUT / "metrics.json"
CHECKPOINT = QUALIFICATION_OUTPUT / "checkpoint.pt"
SEED = 17
WARMUP_BATCHES = 10
MEASURED_BATCHES = 100
REPORT_EVERY_BATCHES = 10
MINIMUM_SPEEDUP = 1.10
MAXIMUM_RELATIVE_LOSS_DIFFERENCE = 0.05
MAXIMUM_PROBE_DIFFERENCE = 0.01


@dataclass(frozen=True, slots=True)
class VariantResult:
    """One controlled optimizer continuation result."""

    fused: bool
    measured_seconds: float
    batches_per_second: float
    examples_per_second: float
    optimizer_steps: int
    losses: tuple[float, ...]
    probes: dict[str, tuple[float, ...]]


def _optimizer(model: torch.nn.Module, *, fused: bool) -> torch.optim.AdamW:
    config = TrainingConfig()
    if fused:
        return create_optimizer(model, config)
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
        fused=False,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _step_value(optimizer: torch.optim.Optimizer) -> int:
    for parameter, state in optimizer.state.items():
        step = state.get("step")
        if isinstance(parameter, torch.Tensor) and isinstance(step, torch.Tensor):
            return int(step.detach().cpu())
    raise RuntimeError("loaded optimizer does not contain an Adam step tensor")


def _parameter_probes(model: torch.nn.Module) -> dict[str, tuple[float, ...]]:
    named_parameters = tuple(model.named_parameters())
    selected_indices = {0, len(named_parameters) // 2, len(named_parameters) - 1}
    selected_indices.update(
        index
        for index, (name, _) in enumerate(named_parameters)
        if name.startswith("scoring_head")
    )
    return {
        name: tuple(parameter.detach().flatten()[:32].cpu().tolist())
        for index, (name, parameter) in enumerate(named_parameters)
        if index in selected_indices
    }


def _run_variant(
    batches: tuple[TrainingBatch, ...],
    *,
    device: torch.device,
    fused: bool,
) -> tuple[VariantResult, torch.nn.Module, torch.optim.AdamW]:
    torch.manual_seed(SEED)
    model = load_candidate_scorer().to(device)
    optimizer = _optimizer(model, fused=fused)
    load_checkpoint(SOURCE_CHECKPOINT, model, optimizer, map_location="cpu")
    assert all(group["fused"] is fused for group in optimizer.param_groups)
    assert all(group["foreach"] is False for group in optimizer.param_groups)
    if fused:
        assert all(
            state["step"].device == parameter.device
            for parameter, state in optimizer.state.items()
            if "step" in state
        )

    warmup = batches[:WARMUP_BATCHES]
    measured = batches[WARMUP_BATCHES:]
    train_epoch(
        model,
        warmup,
        optimizer,
        epoch=2,
        device=device,
    )
    _synchronize(device)
    starting_step = _step_value(optimizer)
    started = time.monotonic()
    losses: list[float] = []
    completed_examples = 0
    for start in range(0, len(measured), REPORT_EVERY_BATCHES):
        chunk = measured[start : start + REPORT_EVERY_BATCHES]
        metrics = train_epoch(
            model,
            chunk,
            optimizer,
            epoch=2,
            device=device,
        )
        losses.append(metrics.mean_loss)
        completed_examples += metrics.examples
        _synchronize(device)
        print(
            json.dumps(
                {
                    "status": "qualification_progress",
                    "fused": fused,
                    "completed_batches": start + len(chunk),
                    "total_batches": len(measured),
                    "mean_loss": metrics.mean_loss,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elapsed = time.monotonic() - started
    optimizer_steps = _step_value(optimizer) - starting_step
    result = VariantResult(
        fused=fused,
        measured_seconds=elapsed,
        batches_per_second=len(measured) / elapsed,
        examples_per_second=completed_examples / elapsed,
        optimizer_steps=optimizer_steps,
        losses=tuple(losses),
        probes=_parameter_probes(model),
    )
    return result, model, optimizer


def _release_mps_cache() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _max_relative_loss_difference(
    baseline: VariantResult,
    fused: VariantResult,
) -> float:
    return max(
        abs(left - right) / max(abs(left), 1e-12)
        for left, right in zip(baseline.losses, fused.losses, strict=True)
    )


def _max_probe_difference(
    baseline: VariantResult,
    fused: VariantResult,
) -> float:
    if baseline.probes.keys() != fused.probes.keys():
        raise RuntimeError("optimizer variants produced different parameter probes")
    return max(
        abs(left - right)
        for name in baseline.probes
        for left, right in zip(
            baseline.probes[name],
            fused.probes[name],
            strict=True,
        )
    )


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("fused AdamW qualification requires MPS")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    QUALIFICATION_OUTPUT.mkdir(parents=True, exist_ok=True)
    batch_count = WARMUP_BATCHES + MEASURED_BATCHES
    batches = tuple(islice(batch_source(), batch_count))
    if len(batches) != batch_count:
        raise RuntimeError("qualification batch source ended early")
    device = torch.device("mps")

    baseline, baseline_model, baseline_optimizer = _run_variant(
        batches,
        device=device,
        fused=False,
    )
    del baseline_model, baseline_optimizer
    _release_mps_cache()
    fused, fused_model, fused_optimizer = _run_variant(
        batches,
        device=device,
        fused=True,
    )
    speedup = baseline.measured_seconds / fused.measured_seconds
    relative_loss_difference = _max_relative_loss_difference(baseline, fused)
    probe_difference = _max_probe_difference(baseline, fused)
    passed = (
        baseline.optimizer_steps == fused.optimizer_steps
        and speedup >= MINIMUM_SPEEDUP
        and relative_loss_difference <= MAXIMUM_RELATIVE_LOSS_DIFFERENCE
        and probe_difference <= MAXIMUM_PROBE_DIFFERENCE
    )

    examples = sum(len(batch.examples) for batch in batches)
    checkpoint_metrics = (
        EpochMetrics(
            epoch=2,
            mean_loss=sum(fused.losses) / len(fused.losses),
            steps=len(batches),
            examples=examples,
        ),
    )
    save_checkpoint(
        CHECKPOINT,
        fused_model,
        fused_optimizer,
        epoch=2,
        step=sum(batch.candidate_batch.candidate_offsets[-1] > 0 for batch in batches),
        batch_offset=len(batches),
        metrics=checkpoint_metrics,
        training_distribution={"QUALIFICATION": examples},
        run_fingerprint="fused-adamw-qualification",
    )
    del fused_model, fused_optimizer
    _release_mps_cache()

    resumed_model = load_candidate_scorer().to(device)
    resumed_optimizer = create_optimizer(resumed_model, TrainingConfig())
    resumed_state = load_checkpoint(
        CHECKPOINT,
        resumed_model,
        resumed_optimizer,
        map_location="cpu",
    )
    resumed_starting_step = _step_value(resumed_optimizer)
    resume_batch = next(
        batch for batch in batches if batch.candidate_batch.candidate_offsets[-1] > 0
    )
    train_epoch(
        resumed_model,
        (resume_batch,),
        resumed_optimizer,
        epoch=2,
        device=device,
    )
    _synchronize(device)
    resume_passed = (
        resumed_state.batch_offset == len(batches)
        and _step_value(resumed_optimizer) == resumed_starting_step + 1
        and all(group["fused"] is True for group in resumed_optimizer.param_groups)
        and all(
            state["step"].device == parameter.device
            for parameter, state in resumed_optimizer.state.items()
            if "step" in state
        )
    )
    del resumed_model, resumed_optimizer
    _release_mps_cache()
    CHECKPOINT.unlink(missing_ok=True)
    CHECKPOINT.with_name(f"{CHECKPOINT.name}.previous").unlink(missing_ok=True)
    passed = passed and resume_passed

    payload = {
        "status": "passed" if passed else "failed",
        "seed": SEED,
        "warmup_batches": WARMUP_BATCHES,
        "measured_batches": MEASURED_BATCHES,
        "minimum_speedup": MINIMUM_SPEEDUP,
        "maximum_relative_loss_difference": MAXIMUM_RELATIVE_LOSS_DIFFERENCE,
        "maximum_probe_difference": MAXIMUM_PROBE_DIFFERENCE,
        "observed_speedup": speedup,
        "observed_relative_loss_difference": relative_loss_difference,
        "observed_probe_difference": probe_difference,
        "fused_checkpoint_resume_passed": resume_passed,
        "baseline": asdict(baseline),
        "fused": asdict(fused),
    }
    METRICS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
