from pathlib import Path

import pytest
import torch

from scripts.evaluate_conversational_adaptation_checkpoints import (
    validate_checkpoint_set,
)


def _checkpoint(path: Path, *, examples: int, weight: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metrics": [
                {
                    "epoch": 1,
                    "mean_loss": 0.1,
                    "steps": examples // 8,
                    "examples": examples,
                }
            ],
            "model_state_dict": {"weight": torch.tensor([weight])},
            "optimizer_state_dict": {},
            "run_fingerprint": "adaptation-run",
        },
        path,
    )


def test_validation_rejects_checkpoint_with_wrong_exposure(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_044058.pt"
    _checkpoint(checkpoint, examples=44_000, weight=1.0)

    with pytest.raises(RuntimeError, match="contains 44000 examples; expected 44058"):
        validate_checkpoint_set({44_058: checkpoint})


def test_validation_rejects_duplicate_model_payloads(tmp_path: Path) -> None:
    first = tmp_path / "checkpoint_010000.pt"
    second = tmp_path / "checkpoint_020000.pt"
    _checkpoint(first, examples=10_000, weight=1.0)
    _checkpoint(second, examples=20_000, weight=1.0)

    with pytest.raises(RuntimeError, match="identical model tensors"):
        validate_checkpoint_set({10_000: first, 20_000: second})
