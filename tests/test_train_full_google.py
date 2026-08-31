import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch
from torch import nn

import scripts.train_full_google as full_training
from premove_itn.training import (
    TrainingConfig,
    TrainingProgress,
    create_optimizer,
    save_checkpoint,
)


class ScalarScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch) -> torch.Tensor:
        return self.score.expand(batch.candidate_offsets[-1])


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_scan_excludes_seen_validation_and_long_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_rows(
        dataset,
        [
            {"partition": "train", "text": "seen", "kind": "KEEP"},
            {"partition": "validation", "text": "held out", "kind": "DATE"},
            {"partition": "train", "text": "new", "kind": "DATE"},
            {"partition": "train", "text": "too long", "kind": "TIME"},
        ],
    )
    monkeypatch.setattr(full_training, "DATASET", dataset)
    monkeypatch.setattr(full_training, "FIRST_UNSEEN_RECORD_ID", 2)
    monkeypatch.setattr(full_training, "MAX_SOURCE_CHARS", 7)

    count, remaining, cumulative, first_id, last_id = full_training.scan_dataset()

    assert count == 1
    assert remaining == {"DATE": 1}
    assert cumulative == {"DATE": 1, "KEEP": 1}
    assert (first_id, last_id) == (2, 2)
    assert list(full_training.eligible_rows()) == [
        (2, {"partition": "train", "text": "new", "kind": "DATE"})
    ]


def test_window_batches_are_stable_and_length_bucketed(monkeypatch) -> None:
    monkeypatch.setattr(full_training, "BATCH_SIZE", 2)
    rows = [
        (3, {"text": "bbb"}),
        (2, {"text": "a"}),
        (1, {"text": "c"}),
    ]

    batches = list(full_training.window_batches(rows))

    assert [[record_id for record_id, _ in batch] for batch in batches] == [
        [1, 2],
        [3],
    ]


def test_batch_source_prefetches_remaining_batches_in_order(monkeypatch) -> None:
    rows = (
        (0, {"text": "bbb", "expected_text": "3"}),
        (1, {"text": "a", "expected_text": "1"}),
        (2, {"text": "cc", "expected_text": "2"}),
    )
    monkeypatch.setattr(full_training, "eligible_rows", lambda: iter(rows))
    monkeypatch.setattr(full_training, "BATCH_SIZE", 1)
    monkeypatch.setattr(full_training, "BUCKET_WINDOW", 3)
    observed: dict[str, int] = {}

    def ordered(function, items, *, workers, max_pending, initializer):
        observed.update(workers=workers, max_pending=max_pending)
        return iter(items)

    monkeypatch.setattr(full_training, "ordered_process_prefetch", ordered)

    batches = list(full_training.batch_source(start_batch_offset=1))

    assert batches == [(('cc', '2'),), (('bbb', '3'),)]
    assert observed == {
        "workers": full_training.PREFETCH_WORKERS,
        "max_pending": full_training.PREFETCH_BATCHES,
    }


def test_progress_reporter_persists_measured_eta_atomically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    progress_path = tmp_path / "progress.json"
    clock = iter((100.0, 110.0))
    monkeypatch.setattr(full_training.time, "monotonic", lambda: next(clock))
    reporter = full_training.ProgressReporter(
        path=progress_path,
        total_batches=100,
        total_examples=800,
        initial_batch_offset=20,
        initial_examples=160,
        checkpoint_every_batches=10,
        report_every_batches=5,
    )

    reporter(
        TrainingProgress(
            epoch=2,
            batch_offset=25,
            optimizer_steps=24,
            examples=200,
        )
    )

    assert json.loads(progress_path.read_text()) == {
        "status": "running",
        "epoch": 2,
        "completed_batches": 25,
        "total_batches": 100,
        "durable_batch_offset": 20,
        "optimizer_steps": 24,
        "completed_examples": 200,
        "total_examples": 800,
        "elapsed_active_seconds": 10.0,
        "batches_per_second": 0.5,
        "examples_per_second": 4.0,
        "eta_seconds": 150.0,
    }
    assert list(tmp_path.iterdir()) == [progress_path]


def test_resume_uses_previous_generation_when_latest_is_malformed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    source = tmp_path / "source.pt"
    model = ScalarScorer()
    optimizer = create_optimizer(model, TrainingConfig(weight_decay=0))
    save_checkpoint(source, model, optimizer, epoch=1)
    save_checkpoint(checkpoint, model, optimizer, epoch=2)
    save_checkpoint(checkpoint, model, optimizer, epoch=3)
    checkpoint.write_bytes(b"malformed")
    monkeypatch.setattr(full_training, "CHECKPOINT", checkpoint)
    monkeypatch.setattr(full_training, "SOURCE_CHECKPOINT", source)

    restored_model = ScalarScorer()
    restored_optimizer = create_optimizer(
        restored_model,
        TrainingConfig(weight_decay=0),
    )
    resume_path, state = full_training.load_resume_checkpoint(
        restored_model,
        restored_optimizer,
    )

    assert resume_path.name == "checkpoint.pt.previous"
    assert state.epoch == 2
