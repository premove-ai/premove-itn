import json
from pathlib import Path

import pytest

import scripts.train_conversational_adaptation as adaptation
from premove_itn.training import TrainingProgress


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_resolve_schedule_preserves_reference_order_and_train_partition(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    schedule = tmp_path / "training_schedule.jsonl"
    manifest = tmp_path / "schedule_manifest.json"
    _write_jsonl(
        first,
        [
            {
                "text": "keep",
                "expected_text": "keep",
                "partition": "train",
                "kind": "KEEP",
            },
            {
                "text": "held out",
                "expected_text": "1",
                "partition": "validation",
                "kind": "CARDINAL",
            },
        ],
    )
    _write_jsonl(
        second,
        [
            {
                "text": "two",
                "expected_text": "2",
                "partition": "train",
                "kind": "CARDINAL",
            }
        ],
    )
    entries = [
        {
            "source_dataset": "second.jsonl",
            "record_id": 0,
            "partition": "train",
            "bucket": "conversational_positive",
            "kind": "CARDINAL",
        },
        {
            "source_dataset": "first.jsonl",
            "record_id": 0,
            "partition": "train",
            "bucket": "candidate_bearing_keep",
            "kind": "KEEP",
        },
    ]
    _write_jsonl(schedule, entries)
    manifest.write_text(
        json.dumps(
            {
                "records": 2,
                "bucket_distribution": {
                    "candidate_bearing_keep": 1,
                    "conversational_positive": 1,
                },
                "inputs": {
                    "first.jsonl": adaptation.sha256(first),
                    "second.jsonl": adaptation.sha256(second),
                },
                "artifacts": {
                    "training_schedule.jsonl": {
                        "bytes": schedule.stat().st_size,
                        "sha256": adaptation.sha256(schedule),
                    }
                },
            }
        )
    )
    monkeypatch.setattr(adaptation, "ROOT", tmp_path)
    monkeypatch.setattr(adaptation, "SCHEDULE", schedule)
    monkeypatch.setattr(adaptation, "SCHEDULE_MANIFEST", manifest)

    rows, _ = adaptation.resolve_schedule()

    assert [(row["text"], row["partition"]) for row in rows] == [
        ("two", "train"),
        ("keep", "train"),
    ]


def test_resolve_schedule_rejects_duplicate_references(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    schedule = tmp_path / "training_schedule.jsonl"
    manifest = tmp_path / "schedule_manifest.json"
    _write_jsonl(
        dataset,
        [
            {
                "text": "one",
                "expected_text": "1",
                "partition": "train",
                "kind": "CARDINAL",
            }
        ],
    )
    entry = {
        "source_dataset": "dataset.jsonl",
        "record_id": 0,
        "partition": "train",
        "bucket": "conversational_positive",
        "kind": "CARDINAL",
    }
    _write_jsonl(schedule, [entry, entry])
    manifest.write_text(
        json.dumps(
            {
                "records": 2,
                "bucket_distribution": {"conversational_positive": 2},
                "inputs": {"dataset.jsonl": adaptation.sha256(dataset)},
                "artifacts": {
                    "training_schedule.jsonl": {
                        "bytes": schedule.stat().st_size,
                        "sha256": adaptation.sha256(schedule),
                    }
                },
            }
        )
    )
    monkeypatch.setattr(adaptation, "ROOT", tmp_path)
    monkeypatch.setattr(adaptation, "SCHEDULE", schedule)
    monkeypatch.setattr(adaptation, "SCHEDULE_MANIFEST", manifest)

    with pytest.raises(RuntimeError, match="duplicate record references"):
        adaptation.resolve_schedule()


def test_progress_reporter_preserves_a_durable_milestone(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    progress_path = tmp_path / "progress.json"
    milestones = tmp_path / "milestones"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(adaptation, "CHECKPOINT", checkpoint)
    monkeypatch.setattr(adaptation, "PROGRESS", progress_path)
    monkeypatch.setattr(adaptation, "MILESTONES", milestones)
    monkeypatch.setattr(adaptation, "MILESTONE_EXPOSURES", (10_000,))
    reporter = adaptation.ProgressReporter(
        total_batches=2_000,
        total_examples=16_000,
        initial_batch_offset=0,
        initial_examples=0,
        started=100.0,
    )
    monkeypatch.setattr(adaptation.time, "monotonic", lambda: 110.0)

    reporter(
        TrainingProgress(
            epoch=1,
            batch_offset=1_250,
            optimizer_steps=1_250,
            examples=10_000,
        )
    )

    milestone = milestones / "checkpoint_010000.pt"
    assert milestone.read_bytes() == b"checkpoint"
    assert checkpoint.stat().st_ino == milestone.stat().st_ino
    assert json.loads(progress_path.read_text())["completed_examples"] == 10_000


def test_progress_reporter_does_not_publish_final_milestone_before_epoch_save(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    progress_path = tmp_path / "progress.json"
    milestones = tmp_path / "milestones"
    checkpoint.write_bytes(b"44,000-example recovery state")
    monkeypatch.setattr(adaptation, "CHECKPOINT", checkpoint)
    monkeypatch.setattr(adaptation, "PROGRESS", progress_path)
    monkeypatch.setattr(adaptation, "MILESTONES", milestones)
    monkeypatch.setattr(adaptation, "MILESTONE_EXPOSURES", (44_058,))
    reporter = adaptation.ProgressReporter(
        total_batches=5_508,
        total_examples=44_058,
        initial_batch_offset=0,
        initial_examples=0,
        started=100.0,
    )
    monkeypatch.setattr(adaptation.time, "monotonic", lambda: 110.0)

    reporter(
        TrainingProgress(
            epoch=1,
            batch_offset=5_508,
            optimizer_steps=5_508,
            examples=44_058,
        )
    )

    assert not (milestones / "checkpoint_044058.pt").exists()


def test_preserve_can_atomically_replace_a_stale_milestone(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    milestone = tmp_path / "milestones/checkpoint_044058.pt"
    checkpoint.write_bytes(b"completed epoch")
    milestone.parent.mkdir()
    milestone.write_bytes(b"44,000-example recovery state")

    adaptation._preserve(checkpoint, milestone, replace=True)

    assert milestone.read_bytes() == b"completed epoch"
    assert checkpoint.stat().st_ino == milestone.stat().st_ino
