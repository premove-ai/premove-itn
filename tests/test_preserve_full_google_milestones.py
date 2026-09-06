import json
import os
from pathlib import Path

from scripts.preserve_full_google_milestones import (
    milestone_schedule,
    preserve_available,
)


def _manifest() -> dict[str, object]:
    return {
        "already_trained_examples": 10_000,
        "remaining_examples": 701_135,
        "batch_size": 8,
        "checkpoint_every_batches": 1_000,
        "checkpoint": "checkpoint.pt",
    }


def test_milestone_schedule_uses_nearest_durable_50k_exposures() -> None:
    assert [item.exposure for item in milestone_schedule(_manifest())] == [
        50_000,
        98_000,
        146_000,
        202_000,
        250_000,
        298_000,
        346_000,
        402_000,
        450_000,
        498_000,
        546_000,
        602_000,
        650_000,
        698_000,
        711_135,
    ]


def test_preserve_available_links_current_completed_generation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"50k")
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(json.dumps({"durable_batch_offset": 5_000}))
    os.utime(progress_path, ns=(checkpoint.stat().st_mtime_ns + 1,) * 2)
    milestones = tmp_path / "milestones"

    events = preserve_available(
        manifest=_manifest(),
        progress={"durable_batch_offset": 5_000},
        progress_path=progress_path,
        checkpoint_path=checkpoint,
        milestone_directory=milestones,
    )

    preserved = milestones / "checkpoint_050000.pt"
    assert preserved.read_bytes() == b"50k"
    assert preserved.stat().st_ino == checkpoint.stat().st_ino
    assert events == [
        {
            "status": "preserved",
            "exposure": 50_000,
            "batch_offset": 5_000,
            "path": str(preserved),
        }
    ]


def test_preserve_available_uses_previous_generation_after_rotation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    previous = tmp_path / "checkpoint.pt.previous"
    checkpoint.write_bytes(b"58k")
    previous.write_bytes(b"50k")
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(json.dumps({"durable_batch_offset": 6_000}))
    newest_checkpoint = max(checkpoint.stat().st_mtime_ns, previous.stat().st_mtime_ns)
    os.utime(progress_path, ns=(newest_checkpoint + 1,) * 2)

    preserve_available(
        manifest=_manifest(),
        progress={"durable_batch_offset": 6_000},
        progress_path=progress_path,
        checkpoint_path=checkpoint,
        milestone_directory=tmp_path / "milestones",
    )

    assert (tmp_path / "milestones/checkpoint_050000.pt").read_bytes() == b"50k"


def test_preserve_available_waits_during_checkpoint_progress_race(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(json.dumps({"durable_batch_offset": 5_000}))
    checkpoint.write_bytes(b"newer checkpoint")

    events = preserve_available(
        manifest=_manifest(),
        progress={"durable_batch_offset": 5_000},
        progress_path=progress_path,
        checkpoint_path=checkpoint,
        milestone_directory=tmp_path / "milestones",
    )

    assert events == []
    assert not (tmp_path / "milestones/checkpoint_050000.pt").exists()
