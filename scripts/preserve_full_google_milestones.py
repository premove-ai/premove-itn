"""Preserve full-Google checkpoints near 50k exposure milestones."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/generated/full_google_train"
MANIFEST = OUTPUT / "manifest.json"
PROGRESS = OUTPUT / "progress.json"
MILESTONES = OUTPUT / "milestones"
POLL_SECONDS = 30
MILESTONE_EXAMPLES = 50_000


@dataclass(frozen=True, slots=True)
class Milestone:
    exposure: int
    batch_offset: int | None


def milestone_schedule(manifest: dict[str, object]) -> tuple[Milestone, ...]:
    """Return nearest durable checkpoints for each 50k total exposure."""
    initial = int(manifest["already_trained_examples"])
    remaining = int(manifest["remaining_examples"])
    batch_size = int(manifest["batch_size"])
    checkpoint_batches = int(manifest["checkpoint_every_batches"])
    checkpoint_examples = batch_size * checkpoint_batches
    total = initial + remaining
    milestones: dict[int, Milestone] = {}
    for desired in range(MILESTONE_EXAMPLES, total, MILESTONE_EXAMPLES):
        distance = desired - initial
        lower, remainder = divmod(distance, checkpoint_examples)
        generation = lower + (remainder * 2 > checkpoint_examples)
        exposure = initial + generation * checkpoint_examples
        milestones[exposure] = Milestone(
            exposure=exposure,
            batch_offset=generation * checkpoint_batches,
        )
    milestones[total] = Milestone(exposure=total, batch_offset=None)
    return tuple(milestones[exposure] for exposure in sorted(milestones))


def _milestone_path(directory: Path, exposure: int) -> Path:
    return directory / f"checkpoint_{exposure:06d}.pt"


def _preserve(source: Path, destination: Path) -> None:
    """Add a durable name for an immutable, atomically published checkpoint."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        return
    directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def preserve_available(
    *,
    manifest: dict[str, object],
    progress: dict[str, object],
    progress_path: Path,
    checkpoint_path: Path,
    milestone_directory: Path,
) -> list[dict[str, object]]:
    """Preserve each milestone still represented by the rotating generations."""
    checkpoint_batches = int(manifest["checkpoint_every_batches"])
    durable_batch = int(progress.get("durable_batch_offset", 0))
    progress_modified = progress_path.stat().st_mtime_ns
    previous_path = checkpoint_path.with_name(f"{checkpoint_path.name}.previous")
    events = []
    for milestone in milestone_schedule(manifest):
        destination = _milestone_path(milestone_directory, milestone.exposure)
        if destination.exists():
            continue
        if milestone.batch_offset is None:
            if progress.get("status") != "completed":
                continue
            source = checkpoint_path
        elif durable_batch == milestone.batch_offset:
            source = checkpoint_path
        elif durable_batch == milestone.batch_offset + checkpoint_batches:
            source = previous_path
        elif durable_batch > milestone.batch_offset + checkpoint_batches:
            events.append(
                {
                    "status": "missed",
                    "exposure": milestone.exposure,
                    "batch_offset": milestone.batch_offset,
                }
            )
            continue
        else:
            continue
        if not source.is_file() or source.stat().st_mtime_ns > progress_modified:
            continue
        _preserve(source, destination)
        events.append(
            {
                "status": "preserved",
                "exposure": milestone.exposure,
                "batch_offset": milestone.batch_offset,
                "path": str(destination),
            }
        )
    return events


def watch(
    *,
    manifest_path: Path = MANIFEST,
    progress_path: Path = PROGRESS,
    milestone_directory: Path = MILESTONES,
    poll_seconds: float = POLL_SECONDS,
) -> None:
    """Watch one running experiment until its final checkpoint is preserved."""
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    manifest = json.loads(manifest_path.read_text())
    checkpoint_path = ROOT / str(manifest["checkpoint"])
    milestone_directory.mkdir(parents=True, exist_ok=True)
    lock_handle = (milestone_directory / ".watcher.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("a milestone watcher is already running") from error
    lock_handle.write(f"{os.getpid()}\n")
    lock_handle.flush()
    final_exposure = milestone_schedule(manifest)[-1].exposure
    while True:
        try:
            progress = json.loads(progress_path.read_text())
            events = preserve_available(
                manifest=manifest,
                progress=progress,
                progress_path=progress_path,
                checkpoint_path=checkpoint_path,
                milestone_directory=milestone_directory,
            )
            for event in events:
                print(json.dumps(event, sort_keys=True), flush=True)
            if _milestone_path(milestone_directory, final_exposure).exists():
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    args = parser.parse_args()
    watch(poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
