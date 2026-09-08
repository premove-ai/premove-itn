"""Restart full-Google training when the trainer exits or stops progressing."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import IO

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/train_full_google.py"
PROGRESS = ROOT / "data/generated/full_google_train/progress.json"
STALE_SECONDS = 15 * 60
POLL_SECONDS = 30
RESTART_DELAY_SECONDS = 30
TERMINATE_TIMEOUT_SECONDS = 30


def _progress_marker(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_supervised(
    command: Sequence[str],
    *,
    progress_path: Path,
    stale_seconds: float,
    poll_seconds: float,
    restart_delay_seconds: float,
    output: int | IO[bytes] | None = None,
) -> int:
    """Run until success, restarting an exited or progress-stale process group."""
    if stale_seconds <= 0 or poll_seconds <= 0 or restart_delay_seconds < 0:
        raise ValueError("supervisor timing values must be positive")
    restarts = 0
    while True:
        process = subprocess.Popen(
            tuple(command),
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
        marker = _progress_marker(progress_path)
        last_progress_at = time.time()
        try:
            while process.poll() is None:
                time.sleep(poll_seconds)
                current_marker = _progress_marker(progress_path)
                if current_marker != marker:
                    marker = current_marker
                    last_progress_at = time.time()
                if time.time() - last_progress_at >= stale_seconds:
                    print(
                        json.dumps(
                            {
                                "status": "stale_progress_restart",
                                "pid": process.pid,
                                "stale_seconds": stale_seconds,
                            }
                        ),
                        flush=True,
                    )
                    _stop_process_group(process)
                    break
        except KeyboardInterrupt:
            _stop_process_group(process)
            raise

        if process.returncode == 0:
            return restarts
        restarts += 1
        print(
            json.dumps(
                {
                    "status": "trainer_restart",
                    "exit_code": process.returncode,
                    "restart": restarts,
                }
            ),
            flush=True,
        )
        time.sleep(restart_delay_seconds)


def main() -> None:
    command = (
        "/usr/bin/caffeinate",
        "-dimsu",
        sys.executable,
        str(TRAINER),
    )
    run_supervised(
        command,
        progress_path=PROGRESS,
        stale_seconds=STALE_SECONDS,
        poll_seconds=POLL_SECONDS,
        restart_delay_seconds=RESTART_DELAY_SECONDS,
    )


if __name__ == "__main__":
    main()
