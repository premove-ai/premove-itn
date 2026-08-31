import subprocess
import sys
from pathlib import Path

from scripts.supervise_full_google import run_supervised


def test_supervisor_restarts_process_when_progress_becomes_stale(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "started-once"
    progress = tmp_path / "progress.json"
    child = """
import pathlib
import sys
import time

marker = pathlib.Path(sys.argv[1])
progress = pathlib.Path(sys.argv[2])
if marker.exists():
    raise SystemExit(0)
marker.write_text("started")
progress.write_text("running")
time.sleep(30)
"""

    restarts = run_supervised(
        (sys.executable, "-c", child, str(marker), str(progress)),
        progress_path=progress,
        stale_seconds=0.1,
        poll_seconds=0.02,
        restart_delay_seconds=0,
        output=subprocess.DEVNULL,
    )

    assert restarts == 1
