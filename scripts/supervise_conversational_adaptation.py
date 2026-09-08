"""Keep the conversational-adaptation trainer awake and restartable."""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.supervise_full_google import run_supervised

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/train_conversational_adaptation.py"
OUTPUT = ROOT / "data/generated/conversational_adaptation_44k_v2"
PROGRESS = OUTPUT / "progress.json"
LOG = OUTPUT / "supervisor.log"


def main() -> None:
    command = ("/usr/bin/caffeinate", "-dimsu", sys.executable, str(TRAINER))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("ab", buffering=0) as output:
        run_supervised(
            command,
            progress_path=PROGRESS,
            stale_seconds=15 * 60,
            poll_seconds=30,
            restart_delay_seconds=30,
            output=output,
        )


if __name__ == "__main__":
    main()
