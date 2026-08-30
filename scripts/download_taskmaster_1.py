"""Download the revision-pinned official Taskmaster-1 source."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPOSITORY = "https://github.com/google-research-datasets/Taskmaster.git"
REVISION = "d92cb6af3005f1dc09c39e75e7daf4a04905e00b"


def download(destination: Path) -> None:
    if destination.exists():
        current = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if current != REVISION:
            raise ValueError(f"{destination} is at {current}; expected {REVISION}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            REPOSITORY,
            str(destination),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "sparse-checkout", "init", "--cone"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "sparse-checkout",
            "set",
            "TM-1-2019",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", REVISION],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        default=Path("data/external/taskmaster"),
    )
    download(parser.parse_args().destination)


if __name__ == "__main__":
    main()
