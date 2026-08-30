"""Download the revision-pinned official Schema-Guided Dialogue source."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPOSITORY = (
    "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git"
)
REVISION = "e852981ae34990f4358979625854259302feaa78"


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
            REPOSITORY,
            str(destination),
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
        default=Path("data/external/schema-guided-dialogue"),
    )
    download(parser.parse_args().destination)


if __name__ == "__main__":
    main()
