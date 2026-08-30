"""Download the revision-pinned official SLURP textual annotations."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPOSITORY = "https://github.com/pswietojanski/slurp.git"
REVISION = "8eb16545762be97ace75334109d73824217311f1"


def download(destination: Path) -> None:
    """Clone and detach the official source repository at ``REVISION``."""
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
        ["git", "-C", str(destination), "checkout", "--detach", REVISION],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        default=Path("data/external/slurp-source"),
    )
    download(parser.parse_args().destination)


if __name__ == "__main__":
    main()
