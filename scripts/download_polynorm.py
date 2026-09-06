"""Download the revision-pinned Apple PolyNorm-Bench source."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPOSITORY = "https://github.com/apple/ml-speech-polynorm-bench.git"
REVISION = "f3c67e047bea6b7c40bc2466c0fdaad51d8ce67d"


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
            "polynorm_bench/en-US",
            "LICENSE",
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
        default=Path("data/external/apple-polynorm"),
    )
    download(parser.parse_args().destination)


if __name__ == "__main__":
    main()
