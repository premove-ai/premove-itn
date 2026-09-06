"""Download revision-pinned VoiceCodeBench metadata without benchmark audio."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPOSITORY = "https://huggingface.co/datasets/besimple-ai/voice-code-bench"
REVISION = "3ccea73877a159eb2a8b17304148c325c5fe5061"


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
            "data/metadata.jsonl",
            "README.md",
            "DATASET_CARD.md",
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
        default=Path("data/external/voice-code-bench"),
    )
    download(parser.parse_args().destination)


if __name__ == "__main__":
    main()
