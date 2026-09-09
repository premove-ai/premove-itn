"""Compatibility wrapper for the release wheel inspector."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.inspect_release_artifact import inspect_wheel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    arguments = parser.parse_args()
    inspect_wheel(arguments.wheel.resolve())
    print(f"verified release wheel: {arguments.wheel}")


if __name__ == "__main__":
    main()
