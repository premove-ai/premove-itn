"""Fail if a release wheel has unsafe contents or incomplete metadata."""

from __future__ import annotations

import argparse
import re
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

REQUIRED_DEPENDENCIES = {
    "huggingface-hub",
    "safetensors",
    "torch",
    "transformers",
}
FORBIDDEN_PARTS = {
    "benchmarks",
    "checkpoints",
    "data",
    "eval",
    "optimizer",
    "training-data",
}
FORBIDDEN_NAMES = {"checkpoint.pt", "model.safetensors"}


def _dependency_name(requirement: str) -> str:
    return re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0]


def inspect_wheel(wheel: Path) -> None:
    """Verify runtime requirements and the code-only wheel boundary."""
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(
            (name for name in names if name.endswith(".dist-info/METADATA")),
            None,
        )
        if metadata_name is None:
            raise RuntimeError("wheel has no package metadata")
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        requirements = metadata.get_all("Requires-Dist", [])
        dependencies = {_dependency_name(value) for value in requirements}
        missing = REQUIRED_DEPENDENCIES - dependencies
        if missing:
            raise RuntimeError(
                f"wheel is missing runtime dependencies: {sorted(missing)}"
            )

        forbidden = []
        for name in names:
            path = PurePosixPath(name)
            lowered_parts = {part.lower() for part in path.parts}
            if path.name.lower() in FORBIDDEN_NAMES or lowered_parts & FORBIDDEN_PARTS:
                forbidden.append(name)
        if forbidden:
            raise RuntimeError(f"wheel contains forbidden artifacts: {forbidden}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    arguments = parser.parse_args()
    inspect_wheel(arguments.wheel.resolve())
    print(f"verified release wheel: {arguments.wheel}")


if __name__ == "__main__":
    main()
