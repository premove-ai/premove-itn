"""Fail if a release wheel or source archive has unsafe contents."""

from __future__ import annotations

import argparse
import re
import tarfile
from configparser import ConfigParser
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


def _forbidden_paths(names: list[str]) -> list[str]:
    forbidden = []
    for name in names:
        path = PurePosixPath(name)
        lowered_parts = {part.lower() for part in path.parts}
        if path.name.lower() in FORBIDDEN_NAMES or lowered_parts & FORBIDDEN_PARTS:
            forbidden.append(name)
    return forbidden


def _reject_forbidden_paths(names: list[str], artifact: Path) -> None:
    forbidden = _forbidden_paths(names)
    if forbidden:
        raise RuntimeError(f"{artifact.name} contains forbidden artifacts: {forbidden}")


def inspect_wheel(wheel: Path) -> None:
    """Verify runtime requirements, CLI metadata, and wheel contents."""
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

        entry_points_name = next(
            (name for name in names if name.endswith(".dist-info/entry_points.txt")),
            None,
        )
        entry_points = ConfigParser()
        if entry_points_name is not None:
            entry_points.read_string(archive.read(entry_points_name).decode())
        if (
            entry_points.get("console_scripts", "premove-itn", fallback=None)
            != "premove_itn.cli:main"
        ):
            raise RuntimeError("wheel is missing the premove-itn CLI entry point")

        _reject_forbidden_paths(names, wheel)


def inspect_sdist(sdist: Path) -> None:
    """Verify that a source archive contains no release-only data artifacts."""
    with tarfile.open(sdist, mode="r:gz") as archive:
        names = archive.getnames()
    if not any(name.endswith("/PKG-INFO") for name in names):
        raise RuntimeError("source archive has no package metadata")
    _reject_forbidden_paths(names, sdist)


def inspect_artifact(artifact: Path) -> None:
    """Dispatch inspection based on the release artifact suffix."""
    if artifact.name.endswith(".whl"):
        inspect_wheel(artifact)
    elif artifact.name.endswith(".tar.gz"):
        inspect_sdist(artifact)
    else:
        raise ValueError(f"unsupported release artifact: {artifact}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path, nargs="+")
    arguments = parser.parse_args()
    for artifact in arguments.artifacts:
        inspect_artifact(artifact.resolve())
        print(f"verified release artifact: {artifact}")


if __name__ == "__main__":
    main()
