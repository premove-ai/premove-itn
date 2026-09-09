"""Verify the public artifact set and write its release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PYTHON_VERSIONS = ("311", "312", "313")


def expected_names(version: str) -> set[str]:
    prefix = f"premove_itn-{version}"
    return {
        *(f"{prefix}-cp{py}-cp{py}-macosx_11_0_arm64.whl" for py in PYTHON_VERSIONS),
        *(
            f"{prefix}-cp{py}-cp{py}-manylinux_2_28_x86_64.whl"
            for py in PYTHON_VERSIONS
        ),
        f"{prefix}.tar.gz",
    }


def build_manifest(
    dist: Path, *, version: str, git_commit: str, model_sha256: str
) -> dict[str, object]:
    files = {path.name: path for path in dist.iterdir() if path.is_file()}
    expected = expected_names(version)
    if set(files) != expected:
        raise RuntimeError(
            f"release artifact set mismatch: missing={sorted(expected - set(files))}, "
            f"unexpected={sorted(set(files) - expected)}"
        )
    artifacts = {
        name: hashlib.sha256(files[name].read_bytes()).hexdigest()
        for name in sorted(files)
    }
    return {
        "schema_version": 1,
        "version": version,
        "git_commit": git_commit,
        "model_repository": "premove-ai/premove-itn",
        "model_revision": "80bda5e2e1fe9542aa628597090242df57c1a157",
        "model_sha256": model_sha256,
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = build_manifest(
        arguments.dist,
        version=arguments.version,
        git_commit=arguments.git_commit,
        model_sha256=arguments.model_sha256,
    )
    arguments.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
