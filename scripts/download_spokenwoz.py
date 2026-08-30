"""Download and verify the pinned official SpokenWOZ text releases."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

SOURCES = {
    "d3aad10f2e5a37e7e1e84375f0db368b5872a044": {
        "repository": "ssz1111/SpokenWOZ-Train-Text",
        "files": {
            "data.json": (
                "017db5b19f6b6e7c1173aadce2ea6ab204c19f6c8f4359d91d0b9fb3d007b58d"
            ),
            "valListFile.json": (
                "f79f5a80afe428a667cfa4a0aed41fe235507c41071fd44647ae85b862089515"
            ),
        },
    },
    "d6c2d9e53b1005e327db582d327b69311092eb65": {
        "repository": "ssz1111/SpokenWOZ-Test-Text-Fixed",
        "files": {
            "data.json": (
                "96b6ff493a8294ffaf03021c024ef4aa2bfe92f980b84c029c81f2aa0f07f728"
            ),
            "testListFile.json": (
                "b1aaf07ebb76bf8e0dbca273ec6ec89739fb70b4e397e9db4f581a595020d0a5"
            ),
        },
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination: Path) -> None:
    for revision, source in SOURCES.items():
        revision_root = destination / revision
        revision_root.mkdir(parents=True, exist_ok=True)
        for filename, expected_hash in source["files"].items():
            path = revision_root / filename
            if path.exists():
                actual_hash = _sha256(path)
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"{path} has SHA-256 {actual_hash}; expected {expected_hash}"
                    )
                continue
            url = (
                f"https://huggingface.co/datasets/{source['repository']}"
                f"/resolve/{revision}/{filename}?download=true"
            )
            temporary = path.with_suffix(path.suffix + ".partial")
            urllib.request.urlretrieve(url, temporary)
            actual_hash = _sha256(temporary)
            if actual_hash != expected_hash:
                temporary.unlink()
                raise ValueError(
                    f"{url} has SHA-256 {actual_hash}; expected {expected_hash}"
                )
            temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        default=Path("data/external/spokenwoz"),
    )
    download(parser.parse_args().destination)


if __name__ == "__main__":
    main()
