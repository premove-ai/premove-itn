"""Fail when repository Markdown contains a missing relative link."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
_EXTERNAL_SCHEMES = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)


def documentation_paths(root: Path) -> list[Path]:
    """Return the maintained Markdown files whose local links are release-facing."""
    paths = [
        root / "README.md",
        root / "CONTRIBUTING.md",
        root / "benchmarks" / "README.md",
    ]
    paths.extend(sorted((root / "docs").rglob("*.md")))
    paths.extend(sorted((root / "eval").rglob("*.md")))
    return [path for path in paths if path.is_file()]


def _relative_target(raw_target: str) -> str | None:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split(maxsplit=1)[0]
    if not target or target.startswith("#") or _EXTERNAL_SCHEMES.match(target):
        return None
    return target.split("#", 1)[0].split("?", 1)[0]


def find_broken_links(root: Path) -> list[str]:
    """Return one diagnostic for every missing local Markdown target."""
    broken: list[str] = []
    for document in documentation_paths(root):
        text = document.read_text(encoding="utf-8")
        for match in _MARKDOWN_LINK.finditer(text):
            target = _relative_target(match.group(1))
            if target is None:
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{document.relative_to(root)} -> {target}")
    return broken


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    broken = find_broken_links(root)
    if broken:
        raise SystemExit("broken local Markdown links:\n" + "\n".join(broken))
    print(f"verified local Markdown links in {len(documentation_paths(root))} files")


if __name__ == "__main__":
    main()
