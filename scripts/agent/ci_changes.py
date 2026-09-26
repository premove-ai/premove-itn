"""Classify changed paths for the conditional CI workflow."""

from __future__ import annotations

import os
import subprocess
import sys

_HARNESS_FILES = {"tests/test_rtk_hook.py", "tests/test_codegraph_mcp.py"}
_DOC_FILES = {
    "README.md",
    "CONTRIBUTING.md",
    "docs.json",
    "index.mdx",
    "style.css",
    "site.js",
    "premove-icon.png",
    "scripts/check_local_links.py",
    "tests/test_docs_routes.py",
}
_POLICY_FILES = {
    "AGENTS.md",
    "AGENT_HARNESS.md",
    "CHANGELOG.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
}
_POLICY_PREFIXES = (
    "LICENSES/",
    ".github/PULL_REQUEST_TEMPLATE/",
    ".github/ISSUE_TEMPLATE/",
)


def _is_harness_path(path: str) -> bool:
    if path.startswith((".codex/", "scripts/agent/")):
        return True
    if path in _HARNESS_FILES:
        return True
    return path.startswith("tests/test_agent_") and path.endswith(".py")


def _is_docs_path(path: str) -> bool:
    if path in _DOC_FILES:
        return True
    if path.startswith(("docs/", "itn/")):
        return True
    return path.startswith(("benchmarks/", "eval/")) and path.endswith((".md", ".mdx"))


def _is_policy_path(path: str) -> bool:
    if path in _POLICY_FILES:
        return True
    return path.startswith(_POLICY_PREFIXES)


def classify(paths: list[str]) -> tuple[bool, bool, bool, bool, bool]:
    """Return whether harness, docs, Python, Rust, and package checks are required."""
    harness = docs = python = rust = package = False

    for path in paths:
        if (
            path == "scripts/agent/ci_changes.py"
            or path.startswith(".github/workflows/")
            or path == "uv.toml"
        ):
            harness = docs = python = rust = package = True
        elif path == "uv.lock":
            harness = docs = python = package = True
        elif path == "pyproject.toml":
            docs = python = package = True
        elif path.startswith("rust/"):
            python = rust = package = True
        elif _is_harness_path(path):
            harness = True
        elif _is_docs_path(path):
            docs = True
        elif (
            path.startswith(("src/", "tests/"))
            or path == "scripts/check_frozen_boundaries.py"
        ):
            python = True
        elif path == "scripts/inspect_release_artifact.py":
            package = True
        elif _is_policy_path(path):
            continue
        else:
            # A new surface gets the full suite until it is classified.
            harness = docs = python = rust = package = True

    return harness, docs, python, rust, package


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: ci_changes.py BASE HEAD", file=sys.stderr)
        return 2

    paths = (
        subprocess.check_output(
            ["git", "diff", "--name-only", "-z", sys.argv[1], sys.argv[2]]
        )
        .decode()
        .split("\0")
    )
    required = classify([path for path in paths if path])

    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        for name, value in zip(
            ("harness", "docs", "python", "rust", "package"),
            required,
            strict=True,
        ):
            print(f"{name}={str(value).lower()}", file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
