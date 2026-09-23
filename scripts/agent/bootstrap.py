"""Bootstrap the repository-local coding-agent harness."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

CODEGRAPH_VERSION = "1.6.0"
ROOT = Path(__file__).resolve().parents[2]
INSTALL_COMMAND = (
    "npx",
    "--yes",
    f"@colbymchenry/codegraph@{CODEGRAPH_VERSION}",
    "install",
    "--target=codex",
    "--location=local",
    "--yes",
)

Runner = Callable[[Sequence[str], Path], str]
Which = Callable[[str], str | None]


class BootstrapError(RuntimeError):
    """A deterministic agent bootstrap failure."""


def _run(command: Sequence[str], root: Path) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise BootstrapError(
            f"Command failed: {' '.join(command)}\n{detail.strip()}"
        ) from error
    return result.stdout


def _require_repository_root(root: Path, cwd: Path) -> None:
    if cwd.resolve() != root.resolve():
        raise BootstrapError(
            "Run this command from the Premove ITN repository root:\n"
            "    python scripts/agent/bootstrap.py"
        )
    if not (root / "pyproject.toml").is_file() or not (root / "AGENTS.md").is_file():
        raise BootstrapError(f"Not a Premove ITN repository root: {root}")


def _ensure_codegraph(root: Path, which: Which, runner: Runner) -> str:
    codegraph = which("codegraph")
    if codegraph is None:
        if which("npx") is None:
            raise BootstrapError(
                f"Premove ITN agent setup requires CodeGraph {CODEGRAPH_VERSION}.\n\n"
                "CodeGraph is not installed and npx is unavailable.\n\n"
                "Install CodeGraph, then rerun:\n"
                "    python scripts/agent/bootstrap.py"
            )
        runner(INSTALL_COMMAND, root)
        codegraph = which("codegraph")
        if codegraph is None:
            raise BootstrapError(
                "CodeGraph installation completed, but codegraph is not on PATH.\n"
                "Add the installed binary to PATH, then rerun bootstrap."
            )

    found_version = runner((codegraph, "--version"), root).strip().removeprefix("v")
    if found_version != CODEGRAPH_VERSION:
        install_command = " ".join(INSTALL_COMMAND)
        raise BootstrapError(
            "CodeGraph version mismatch.\n\n"
            f"Required by this repository: {CODEGRAPH_VERSION}\n"
            f"Found: {found_version or '<unknown>'}\n\n"
            f"Run:\n{install_command}"
        )
    print(f"✓ CodeGraph {CODEGRAPH_VERSION}")
    return codegraph


def _ensure_index(root: Path, codegraph: str, runner: Runner) -> None:
    index_exists = (root / ".codegraph").is_dir()
    if not index_exists:
        runner((codegraph, "init"), root)
        print("+ initialized CodeGraph")
    else:
        print("✓ CodeGraph index exists")

    try:
        status = json.loads(runner((codegraph, "status", "--json"), root))
    except json.JSONDecodeError as error:
        raise BootstrapError(
            "CodeGraph returned invalid health status JSON."
        ) from error

    counts = (
        status.get("fileCount", 0),
        status.get("nodeCount", 0),
        status.get("edgeCount", 0),
    )
    if (
        status.get("initialized") is not True
        or status.get("index", {}).get("state") != "complete"
        or not all(isinstance(count, int) and count > 0 for count in counts)
    ):
        raise BootstrapError("CodeGraph index is not healthy.")
    print(f"✓ {counts[0]} indexed files")
    print(f"✓ {counts[1]} nodes")
    print(f"✓ {counts[2]} edges")


def bootstrap(
    *,
    root: Path = ROOT,
    cwd: Path | None = None,
    which: Which = shutil.which,
    runner: Runner = _run,
) -> None:
    """Install, initialize, and verify the pinned agent harness."""
    print("Premove ITN agent bootstrap\n")
    _require_repository_root(root, cwd or Path.cwd())
    print("✓ repository root")
    codegraph = _ensure_codegraph(root, which, runner)
    _ensure_index(root, codegraph, runner)
    print("\nAgent harness ready.")


def main() -> int:
    """Run bootstrap and convert expected failures to a nonzero exit."""
    try:
        bootstrap()
    except BootstrapError as error:
        print(f"Agent bootstrap failed:\n{error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
