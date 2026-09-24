"""Bootstrap the repository-local coding-agent harness."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

CODEGRAPH_VERSION = "1.6.0"
ROOT = Path(__file__).resolve().parents[2]
INSTALLER_BASE_URL = (
    f"https://raw.githubusercontent.com/colbymchenry/codegraph/v{CODEGRAPH_VERSION}"
)
Runner = Callable[[Sequence[str], Path], str]
Which = Callable[[str], str | None]
Installer = Callable[[Path], str]


class BootstrapError(RuntimeError):
    """A deterministic agent bootstrap failure."""


def standalone_codegraph_path() -> Path:
    """Return the executable path used by CodeGraph's standalone installer."""
    if platform.system() == "Windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if not local_app_data:
            raise BootstrapError("LOCALAPPDATA is unavailable for CodeGraph.")
        return Path(local_app_data) / "codegraph" / "current" / "bin" / "codegraph.cmd"
    return Path.home() / ".local" / "bin" / "codegraph"


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
            "    uv run --locked python scripts/agent/bootstrap.py"
        )
    if not (root / "pyproject.toml").is_file() or not (root / "AGENTS.md").is_file():
        raise BootstrapError(f"Not a Premove ITN repository root: {root}")


def _install_codegraph(root: Path) -> str:
    """Run CodeGraph's pinned standalone installer and return its executable."""
    windows = platform.system() == "Windows"
    script_name = "install.ps1" if windows else "install.sh"
    url = f"{INSTALLER_BASE_URL}/{script_name}"
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            script = Path(temporary_directory) / script_name
            urllib.request.urlretrieve(url, script)
            environment = os.environ.copy()
            environment["CODEGRAPH_VERSION"] = f"v{CODEGRAPH_VERSION}"
            command = (
                (
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    script,
                )
                if windows
                else ("/bin/sh", script)
            )
            subprocess.run(command, cwd=root, env=environment, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise BootstrapError(
            f"Could not install CodeGraph {CODEGRAPH_VERSION}: {error}"
        ) from error

    return str(standalone_codegraph_path())


def _ensure_codegraph(
    root: Path,
    which: Which,
    runner: Runner,
    installer: Installer,
) -> str:
    codegraph = which("codegraph")
    if codegraph is None:
        codegraph = installer(root)
        installed_binary = Path(codegraph)
        if not installed_binary.is_file() or (
            platform.system() != "Windows" and not os.access(installed_binary, os.X_OK)
        ):
            raise BootstrapError(
                "CodeGraph installer completed, but its executable could not be "
                f"resolved at {codegraph}."
            )

    found_version = runner((codegraph, "--version"), root).strip().removeprefix("v")
    if found_version != CODEGRAPH_VERSION:
        raise BootstrapError(
            "CodeGraph version mismatch.\n\n"
            f"Required by this repository: {CODEGRAPH_VERSION}\n"
            f"Found: {found_version or '<unknown>'}\n\n"
            "Remove or replace the existing CodeGraph installation, then rerun:\n"
            "    uv run --locked python scripts/agent/bootstrap.py"
        )
    print(f"✓ CodeGraph {CODEGRAPH_VERSION}")
    return codegraph


def _read_status(root: Path, codegraph: str, runner: Runner) -> dict[str, object]:
    try:
        status = json.loads(runner((codegraph, "status", "--json"), root))
    except json.JSONDecodeError as error:
        raise BootstrapError(
            "CodeGraph returned invalid health status JSON."
        ) from error
    if not isinstance(status, dict):
        raise BootstrapError("CodeGraph returned invalid health status JSON.")
    return status


def _ensure_index(root: Path, codegraph: str, runner: Runner) -> None:
    status = _read_status(root, codegraph, runner)
    if status.get("initialized") is not True:
        runner((codegraph, "init", "--yes"), root)
        print("+ initialized CodeGraph")
        status = _read_status(root, codegraph, runner)
    else:
        print("✓ CodeGraph index exists")

    counts = (
        status.get("fileCount", 0),
        status.get("nodeCount", 0),
        status.get("edgeCount", 0),
    )
    if (
        status.get("initialized") is not True
        or status.get("index", {}).get("state") != "complete"
        or status.get("index", {}).get("builtWithVersion") != CODEGRAPH_VERSION
        or status.get("index", {}).get("pendingRefs") != 0
        or status.get("index", {}).get("reindexRecommended") is not False
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
    installer: Installer = _install_codegraph,
) -> None:
    """Install, initialize, and verify the pinned agent harness."""
    print("Premove ITN agent bootstrap\n")
    _require_repository_root(root, cwd or Path.cwd())
    print("✓ repository root")
    codegraph = _ensure_codegraph(root, which, runner, installer)
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
