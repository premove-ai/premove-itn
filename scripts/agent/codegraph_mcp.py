"""Launch the repository's pinned CodeGraph MCP server."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path


class LauncherError(RuntimeError):
    """CodeGraph cannot be resolved for the MCP server."""


def resolve_codegraph(
    *,
    which: Callable[[str], str | None] = shutil.which,
    home: Path | None = None,
    system: str | None = None,
    environment: Mapping[str, str] = os.environ,
) -> str:
    """Resolve CodeGraph from PATH or its standalone install location."""
    executable = which("codegraph")
    if executable:
        return executable

    if (system or platform.system()) == "Windows":
        local_app_data = environment.get("LOCALAPPDATA")
        candidate = (
            Path(local_app_data) / "codegraph" / "current" / "bin" / "codegraph.cmd"
            if local_app_data
            else None
        )
    else:
        candidate = (home or Path.home()) / ".local" / "bin" / "codegraph"

    if candidate is not None and candidate.is_file():
        return str(candidate)
    raise LauncherError(
        "CodeGraph is unavailable. Run: uv run --locked python "
        "scripts/agent/bootstrap.py"
    )


def main() -> int:
    """Run the MCP server with inherited standard streams."""
    try:
        codegraph = resolve_codegraph()
    except LauncherError as error:
        print(error, file=os.sys.stderr)
        return 1
    return subprocess.run((codegraph, "serve", "--mcp"), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
