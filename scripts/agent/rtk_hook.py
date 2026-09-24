"""Adapt Codex PreToolUse events to the pinned RTK rewrite interface."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

Which = Callable[[str], str | None]
Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class RtkError(RuntimeError):
    """RTK cannot be resolved for the Codex hook."""


def recall_status(output: str) -> tuple[str, bool]:
    """Return the configured recall mode and whether it is effective."""
    mode = "<unknown>"
    for line in output.splitlines():
        if line.startswith("recall mode:"):
            mode = line.partition(":")[2].strip()
            break
    disabled = any(
        marker in output
        for marker in (
            "RTK_RECALL=0",
            "RTK_TEE=0",
            "recovery disabled for this environment",
        )
    )
    return mode, not disabled


def standalone_rtk_path(home: Path | None = None) -> Path:
    """Return the repository bootstrap's machine-local RTK path."""
    executable = "rtk.exe" if os.name == "nt" else "rtk"
    return (home or Path.home()) / ".local" / "bin" / executable


def resolve_rtk(*, which: Which = shutil.which, home: Path | None = None) -> str:
    """Resolve RTK from PATH or the repository bootstrap install location."""
    executable = which("rtk")
    if executable:
        return executable
    standalone = standalone_rtk_path(home)
    if standalone.is_file() and (os.name == "nt" or os.access(standalone, os.X_OK)):
        return str(standalone)
    raise RtkError(
        "RTK is unavailable. Run: uv run --locked python scripts/agent/bootstrap.py"
    )


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def rewrite_event(
    event: object,
    *,
    which: Which = shutil.which,
    home: Path | None = None,
    runner: Runner = _run,
) -> dict[str, object] | None:
    """Return a Codex command rewrite, or fail open for unsupported input."""
    if not isinstance(event, dict) or event.get("tool_name") != "Bash":
        return None
    tool_input = event.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        rtk = resolve_rtk(which=which, home=home)
        result = runner((rtk, "hook", "check", "--agent", "codex", command))
    except (OSError, RtkError):
        return None
    if result.returncode != 0:
        return None
    rewritten = result.stdout.strip()
    if not rewritten or rewritten == command:
        return None
    updated_input = dict(tool_input)
    updated_input["command"] = rewritten
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": updated_input,
        }
    }


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        return 0
    rewritten = rewrite_event(event)
    if rewritten is not None:
        print(json.dumps(rewritten, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
