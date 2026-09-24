"""Create a clean change branch directly from a remote base."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[Sequence[str], Path], CommandResult]


class StartChangeError(RuntimeError):
    """A safe branch-creation precondition failed."""


def _run(command: Sequence[str], root: Path) -> CommandResult:
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    return CommandResult(result.returncode, result.stdout, result.stderr)


def _require_success(result: CommandResult, command: Sequence[str]) -> None:
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise StartChangeError(f"Command failed: {' '.join(command)}\n{detail}")


def start_change(
    base: str,
    branch: str,
    *,
    root: Path = ROOT,
    cwd: Path | None = None,
    runner: Runner = _run,
) -> None:
    """Create branch from the exact fetched remote base."""
    if (cwd or Path.cwd()).resolve() != root.resolve():
        raise StartChangeError("Run this command from the repository root.")
    if branch == "main":
        raise StartChangeError("Refusing to create or replace main.")
    if not branch or branch.startswith("-") or not base or base.startswith("-"):
        raise StartChangeError("Base and branch must be valid Git branch names.")

    status_command = ("git", "status", "--porcelain", "--untracked-files=all")
    status = runner(status_command, root)
    _require_success(status, status_command)
    if status.stdout.strip():
        raise StartChangeError("Working tree is not clean.")

    for reference in (f"refs/heads/{branch}",):
        command = ("git", "show-ref", "--verify", "--quiet", reference)
        result = runner(command, root)
        if result.returncode == 0:
            raise StartChangeError(f"Target branch already exists: {branch}")
        if result.returncode != 1:
            _require_success(result, command)

    remote_target = (
        "git",
        "ls-remote",
        "--exit-code",
        "--heads",
        "origin",
        f"refs/heads/{branch}",
    )
    remote_target_result = runner(remote_target, root)
    if remote_target_result.returncode == 0:
        raise StartChangeError(f"Target branch already exists: origin/{branch}")
    if remote_target_result.returncode != 2:
        _require_success(remote_target_result, remote_target)

    fetch = ("git", "fetch", "origin", base)
    _require_success(runner(fetch, root), fetch)

    remote_base = (
        "git",
        "show-ref",
        "--verify",
        "--quiet",
        f"refs/remotes/origin/{base}",
    )
    if runner(remote_base, root).returncode != 0:
        raise StartChangeError(f"Remote base does not exist: origin/{base}")

    switch = ("git", "switch", "-c", branch, "--no-track", f"origin/{base}")
    _require_success(runner(switch, root), switch)
    print(f"branch: {branch}")
    print(f"base: origin/{base}")
    print("working tree: clean")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--branch", required=True)
    arguments = parser.parse_args(argv)
    try:
        start_change(arguments.base, arguments.branch)
    except StartChangeError as error:
        print(f"Could not start change: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
