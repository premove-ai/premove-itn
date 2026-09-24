"""Report compact, read-only repository state."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CommandResult:
    """Captured command output without implicit failure handling."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[Sequence[str], Path], CommandResult]


class StateError(RuntimeError):
    """A repository state inspection failure."""


@dataclass(frozen=True)
class RepositoryState:
    """The compact state exposed by the command."""

    branch: str
    head: str
    upstream: str | None
    staged: int
    modified: int
    untracked: int
    diff_check: str
    changed: tuple[str, ...]


def _run(command: Sequence[str], root: Path) -> CommandResult:
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    return CommandResult(result.returncode, result.stdout, result.stderr)


def _git(
    runner: Runner, root: Path, *arguments: str, allow_failure: bool = False
) -> CommandResult:
    result = runner(("git", *arguments), root)
    if result.returncode and not allow_failure:
        detail = result.stderr.strip() or result.stdout.strip()
        raise StateError(f"git {' '.join(arguments)} failed: {detail}")
    return result


def _parse_status(output: str) -> tuple[int, int, int, tuple[str, ...]]:
    staged = modified = untracked = 0
    paths: set[str] = set()
    for line in output.splitlines():
        if len(line) < 3:
            continue
        index, worktree = line[0], line[1]
        path = line[3:].strip('"')
        if " -> " in path:
            path = path.rsplit(" -> ", maxsplit=1)[1].strip('"')
        if index == "?" and worktree == "?":
            untracked += 1
        else:
            staged += index != " "
            modified += worktree != " "
        paths.add(path)
    return staged, modified, untracked, tuple(sorted(paths))


def inspect_state(*, root: Path = ROOT, runner: Runner = _run) -> RepositoryState:
    """Collect repository state without modifying the checkout."""
    branch = _git(runner, root, "branch", "--show-current").stdout.strip() or "detached"
    head = _git(runner, root, "rev-parse", "--short", "HEAD").stdout.strip()
    upstream_result = _git(
        runner,
        root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        allow_failure=True,
    )
    upstream = (
        upstream_result.stdout.strip() if upstream_result.returncode == 0 else None
    )
    status = _git(runner, root, "status", "--porcelain=v1").stdout
    staged, modified, untracked, changed = _parse_status(status)
    diff_result = _git(runner, root, "diff", "--check", allow_failure=True)
    return RepositoryState(
        branch=branch,
        head=head,
        upstream=upstream,
        staged=staged,
        modified=modified,
        untracked=untracked,
        diff_check="clean" if diff_result.returncode == 0 else "failed",
        changed=changed,
    )


def format_text(state: RepositoryState) -> str:
    """Render the compact human-readable report."""
    total = state.modified + state.untracked
    lines = [
        f"branch: {state.branch}",
        f"head: {state.head}",
        f"upstream: {state.upstream or 'none'}",
        f"working tree: {state.modified} modified, {state.untracked} untracked",
        f"staged: {state.staged}",
        f"diff-check: {state.diff_check}",
    ]
    if state.changed:
        lines.extend(("", "changed:", *(f"  {path}" for path in state.changed)))
    elif total == 0 and state.staged == 0:
        lines.extend(("", "changed: none"))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run state inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    arguments = parser.parse_args(argv)
    try:
        state = inspect_state()
    except StateError as error:
        print(f"Repository state failed: {error}")
        return 1
    print(json.dumps(asdict(state), indent=2) if arguments.json else format_text(state))
    return 1 if state.diff_check == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
