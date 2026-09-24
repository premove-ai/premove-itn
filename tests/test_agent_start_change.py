from __future__ import annotations

from pathlib import Path

import pytest

from scripts.agent import start_change


class BranchRunner:
    def __init__(
        self,
        *,
        dirty: bool = False,
        local_exists: bool = False,
        remote_exists: bool = False,
        base_exists: bool = True,
        fetch_fails: bool = False,
    ) -> None:
        self.dirty = dirty
        self.local_exists = local_exists
        self.remote_exists = remote_exists
        self.base_exists = base_exists
        self.fetch_fails = fetch_fails
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: object, _root: Path) -> start_change.CommandResult:
        normalized = tuple(command)  # type: ignore[arg-type]
        self.commands.append(normalized)
        if normalized[1:3] == ("status", "--porcelain"):
            return start_change.CommandResult(0, " M file.py\n" if self.dirty else "")
        if normalized[1:4] == ("show-ref", "--verify", "--quiet"):
            reference = normalized[-1]
            if reference == "refs/heads/chore/new":
                exists = self.local_exists
            elif reference == "refs/remotes/origin/chore/new":
                exists = self.remote_exists
            else:
                exists = self.base_exists
            return start_change.CommandResult(0 if exists else 1)
        if normalized[1] == "fetch":
            return start_change.CommandResult(1 if self.fetch_fails else 0)
        if normalized[1] == "switch":
            return start_change.CommandResult(0)
        raise AssertionError(normalized)


def test_fetches_then_creates_branch_from_remote_base(tmp_path: Path) -> None:
    runner = BranchRunner()

    start_change.start_change(
        "epic/base", "chore/new", root=tmp_path, cwd=tmp_path, runner=runner
    )

    fetch = ("git", "fetch", "origin", "epic/base")
    switch = (
        "git",
        "switch",
        "-c",
        "chore/new",
        "--no-track",
        "origin/epic/base",
    )
    assert runner.commands.index(fetch) < runner.commands.index(switch)
    assert not any(command[1] in {"commit", "push"} for command in runner.commands)


@pytest.mark.parametrize(
    ("runner", "message"),
    (
        (BranchRunner(dirty=True), "not clean"),
        (BranchRunner(local_exists=True), "already exists"),
        (BranchRunner(remote_exists=True), "already exists"),
        (BranchRunner(base_exists=False), "Remote base does not exist"),
    ),
)
def test_invalid_start_is_rejected(
    tmp_path: Path, runner: BranchRunner, message: str
) -> None:
    with pytest.raises(start_change.StartChangeError, match=message):
        start_change.start_change(
            "epic/base", "chore/new", root=tmp_path, cwd=tmp_path, runner=runner
        )


def test_main_target_is_rejected_before_git(tmp_path: Path) -> None:
    runner = BranchRunner()

    with pytest.raises(start_change.StartChangeError, match="main"):
        start_change.start_change(
            "epic/base", "main", root=tmp_path, cwd=tmp_path, runner=runner
        )

    assert runner.commands == []


def test_fetch_failure_does_not_switch(tmp_path: Path) -> None:
    runner = BranchRunner(fetch_fails=True)

    with pytest.raises(start_change.StartChangeError, match="Command failed"):
        start_change.start_change(
            "epic/base", "chore/new", root=tmp_path, cwd=tmp_path, runner=runner
        )

    assert not any(command[1] == "switch" for command in runner.commands)
