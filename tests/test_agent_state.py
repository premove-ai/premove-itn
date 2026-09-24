from __future__ import annotations

import json
from pathlib import Path

from scripts.agent import state


class StateRunner:
    def __init__(self, status: str = "", *, diff_check: int = 0) -> None:
        self.status = status
        self.diff_check = diff_check
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: object, _root: Path) -> state.CommandResult:
        normalized = tuple(command)  # type: ignore[arg-type]
        self.commands.append(normalized)
        if normalized[-2:] == ("branch", "--show-current"):
            return state.CommandResult(0, "chore/example\n")
        if normalized[-3:] == ("rev-parse", "--short", "HEAD"):
            return state.CommandResult(0, "abcdef1\n")
        if normalized[-1] == "@{upstream}":
            return state.CommandResult(1, stderr="no upstream")
        if normalized[-2:] == ("status", "--porcelain=v1"):
            return state.CommandResult(0, self.status)
        if normalized[-2:] == ("diff", "--check"):
            return state.CommandResult(self.diff_check, stderr="whitespace error")
        raise AssertionError(normalized)


def test_clean_tree_has_compact_output(tmp_path: Path) -> None:
    result = state.inspect_state(root=tmp_path, runner=StateRunner())

    assert result.changed == ()
    assert result.diff_check == "clean"
    assert "working tree: 0 modified, 0 untracked" in state.format_text(result)
    assert "changed: none" in state.format_text(result)


def test_staged_unstaged_and_untracked_files_are_counted(tmp_path: Path) -> None:
    runner = StateRunner("M  staged.py\n M modified.py\n?? new.py\nMM both.py\n")

    result = state.inspect_state(root=tmp_path, runner=runner)

    assert (result.staged, result.modified, result.untracked) == (2, 2, 1)
    assert result.changed == ("both.py", "modified.py", "new.py", "staged.py")


def test_failed_diff_check_is_reported(tmp_path: Path) -> None:
    result = state.inspect_state(root=tmp_path, runner=StateRunner(diff_check=2))

    assert result.diff_check == "failed"
    assert "diff-check: failed" in state.format_text(result)


def test_json_shape_is_machine_readable(tmp_path: Path) -> None:
    result = state.inspect_state(root=tmp_path, runner=StateRunner("?? new.py\n"))
    document = json.loads(json.dumps(state.asdict(result)))

    assert document["branch"] == "chore/example"
    assert document["changed"] == ["new.py"]


def test_inspection_does_not_modify_repository(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    marker.write_text("unchanged")
    before = marker.read_bytes()

    state.inspect_state(root=tmp_path, runner=StateRunner())

    assert marker.read_bytes() == before
