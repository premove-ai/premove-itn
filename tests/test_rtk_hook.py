from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

from scripts.agent import rtk_hook


def _result(returncode: int, stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(("rtk", "rewrite"), returncode, stdout, "")


def test_supported_bash_command_is_rewritten() -> None:
    event = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "git status", "description": "Check status"},
    }

    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _result(0, "rtk git status\n")

    rewritten = rtk_hook.rewrite_event(
        event,
        which=lambda _tool: "/bin/rtk",
        runner=run,
    )

    assert rewritten == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": {
                "command": "rtk git status",
                "description": "Check status",
            },
        }
    }
    assert commands == [("/bin/rtk", "hook", "check", "--agent", "codex", "git status")]


def test_unsupported_command_passes_through() -> None:
    rewritten = rtk_hook.rewrite_event(
        {"tool_name": "Bash", "tool_input": {"command": "echo hello"}},
        which=lambda _tool: "/bin/rtk",
        runner=lambda command: _result(1),
    )

    assert rewritten is None


def test_self_compressing_check_passes_through_without_rtk() -> None:
    commands: list[tuple[str, ...]] = []

    rewritten = rtk_hook.rewrite_event(
        {
            "tool_name": "Bash",
            "tool_input": {
                "command": "uv run --locked python scripts/agent/check.py full"
            },
        },
        which=lambda _tool: "/bin/rtk",
        runner=lambda command: commands.append(tuple(command)) or _result(0),
    )

    assert rewritten is None
    assert commands == []


def test_check_path_as_data_can_still_be_rewritten() -> None:
    rewritten = rtk_hook.rewrite_event(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "rg check.py scripts/agent/check.py"},
        },
        which=lambda _tool: "/bin/rtk",
        runner=lambda _command: _result(0, "rtk rg check.py scripts/agent/check.py\n"),
    )

    assert rewritten is not None


def test_check_in_shell_chain_can_still_be_rewritten() -> None:
    command = "uv run python scripts/agent/check.py full && git status"
    rewritten = rtk_hook.rewrite_event(
        {"tool_name": "Bash", "tool_input": {"command": command}},
        which=lambda _tool: "/bin/rtk",
        runner=lambda _command: _result(0, f"rtk test {command}\n"),
    )

    assert rewritten is not None


def test_check_in_newline_chain_can_still_be_rewritten() -> None:
    command = "uv run python scripts/agent/check.py full\ngit status"
    rewritten = rtk_hook.rewrite_event(
        {"tool_name": "Bash", "tool_input": {"command": command}},
        which=lambda _tool: "/bin/rtk",
        runner=lambda _command: _result(0, f"rtk test {command}\n"),
    )

    assert rewritten is not None


def test_lookalike_check_path_can_still_be_rewritten() -> None:
    command = "python /tmp/attack-scripts/agent/check.py full"
    rewritten = rtk_hook.rewrite_event(
        {"tool_name": "Bash", "tool_input": {"command": command}},
        which=lambda _tool: "/bin/rtk",
        runner=lambda _command: _result(0, f"rtk test {command}\n"),
    )

    assert rewritten is not None


def test_check_path_after_non_python_command_can_still_be_rewritten() -> None:
    command = "echo python scripts/agent/check.py"
    rewritten = rtk_hook.rewrite_event(
        {"tool_name": "Bash", "tool_input": {"command": command}},
        which=lambda _tool: "/bin/rtk",
        runner=lambda _command: _result(0, f"rtk test {command}\n"),
    )

    assert rewritten is not None


def test_disabled_command_passes_through() -> None:
    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _result(1)

    rewritten = rtk_hook.rewrite_event(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "RTK_DISABLED=1 git status"},
        },
        which=lambda _tool: "/bin/rtk",
        runner=run,
    )

    assert rewritten is None
    assert commands == [
        (
            "/bin/rtk",
            "hook",
            "check",
            "--agent",
            "codex",
            "RTK_DISABLED=1 git status",
        )
    ]


def test_missing_rtk_fails_open(tmp_path: Path) -> None:
    rewritten = rtk_hook.rewrite_event(
        {"tool_name": "Bash", "tool_input": {"command": "git status"}},
        which=lambda _tool: None,
        home=tmp_path,
    )

    assert rewritten is None


def test_non_bash_event_passes_through() -> None:
    assert (
        rtk_hook.rewrite_event(
            {"tool_name": "Read", "tool_input": {"command": "git status"}}
        )
        is None
    )


def test_main_emits_codex_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    event = {"tool_name": "Bash", "tool_input": {"command": "git status"}}
    monkeypatch.setattr(rtk_hook.sys, "stdin", io.StringIO(json.dumps(event)))
    monkeypatch.setattr(
        rtk_hook,
        "rewrite_event",
        lambda _event: {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": {"command": "rtk git status"},
            }
        },
    )

    assert rtk_hook.main() == 0
    assert (
        json.loads(capsys.readouterr().out)["hookSpecificOutput"]["permissionDecision"]
        == "allow"
    )
