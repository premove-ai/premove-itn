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
