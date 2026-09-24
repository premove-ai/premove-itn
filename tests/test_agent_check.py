from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.agent import check, rtk_hook

RECALL_HASH = "abc123def456"
RAW_FAILURE = "raw output\nraw error\n"


class CheckRunner:
    def __init__(
        self,
        *,
        changed: tuple[str, ...] = (),
        untracked: tuple[str, ...] = (),
        fail_at: int | None = None,
    ) -> None:
        self.changed = changed
        self.untracked = untracked
        self.fail_at = fail_at
        self.commands: list[tuple[tuple[str, ...], Path, dict[str, str] | None]] = []

    def __call__(
        self,
        command: object,
        root: Path,
        environment: dict[str, str] | None,
    ) -> str:
        normalized = tuple(command)  # type: ignore[arg-type]
        self.commands.append((normalized, root, environment))
        if self.fail_at == len(self.commands):
            raise check.CheckError("expected failure")
        if normalized[:3] == ("git", "diff", "--name-only"):
            return "\n".join(self.changed)
        if normalized[:3] == ("git", "ls-files", "--others"):
            return "\n".join(self.untracked)
        if normalized[:3] == ("uv", "build", "--out-dir"):
            output = Path(normalized[3])
            output.mkdir(parents=True)
            (output / "package.whl").write_bytes(b"wheel")
            (output / "package.tar.gz").write_bytes(b"sdist")
        if normalized[:2] == ("uv", "venv"):
            venv = Path(normalized[-1])
            (venv / "bin").mkdir(parents=True)
        return ""

    @property
    def command_list(self) -> list[tuple[str, ...]]:
        return [command for command, _, _ in self.commands]


def _touch(root: Path, *paths: str) -> None:
    for value in paths:
        path = root / value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content")


def test_python_profile_runs_exact_commands(tmp_path: Path) -> None:
    runner = CheckRunner()

    check.check_python(root=tmp_path, runner=runner)

    assert runner.command_list == [
        ("uv", "run", "--locked", "ruff", "format", "--check", "."),
        ("uv", "run", "--locked", "ruff", "check", "."),
        ("uv", "run", "--locked", "pytest"),
        ("uv", "run", "--locked", "python", "scripts/check_local_links.py"),
        ("uv", "run", "--locked", "python", "scripts/check_frozen_boundaries.py"),
    ]


def test_rust_profile_runs_format_then_tests(tmp_path: Path) -> None:
    runner = CheckRunner()

    check.check_rust(root=tmp_path, runner=runner)

    assert runner.command_list == [
        ("cargo", "fmt", "--manifest-path", "rust/Cargo.toml", "--", "--check"),
        ("cargo", "test", "--manifest-path", "rust/Cargo.toml"),
    ]


def test_profiles_stop_on_first_failure(tmp_path: Path) -> None:
    runner = CheckRunner(fail_at=2)

    with pytest.raises(check.CheckError, match="expected failure"):
        check.check_python(root=tmp_path, runner=runner)

    assert len(runner.commands) == 2


def test_real_runner_surfaces_failed_command_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(
            1,
            ("tool", "check"),
            output="captured output\n",
            stderr="captured error\n",
        )

    def unavailable() -> str:
        raise check.RtkError("missing")

    monkeypatch.setattr(check, "resolve_rtk", unavailable)
    monkeypatch.setattr(check.subprocess, "run", fail)

    with pytest.raises(check.CheckError, match="tool check"):
        check._run_validation(("tool", "check"), tmp_path)

    captured = capsys.readouterr()
    assert "captured output" in captured.out
    assert "captured error" in captured.err


def test_validation_success_is_one_line_without_using_rtk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands: list[tuple[str, ...]] = []

    def succeed(command: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        normalized = tuple(command)  # type: ignore[arg-type]
        commands.append(normalized)
        return subprocess.CompletedProcess(normalized, 0, "many lines\n", "warning\n")

    def unexpected_rtk() -> str:
        raise AssertionError("successful validation must not resolve RTK")

    monkeypatch.setattr(check, "resolve_rtk", unexpected_rtk)
    monkeypatch.setattr(check.subprocess, "run", succeed)

    check._run_validation(("uv", "run", "--locked", "pytest"), tmp_path)

    captured = capsys.readouterr()
    assert captured.out == "PASS pytest\n"
    assert captured.err == ""
    assert commands == [("uv", "run", "--locked", "pytest")]


def test_validation_failure_is_compacted_with_recall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands: list[tuple[str, ...]] = []

    def run(command: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        normalized = tuple(command)  # type: ignore[arg-type]
        commands.append(normalized)
        if len(commands) == 1:
            raise subprocess.CalledProcessError(
                1, normalized, output="raw output\n", stderr="raw error\n"
            )
        if normalized[1:2] == ("recall",):
            return subprocess.CompletedProcess(normalized, 0, RAW_FAILURE, "")
        compact = f"FAIL pytest\n[full output: rtk recall {RECALL_HASH}]\n"
        return subprocess.CompletedProcess(normalized, 1, compact, "")

    monkeypatch.setattr(check, "resolve_rtk", lambda: "/bin/rtk")
    monkeypatch.setattr(check.subprocess, "run", run)

    with pytest.raises(check.CheckError, match="uv run --locked pytest"):
        check._run_validation(("uv", "run", "--locked", "pytest"), tmp_path)

    captured = capsys.readouterr()
    assert captured.out == f"FAIL pytest\n[full output: rtk recall {RECALL_HASH}]\n"
    assert "raw output" not in captured.out
    assert commands[0] == ("uv", "run", "--locked", "pytest")
    assert commands[1][:2] == ("/bin/rtk", "test")
    assert commands[1][-1] == "output.log"
    assert "premove-failure-" not in " ".join(commands[1])
    assert commands[2] == ("/bin/rtk", "recall", RECALL_HASH, "--full")


def test_validation_failure_is_raw_when_recall_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands: list[tuple[str, ...]] = []

    def run(command: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        normalized = tuple(command)  # type: ignore[arg-type]
        commands.append(normalized)
        if len(commands) == 1:
            raise subprocess.CalledProcessError(
                1, normalized, output="raw output\n", stderr="raw error\n"
            )
        return subprocess.CompletedProcess(
            normalized, 1, "compacted without recall\n", ""
        )

    monkeypatch.setattr(check, "resolve_rtk", lambda: "/bin/rtk")
    monkeypatch.setattr(check.subprocess, "run", run)

    with pytest.raises(check.CheckError):
        check._run_validation(("cargo", "test"), tmp_path)

    captured = capsys.readouterr()
    assert captured.out == "raw output\n"
    assert captured.err == "raw error\n"


def test_existing_recall_marker_does_not_replace_raw_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw = f"raw output\n[full output: rtk recall {RECALL_HASH}]\n"

    def fail(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, ("cargo", "test"), output=raw)

    def unexpected_rtk() -> str:
        raise AssertionError("pre-existing recall markers must use raw fallback")

    monkeypatch.setattr(check, "resolve_rtk", unexpected_rtk)
    monkeypatch.setattr(check.subprocess, "run", fail)

    with pytest.raises(check.CheckError):
        check._run_validation(("cargo", "test"), tmp_path)

    assert capsys.readouterr().out == raw


def test_invalid_recall_hash_falls_back_to_raw_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands: list[tuple[str, ...]] = []

    def run(command: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        normalized = tuple(command)  # type: ignore[arg-type]
        commands.append(normalized)
        if len(commands) == 1:
            raise subprocess.CalledProcessError(1, normalized, output=RAW_FAILURE)
        return subprocess.CompletedProcess(
            normalized, 1, "FAIL pytest\n[full output: rtk recall garbage]\n", ""
        )

    monkeypatch.setattr(check, "resolve_rtk", lambda: "/bin/rtk")
    monkeypatch.setattr(check.subprocess, "run", run)

    with pytest.raises(check.CheckError):
        check._run_validation(("uv", "run", "pytest"), tmp_path)

    assert capsys.readouterr().out == RAW_FAILURE
    assert len(commands) == 2


@pytest.mark.parametrize(
    "recall_result",
    (
        subprocess.CompletedProcess(("rtk", "recall"), 1, "", "not found\n"),
        subprocess.CompletedProcess(("rtk", "recall"), 0, "different output\n", ""),
    ),
    ids=("recall-command-failure", "recall-content-mismatch"),
)
def test_unrecoverable_recall_falls_back_to_raw_failure(
    recall_result: subprocess.CompletedProcess[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    def run(command: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        normalized = tuple(command)  # type: ignore[arg-type]
        if calls == 1:
            raise subprocess.CalledProcessError(1, normalized, output=RAW_FAILURE)
        if calls == 2:
            compact = f"FAIL pytest\n[full output: rtk recall {RECALL_HASH}]\n"
            return subprocess.CompletedProcess(normalized, 1, compact, "")
        return recall_result

    monkeypatch.setattr(check, "resolve_rtk", lambda: "/bin/rtk")
    monkeypatch.setattr(check.subprocess, "run", run)

    with pytest.raises(check.CheckError):
        check._run_validation(("uv", "run", "pytest"), tmp_path)

    assert capsys.readouterr().out == RAW_FAILURE


def test_validation_failure_is_raw_when_replay_staging_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(
            1, ("cargo", "test"), output="raw output\n", stderr="raw error\n"
        )

    def staging_failure(_directory: Path) -> Path:
        raise OSError("disk unavailable")

    monkeypatch.setattr(check, "resolve_rtk", lambda: "/bin/rtk")
    monkeypatch.setattr(check, "_replay_script", staging_failure)
    monkeypatch.setattr(check.subprocess, "run", fail)

    with pytest.raises(check.CheckError):
        check._run_validation(("cargo", "test"), tmp_path)

    captured = capsys.readouterr()
    assert captured.out == "raw output\n"
    assert captured.err == "raw error\n"


def test_ruff_failure_uses_error_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[str, ...]] = []

    def run(command: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        normalized = tuple(command)  # type: ignore[arg-type]
        commands.append(normalized)
        if len(commands) == 1:
            raise subprocess.CalledProcessError(1, normalized, output="ruff error\n")
        if normalized[1:2] == ("recall",):
            return subprocess.CompletedProcess(normalized, 0, "ruff error\n", "")
        return subprocess.CompletedProcess(
            normalized,
            1,
            f"error\n[full output: rtk recall {RECALL_HASH}]\n",
            "",
        )

    monkeypatch.setattr(check, "resolve_rtk", lambda: "/bin/rtk")
    monkeypatch.setattr(check.subprocess, "run", run)

    with pytest.raises(check.CheckError):
        check._run_validation(
            ("uv", "run", "ruff", "check", "tests/test_example.py"), tmp_path
        )

    assert commands[1][:2] == ("/bin/rtk", "err")


def test_failure_replay_uses_relative_paths_when_temp_path_has_spaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    temporary_parent = tmp_path / "path with spaces"
    temporary_parent.mkdir()

    def run(command: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        normalized = tuple(command)  # type: ignore[arg-type]
        if normalized[1:2] == ("recall",):
            assert kwargs["cwd"] == tmp_path
            return subprocess.CompletedProcess(normalized, 0, RAW_FAILURE, "")
        assert " " in str(kwargs["cwd"])
        assert all(" " not in argument for argument in normalized[2:])
        return subprocess.CompletedProcess(
            normalized,
            1,
            f"error\n[full output: rtk recall {RECALL_HASH}]\n",
            "",
        )

    monkeypatch.setattr(check.tempfile, "tempdir", str(temporary_parent))
    monkeypatch.setattr(check, "resolve_rtk", lambda: "/bin/rtk")
    monkeypatch.setattr(check.subprocess, "run", run)

    compacted = check._print_compact_failure(
        ("cargo", "test"), "raw output\n", "raw error\n", tmp_path, None
    )

    assert compacted
    assert f"rtk recall {RECALL_HASH}" in capsys.readouterr().out


def test_relative_recall_database_falls_back_to_raw(tmp_path: Path) -> None:
    rtk = shutil.which("rtk")
    if rtk is None:
        pytest.skip("RTK is not installed")
    version = subprocess.run(
        (rtk, "--version"), check=False, capture_output=True, text=True
    )
    if version.stdout.strip() != "rtk 0.49.0":
        pytest.skip("pinned RTK 0.49.0 is not installed")
    environment = os.environ.copy()
    environment.pop("RTK_RECALL", None)
    environment.pop("RTK_TEE", None)
    environment["RTK_RECALL_DB"] = "relative-recall.db"
    raw = "noise\n" * 600 + "FAILED: relative recall database\n"

    compacted = check._print_compact_failure(
        ("cargo", "test"), raw, None, tmp_path, environment
    )

    assert not compacted


def test_hook_exempts_check_and_preserves_failure_recovery(tmp_path: Path) -> None:
    rtk = shutil.which("rtk")
    if rtk is None:
        pytest.skip("RTK is not installed")
    version = subprocess.run(
        (rtk, "--version"), check=False, capture_output=True, text=True
    )
    if version.stdout.strip() != "rtk 0.49.0":
        pytest.skip("pinned RTK 0.49.0 is not installed")

    sentinel = "hook_check_sentinel"
    failing_test = tmp_path / f"test_{sentinel}.py"
    failing_test.write_text(
        "def test_long_failure():\n"
        "    print('noise\\n' * 600, end='')\n"
        f"    raise AssertionError({sentinel!r})\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("RTK_RECALL", None)
    environment.pop("RTK_TEE", None)
    environment["RTK_RECALL_DB"] = str(tmp_path / "recall.db")

    command = (
        "uv",
        "run",
        "--locked",
        "python",
        "scripts/agent/check.py",
        "focused",
        "--pytest",
        str(failing_test),
    )
    rewritten = rtk_hook.rewrite_event(
        {
            "tool_name": "Bash",
            "tool_input": {"command": shlex.join(command)},
        }
    )
    assert rewritten is None

    result = subprocess.run(
        command,
        cwd=check.ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    visible = f"{result.stdout}{result.stderr}"

    assert result.returncode != 0
    assert sentinel in visible
    handles = check.RECALL_HINT.findall(visible)
    assert len(handles) == 1, visible

    recall = subprocess.run(
        (rtk, "recall", handles[0], "--full"),
        cwd=check.ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert recall.returncode == 0
    assert "noise\n" in recall.stdout
    assert sentinel in recall.stdout


def test_capture_output_is_raw_and_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def succeed(command: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "src/example.py\n", "")

    monkeypatch.setattr(check.subprocess, "run", succeed)

    output = check._run_capture(("git", "diff", "--name-only"), tmp_path)

    assert output == "src/example.py\n"
    assert capsys.readouterr().out == ""


def test_full_profile_runs_in_canonical_order(tmp_path: Path) -> None:
    runner = CheckRunner()

    check.check_full(root=tmp_path, runner=runner)

    commands = runner.command_list
    assert commands[0] == ("uv", "run", "--locked", "ruff", "format", "--check", ".")
    assert commands.index(
        ("cargo", "fmt", "--manifest-path", "rust/Cargo.toml", "--", "--check")
    ) > commands.index(
        ("uv", "run", "--locked", "python", "scripts/check_frozen_boundaries.py")
    )
    assert commands[-2:] == [
        ("git", "diff", "--check"),
        ("git", "diff", "--cached", "--check"),
    ]


def test_focused_python_uses_explicit_semantic_tests(tmp_path: Path) -> None:
    _touch(tmp_path, "src/premove_itn/example.py", "tests/test_example.py")
    runner = CheckRunner(
        changed=("src/premove_itn/example.py", "tests/test_example.py")
    )

    check.check_focused(
        ("tests/test_example.py",),
        root=tmp_path,
        runner=runner,
        capture_runner=runner,
    )

    assert (
        "uv",
        "run",
        "--locked",
        "ruff",
        "format",
        "--check",
        "src/premove_itn/example.py",
        "tests/test_example.py",
    ) in runner.command_list
    assert (
        "uv",
        "run",
        "--locked",
        "pytest",
        "tests/test_example.py",
    ) in runner.command_list


def test_focused_python_warns_without_semantic_tests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _touch(tmp_path, "src/premove_itn/example.py")
    runner = CheckRunner(changed=("src/premove_itn/example.py",))

    check.check_focused((), root=tmp_path, runner=runner, capture_runner=runner)

    assert "without an explicit semantic pytest target" in capsys.readouterr().err
    assert not any(
        command[:4] == ("uv", "run", "--locked", "pytest")
        for command in runner.command_list
    )


def test_focused_rust_and_docs_use_mechanical_checks(tmp_path: Path) -> None:
    _touch(tmp_path, "rust/src/lib.rs", "README.md")
    runner = CheckRunner(changed=("rust/src/lib.rs", "README.md"))

    check.check_focused((), root=tmp_path, runner=runner, capture_runner=runner)

    assert runner.command_list[0] == ("git", "diff", "--name-only", "HEAD")
    assert (
        "cargo",
        "test",
        "--manifest-path",
        "rust/Cargo.toml",
    ) in runner.command_list
    assert (
        "uv",
        "run",
        "--locked",
        "python",
        "scripts/check_local_links.py",
    ) in runner.command_list
    assert (
        "uv",
        "run",
        "--locked",
        "pytest",
        "-q",
        "tests/test_docs_routes.py",
    ) in runner.command_list


def test_focused_deleted_rust_file_runs_rust_checks(tmp_path: Path) -> None:
    runner = CheckRunner(changed=("rust/src/some_realizer.rs",))

    check.check_focused((), root=tmp_path, runner=runner, capture_runner=runner)

    assert runner.command_list[0] == ("git", "diff", "--name-only", "HEAD")
    assert (
        "cargo",
        "test",
        "--manifest-path",
        "rust/Cargo.toml",
    ) in runner.command_list


def test_focused_deleted_itn_doc_runs_docs_checks(tmp_path: Path) -> None:
    runner = CheckRunner(changed=("itn/docs/foo.md",))

    check.check_focused((), root=tmp_path, runner=runner, capture_runner=runner)

    assert (
        "uv",
        "run",
        "--locked",
        "python",
        "scripts/check_local_links.py",
    ) in runner.command_list
    assert (
        "uv",
        "run",
        "--locked",
        "pytest",
        "-q",
        "tests/test_docs_routes.py",
    ) in runner.command_list


def test_package_uses_temporary_output_and_inspects_both_artifacts(
    tmp_path: Path,
) -> None:
    runner = CheckRunner()

    check.check_package(root=tmp_path, runner=runner)

    assert runner.command_list[0] == ("uv", "lock", "--check")
    build = runner.command_list[1]
    inspect = runner.command_list[2]
    assert build[:3] == ("uv", "build", "--out-dir")
    assert Path(build[3]).parent != tmp_path
    assert inspect[1] == "scripts/inspect_release_artifact.py"
    assert inspect[-2].endswith(".whl")
    assert inspect[-1].endswith(".tar.gz")
    assert not Path(build[3]).exists()


def test_package_smoke_runs_outside_checkout_without_pythonpath(tmp_path: Path) -> None:
    runner = CheckRunner()

    check.check_package(root=tmp_path, runner=runner, smoke=True)

    smoke_calls = [
        item
        for item in runner.commands
        if "-c" in item[0] or item[0][-1] in {"--help", "--version"}
    ]
    assert len(smoke_calls) == 3
    for _, cwd, environment in smoke_calls:
        assert cwd != tmp_path
        assert environment is not None
        assert "PYTHONPATH" not in environment


def test_read_only_profiles_do_not_modify_repository(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    marker.write_text("unchanged")
    before = marker.read_bytes()

    check.check_python(root=tmp_path, runner=CheckRunner())
    check.check_rust(root=tmp_path, runner=CheckRunner())

    assert marker.read_bytes() == before
