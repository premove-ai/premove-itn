from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.agent import check


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

    monkeypatch.setattr(check.subprocess, "run", fail)

    with pytest.raises(check.CheckError, match="tool check"):
        check._run(("tool", "check"), tmp_path)

    captured = capsys.readouterr()
    assert "captured output" in captured.out
    assert "captured error" in captured.err


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

    check.check_focused(("tests/test_example.py",), root=tmp_path, runner=runner)

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

    check.check_focused((), root=tmp_path, runner=runner)

    assert "without an explicit semantic pytest target" in capsys.readouterr().err
    assert not any(
        command[:4] == ("uv", "run", "--locked", "pytest")
        for command in runner.command_list
    )


def test_focused_rust_and_docs_use_mechanical_checks(tmp_path: Path) -> None:
    _touch(tmp_path, "rust/src/lib.rs", "README.md")
    runner = CheckRunner(changed=("rust/src/lib.rs", "README.md"))

    check.check_focused((), root=tmp_path, runner=runner)

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
