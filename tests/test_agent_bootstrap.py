from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.agent import bootstrap


def _repository(tmp_path: Path, *, indexed: bool = True) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'premove-itn'\n")
    (tmp_path / "AGENTS.md").write_text("# Agents\n")
    if indexed:
        (tmp_path / ".codegraph").mkdir()
    return tmp_path


def _healthy_status() -> str:
    return json.dumps(
        {
            "initialized": True,
            "fileCount": 59,
            "nodeCount": 1168,
            "edgeCount": 3162,
            "index": {
                "state": "complete",
                "builtWithVersion": bootstrap.CODEGRAPH_VERSION,
                "pendingRefs": 0,
                "reindexRecommended": False,
            },
        }
    )


class RecordingRunner:
    def __init__(
        self,
        version: str = bootstrap.CODEGRAPH_VERSION,
        *,
        initialized: bool = True,
    ) -> None:
        self.version = version
        self.initialized = initialized
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: object, root: Path) -> str:
        normalized = tuple(command)  # type: ignore[arg-type]
        self.commands.append(normalized)
        if normalized[-1] == "--version":
            return f"{self.version}\n"
        if normalized[-2:] == ("init", "--yes"):
            self.initialized = True
        if normalized[-2:] == ("status", "--json"):
            status = json.loads(_healthy_status())
            status["initialized"] = self.initialized
            return json.dumps(status)
        return ""


def test_correct_installed_version_does_not_install(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    runner = RecordingRunner()

    def unexpected_install(_root: Path) -> str:
        raise AssertionError("installer must not run")

    bootstrap.bootstrap(
        root=root,
        cwd=root,
        which=lambda tool: "/bin/codegraph" if tool == "codegraph" else None,
        runner=runner,
        installer=unexpected_install,
    )

    assert ("/bin/codegraph", "init", "--yes") not in runner.commands


def test_missing_codegraph_uses_pinned_installer(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    runner = RecordingRunner()
    installed = tmp_path / "bin" / "codegraph"
    installed.parent.mkdir()

    def install(_root: Path) -> str:
        installed.write_text("#!/bin/sh\n")
        installed.chmod(0o755)
        return str(installed)

    bootstrap.bootstrap(
        root=root,
        cwd=root,
        which=lambda _tool: None,
        runner=runner,
        installer=install,
    )

    assert (str(installed), "--version") in runner.commands


def test_successful_installer_without_binary_fails(tmp_path: Path) -> None:
    root = _repository(tmp_path)

    with pytest.raises(bootstrap.BootstrapError, match="could not be resolved"):
        bootstrap.bootstrap(
            root=root,
            cwd=root,
            which=lambda _tool: None,
            runner=RecordingRunner(),
            installer=lambda _root: str(tmp_path / "missing" / "codegraph"),
        )


def test_wrong_version_fails_with_exact_install_command(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    runner = RecordingRunner(version="1.7.2")

    with pytest.raises(bootstrap.BootstrapError, match="Required.*1.6.0") as error:
        bootstrap.bootstrap(
            root=root,
            cwd=root,
            which=lambda _tool: "/bin/codegraph",
            runner=runner,
        )

    assert "uv run python scripts/agent/bootstrap.py" in str(error.value)


def test_missing_index_initializes_once(tmp_path: Path) -> None:
    root = _repository(tmp_path, indexed=False)
    runner = RecordingRunner(initialized=False)

    bootstrap.bootstrap(
        root=root,
        cwd=root,
        which=lambda _tool: "/bin/codegraph",
        runner=runner,
    )

    assert runner.commands.count(("/bin/codegraph", "init", "--yes")) == 1


def test_partial_index_directory_is_reinitialized(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    runner = RecordingRunner(initialized=False)

    bootstrap.bootstrap(
        root=root,
        cwd=root,
        which=lambda _tool: "/bin/codegraph",
        runner=runner,
    )

    assert runner.commands.count(("/bin/codegraph", "init", "--yes")) == 1


def test_existing_index_is_not_initialized(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    runner = RecordingRunner()

    bootstrap.bootstrap(
        root=root,
        cwd=root,
        which=lambda _tool: "/bin/codegraph",
        runner=runner,
    )

    assert ("/bin/codegraph", "init", "--yes") not in runner.commands


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("builtWithVersion", "1.5.0"),
        ("pendingRefs", 1),
        ("reindexRecommended", True),
    ),
)
def test_incomplete_index_is_unhealthy(
    tmp_path: Path, field: str, value: object
) -> None:
    root = _repository(tmp_path)
    runner = RecordingRunner()

    def unhealthy_runner(command: object, command_root: Path) -> str:
        normalized = tuple(command)  # type: ignore[arg-type]
        if normalized[-2:] == ("status", "--json"):
            status = json.loads(_healthy_status())
            status["index"][field] = value
            return json.dumps(status)
        return runner(command, command_root)

    with pytest.raises(bootstrap.BootstrapError, match="not healthy"):
        bootstrap.bootstrap(
            root=root,
            cwd=root,
            which=lambda _tool: "/bin/codegraph",
            runner=unhealthy_runner,
        )


def test_status_failure_produces_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> None:
        raise bootstrap.BootstrapError("status failed")

    monkeypatch.setattr(bootstrap, "bootstrap", fail)
    assert bootstrap.main() == 1


def test_repeated_bootstrap_does_not_modify_repository(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    runner = RecordingRunner()
    files_before = sorted(path.relative_to(root) for path in root.rglob("*"))

    for _ in range(2):
        bootstrap.bootstrap(
            root=root,
            cwd=root,
            which=lambda _tool: "/bin/codegraph",
            runner=runner,
        )

    assert sorted(path.relative_to(root) for path in root.rglob("*")) == files_before
    assert ("/bin/codegraph", "init", "--yes") not in runner.commands
