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
            "index": {"state": "complete"},
        }
    )


class RecordingRunner:
    def __init__(self, version: str = bootstrap.CODEGRAPH_VERSION) -> None:
        self.version = version
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: object, root: Path) -> str:
        normalized = tuple(command)  # type: ignore[arg-type]
        self.commands.append(normalized)
        if normalized[-1] == "--version":
            return f"{self.version}\n"
        if normalized[-2:] == ("status", "--json"):
            return _healthy_status()
        return ""


def test_correct_installed_version_does_not_install(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    runner = RecordingRunner()

    bootstrap.bootstrap(
        root=root,
        cwd=root,
        which=lambda tool: "/bin/codegraph" if tool == "codegraph" else None,
        runner=runner,
    )

    assert bootstrap.INSTALL_COMMAND not in runner.commands
    assert ("/bin/codegraph", "init") not in runner.commands


def test_missing_codegraph_uses_pinned_installer(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    runner = RecordingRunner()
    lookups = iter((None, "/bin/npx", "/bin/codegraph"))

    bootstrap.bootstrap(
        root=root,
        cwd=root,
        which=lambda _tool: next(lookups),
        runner=runner,
    )

    assert bootstrap.INSTALL_COMMAND in runner.commands


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

    assert " ".join(bootstrap.INSTALL_COMMAND) in str(error.value)


def test_missing_index_initializes_once(tmp_path: Path) -> None:
    root = _repository(tmp_path, indexed=False)
    runner = RecordingRunner()

    bootstrap.bootstrap(
        root=root,
        cwd=root,
        which=lambda _tool: "/bin/codegraph",
        runner=runner,
    )

    assert runner.commands.count(("/bin/codegraph", "init")) == 1


def test_existing_index_is_not_initialized(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    runner = RecordingRunner()

    bootstrap.bootstrap(
        root=root,
        cwd=root,
        which=lambda _tool: "/bin/codegraph",
        runner=runner,
    )

    assert ("/bin/codegraph", "init") not in runner.commands


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
    assert bootstrap.INSTALL_COMMAND not in runner.commands
    assert ("/bin/codegraph", "init") not in runner.commands
