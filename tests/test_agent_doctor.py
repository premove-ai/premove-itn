from __future__ import annotations

import json
from pathlib import Path

from scripts.agent import bootstrap, doctor


def _repository(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'premove-itn'\n")
    (tmp_path / "AGENTS.md").write_text("# Agents\n")
    agents = tmp_path / ".codex" / "agents"
    agents.mkdir(parents=True)
    registrations = []
    for name, (model, effort) in doctor.EXPECTED_ROLES.items():
        registrations.extend(
            (
                f"[agents.{name}]",
                f'config_file = "./agents/{name}.toml"',
                "",
            )
        )
        (agents / f"{name}.toml").write_text(
            f'model = "{model}"\nmodel_reasoning_effort = "{effort}"\n'
            'developer_instructions = "test"\n'
        )
    (tmp_path / ".codex" / "config.toml").write_text("\n".join(registrations))
    return tmp_path


def _healthy_status(**index_overrides: object) -> str:
    index = {
        "state": "complete",
        "builtWithVersion": bootstrap.CODEGRAPH_VERSION,
        "pendingRefs": 0,
        "reindexRecommended": False,
    }
    index.update(index_overrides)
    return json.dumps(
        {
            "initialized": True,
            "fileCount": 64,
            "nodeCount": 1233,
            "index": index,
        }
    )


class DoctorRunner:
    def __init__(
        self, *, fail: tuple[str, ...] | None = None, status: str | None = None
    ):
        self.fail = fail
        self.status = status or _healthy_status()
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: object, _root: Path) -> str:
        normalized = tuple(command)  # type: ignore[arg-type]
        self.commands.append(normalized)
        if normalized == self.fail:
            raise doctor.DoctorError("diagnostic failed")
        if normalized == ("codex", "doctor", "--summary", "--json"):
            return json.dumps(
                {
                    "checks": {
                        "config.load": {"status": "ok", "summary": "config loaded"}
                    }
                }
            )
        if normalized == ("codex", "mcp", "list"):
            return "codegraph python scripts/agent/codegraph_mcp.py enabled"
        if normalized == ("codegraph", "--version"):
            return f"{bootstrap.CODEGRAPH_VERSION}\n"
        if normalized == ("codegraph", "status", "--json"):
            return self.status
        return ""


def _which(tool: str) -> str:
    return f"/bin/{tool}"


def test_healthy_harness_reports_every_role_and_index(tmp_path: Path) -> None:
    root = _repository(tmp_path)

    results = doctor.diagnose(root=root, cwd=root, which=_which, runner=DoctorRunner())
    report = doctor.render(results)

    assert all(result.ok for result in results)
    assert "PASS architect: gpt-6-astra / medium" in report
    assert "PASS CodeGraph 1.6.0" not in report
    assert "PASS CodeGraph version: 1.6.0" in report
    assert report.endswith("Harness healthy.")


def test_missing_required_binary_is_reported(tmp_path: Path) -> None:
    root = _repository(tmp_path)

    results = doctor.diagnose(
        root=root,
        cwd=root,
        which=lambda tool: None if tool == "cargo" else _which(tool),
        runner=DoctorRunner(),
    )

    assert doctor.Diagnostic(False, "cargo", "not found on PATH") in results


def test_bad_github_auth_is_reported(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    runner = DoctorRunner(fail=("gh", "auth", "status"))

    results = doctor.diagnose(root=root, cwd=root, which=_which, runner=runner)

    assert (
        doctor.Diagnostic(False, "GitHub authentication", "diagnostic failed")
        in results
    )


def test_codex_config_failure_is_reported(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    runner = DoctorRunner(fail=("codex", "doctor", "--summary", "--json"))

    results = doctor.diagnose(root=root, cwd=root, which=_which, runner=runner)

    assert doctor.Diagnostic(False, "Codex config", "diagnostic failed") in results


def test_codegraph_bad_version_and_unhealthy_index_are_reported(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    runner = DoctorRunner(status=_healthy_status(pendingRefs=4))

    def mismatched(command: object, command_root: Path) -> str:
        normalized = tuple(command)  # type: ignore[arg-type]
        if normalized == ("codegraph", "--version"):
            return "1.7.0\n"
        return runner(normalized, command_root)

    results = doctor.diagnose(root=root, cwd=root, which=_which, runner=mismatched)
    report = doctor.render(results)

    assert "FAIL CodeGraph version: expected 1.6.0, found 1.7.0" in report
    assert "FAIL CodeGraph index: pending references = 4" in report
    assert "uv run python scripts/agent/bootstrap.py" in report


def test_diagnosis_does_not_modify_repository(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    doctor.diagnose(root=root, cwd=root, which=_which, runner=DoctorRunner())

    after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before
