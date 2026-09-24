"""Diagnose the repository-local coding-agent harness without repairing it."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.agent.bootstrap import CODEGRAPH_VERSION
except ModuleNotFoundError:  # Direct script execution puts scripts/agent on sys.path.
    from bootstrap import CODEGRAPH_VERSION

ROOT = Path(__file__).resolve().parents[2]
Runner = Callable[[Sequence[str], Path], str]
Which = Callable[[str], str | None]

EXPECTED_ROLES = {
    "explorer": ("gpt-5.6-luna", "medium"),
    "implementer": ("gpt-5.6-sol", "medium"),
    "reviewer": ("gpt-5.6-sol", "high"),
    "architect": ("gpt-6-astra", "medium"),
}


class DoctorError(RuntimeError):
    """A harness diagnostic failed."""


@dataclass(frozen=True)
class Diagnostic:
    ok: bool
    name: str
    detail: str


def _run(command: Sequence[str], root: Path) -> str:
    try:
        result = subprocess.run(
            command, cwd=root, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as error:
        if (
            tuple(command) == ("codex", "doctor", "--summary", "--json")
            and error.stdout
        ):
            return error.stdout
        detail = error.stderr or str(error)
        raise DoctorError(detail.strip()) from error
    except OSError as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise DoctorError(detail.strip()) from error
    return result.stdout


def _check_roles(root: Path) -> list[Diagnostic]:
    config = tomllib.loads((root / ".codex/config.toml").read_text())
    configured = config.get("agents", {})
    results = []
    for name, expected in EXPECTED_ROLES.items():
        registration = configured.get(name)
        if not isinstance(registration, dict) or "config_file" not in registration:
            results.append(Diagnostic(False, name, "role is not registered"))
            continue
        role_path = root / ".codex" / registration["config_file"]
        role = tomllib.loads(role_path.read_text())
        actual = (role.get("model"), role.get("model_reasoning_effort"))
        if actual != expected:
            results.append(
                Diagnostic(False, name, f"expected {expected[0]} / {expected[1]}")
            )
        else:
            results.append(Diagnostic(True, name, f"{actual[0]} / {actual[1]}"))
    return results


def _codegraph_health(status_text: str) -> str:
    try:
        status = json.loads(status_text)
    except json.JSONDecodeError as error:
        raise DoctorError("invalid status JSON") from error
    index = status.get("index", {})
    if status.get("initialized") is not True:
        raise DoctorError("index is not initialized")
    if index.get("state") != "complete":
        raise DoctorError(f"index state = {index.get('state', '<missing>')}")
    if index.get("builtWithVersion") != CODEGRAPH_VERSION:
        raise DoctorError(
            f"index version = {index.get('builtWithVersion', '<missing>')}"
        )
    if index.get("pendingRefs") != 0:
        raise DoctorError(f"pending references = {index.get('pendingRefs')}")
    if index.get("reindexRecommended") is not False:
        raise DoctorError("reindex is recommended")
    return f"{status.get('fileCount', 0)} files, {status.get('nodeCount', 0)} nodes"


def _codex_config_health(report_text: str) -> str:
    try:
        report = json.loads(report_text)
    except json.JSONDecodeError as error:
        raise DoctorError("invalid doctor JSON") from error
    config = report.get("checks", {}).get("config.load", {})
    if config.get("status") != "ok":
        raise DoctorError(config.get("summary", "config did not load"))
    return "loaded"


def diagnose(
    *,
    root: Path = ROOT,
    cwd: Path | None = None,
    which: Which = shutil.which,
    runner: Runner = _run,
) -> tuple[Diagnostic, ...]:
    """Run all read-only diagnostics and retain every failure."""
    results: list[Diagnostic] = []

    repository_ok = (cwd or Path.cwd()).resolve() == root.resolve() and all(
        (root / marker).is_file() for marker in ("pyproject.toml", "AGENTS.md")
    )
    results.append(
        Diagnostic(
            repository_ok, "repository", str(root) if repository_ok else "wrong root"
        )
    )
    results.append(Diagnostic(True, "Python", platform.python_version()))

    binaries: dict[str, str] = {}
    for tool in ("uv", "git", "cargo", "gh", "codex", "codegraph"):
        executable = which(tool)
        if executable:
            binaries[tool] = executable
            results.append(Diagnostic(True, tool, executable))
        else:
            results.append(Diagnostic(False, tool, "not found on PATH"))

    def command_check(
        name: str, command: Sequence[str], validate: Callable[[str], str]
    ) -> None:
        if command[0] not in binaries:
            return
        try:
            detail = validate(runner(command, root))
        except (
            DoctorError,
            OSError,
            KeyError,
            TypeError,
            tomllib.TOMLDecodeError,
        ) as error:
            results.append(Diagnostic(False, name, str(error)))
        else:
            results.append(Diagnostic(True, name, detail))

    command_check(
        "GitHub authentication", ("gh", "auth", "status"), lambda _: "authenticated"
    )
    command_check(
        "Codex config",
        ("codex", "doctor", "--summary", "--json"),
        _codex_config_health,
    )
    try:
        results.extend(_check_roles(root))
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        results.append(Diagnostic(False, "configured roles", str(error)))
    command_check(
        "MCP configuration",
        ("codex", "mcp", "list"),
        lambda output: (
            "CodeGraph enabled"
            if "codegraph" in output and "enabled" in output
            else _raise("CodeGraph MCP is not enabled")
        ),
    )
    command_check(
        "CodeGraph version",
        ("codegraph", "--version"),
        lambda output: (
            CODEGRAPH_VERSION
            if output.strip().removeprefix("v") == CODEGRAPH_VERSION
            else _raise(f"expected {CODEGRAPH_VERSION}, found {output.strip()}")
        ),
    )
    command_check(
        "CodeGraph index",
        ("codegraph", "status", "--json"),
        _codegraph_health,
    )
    return tuple(results)


def _raise(message: str) -> str:
    raise DoctorError(message)


def render(results: Sequence[Diagnostic]) -> str:
    lines = ["Premove ITN agent harness", ""]
    for result in results:
        lines.append(
            f"{'PASS' if result.ok else 'FAIL'} {result.name}: {result.detail}"
        )
    failures = [result for result in results if not result.ok]
    if failures:
        lines.extend(
            (
                "",
                "Harness unhealthy.",
                "",
                "Repair:",
                "    uv run python scripts/agent/bootstrap.py",
            )
        )
    else:
        lines.extend(("", "Harness healthy."))
    return "\n".join(lines)


def main() -> int:
    results = diagnose()
    print(render(results))
    return 1 if any(not result.ok for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
