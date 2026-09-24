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
    from scripts.agent.bootstrap import CODEGRAPH_VERSION, RTK_VERSION
    from scripts.agent.codegraph_mcp import LauncherError, resolve_codegraph
    from scripts.agent.rtk_hook import RtkError, recall_status, resolve_rtk
except ModuleNotFoundError:  # Direct script execution puts scripts/agent on sys.path.
    from bootstrap import CODEGRAPH_VERSION, RTK_VERSION
    from codegraph_mcp import LauncherError, resolve_codegraph
    from rtk_hook import RtkError, recall_status, resolve_rtk

ROOT = Path(__file__).resolve().parents[2]
Runner = Callable[[Sequence[str], Path], str]
Which = Callable[[str], str | None]

EXPECTED_ROLES = {
    "explorer": ("gpt-5.6-luna", "medium"),
    "implementer": ("gpt-5.6-sol", "medium"),
    "reviewer": ("gpt-5.6-sol", "high"),
    "architect": ("gpt-6-astra", "medium"),
}
EXPECTED_ROOT_ROUTING = {
    "model": "gpt-5.6-sol",
    "model_reasoning_effort": "medium",
}
EXPECTED_AGENT_DEFAULTS = {
    "enabled": True,
    "max_concurrent_threads_per_session": 4,
    "default_subagent_model": "gpt-5.6-sol",
    "default_subagent_reasoning_effort": "medium",
}
EXPECTED_MCP = {
    "command": "uv",
    "args": ["run", "--no-project", "python", "scripts/agent/codegraph_mcp.py"],
}
EXPECTED_RTK_HOOKS = {
    "PreToolUse": [
        {
            "matcher": "^Bash$",
            "hooks": [
                {
                    "type": "command",
                    "command": 'uv run --no-project python -c "import runpy, '
                    "subprocess; root = subprocess.check_output(['git', "
                    "'rev-parse', '--show-toplevel'], text=True).strip(); "
                    "runpy.run_path(root + '/scripts/agent/rtk_hook.py', "
                    "run_name='__main__')\"",
                    "timeout": 5,
                    "statusMessage": "Applying RTK output compression",
                }
            ],
        }
    ]
}


def _required_uv_version(root: Path) -> str:
    requirement = tomllib.loads((root / "uv.toml").read_text()).get("required-version")
    if not isinstance(requirement, str) or not requirement.startswith("=="):
        raise DoctorError("uv.toml must pin one exact required-version")
    return requirement.removeprefix("==")


def _uv_version_health(output: str, root: Path) -> str:
    required = _required_uv_version(root)
    parts = output.split()
    found = parts[1] if len(parts) >= 2 and parts[0] == "uv" else "<unknown>"
    if found != required:
        raise DoctorError(f"required {required}, found {found}")
    return f"{found} (required {required})"


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


def _check_routing(root: Path) -> list[Diagnostic]:
    config = tomllib.loads((root / ".codex/config.toml").read_text())
    results = []
    actual_root = {key: config.get(key) for key in EXPECTED_ROOT_ROUTING}
    results.append(
        Diagnostic(
            actual_root == EXPECTED_ROOT_ROUTING,
            "coordinator routing",
            "gpt-5.6-sol / medium"
            if actual_root == EXPECTED_ROOT_ROUTING
            else f"expected {EXPECTED_ROOT_ROUTING}, found {actual_root}",
        )
    )
    configured = config.get("agents", {})
    actual_defaults = {key: configured.get(key) for key in EXPECTED_AGENT_DEFAULTS}
    results.append(
        Diagnostic(
            actual_defaults == EXPECTED_AGENT_DEFAULTS,
            "agent defaults",
            "enabled, gpt-5.6-sol / medium, max 4"
            if actual_defaults == EXPECTED_AGENT_DEFAULTS
            else f"expected {EXPECTED_AGENT_DEFAULTS}, found {actual_defaults}",
        )
    )
    metadata_keys = set(EXPECTED_AGENT_DEFAULTS)
    registered_roles = set(configured) - metadata_keys
    expected_roles = set(EXPECTED_ROLES)
    results.append(
        Diagnostic(
            registered_roles == expected_roles,
            "registered roles",
            ", ".join(sorted(expected_roles))
            if registered_roles == expected_roles
            else f"expected {sorted(expected_roles)}, found {sorted(registered_roles)}",
        )
    )
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
    actual_mcp = config.get("mcp_servers", {}).get("codegraph")
    results.append(
        Diagnostic(
            actual_mcp == EXPECTED_MCP,
            "CodeGraph MCP launcher",
            "uv run --no-project"
            if actual_mcp == EXPECTED_MCP
            else f"expected {EXPECTED_MCP}, found {actual_mcp}",
        )
    )
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


def _rtk_hook_health(root: Path) -> str:
    try:
        configuration = json.loads((root / ".codex/hooks.json").read_text())
    except json.JSONDecodeError as error:
        raise DoctorError("invalid .codex/hooks.json") from error
    if not isinstance(configuration, dict):
        raise DoctorError("invalid .codex/hooks.json")
    actual = configuration.get("hooks")
    if actual != EXPECTED_RTK_HOOKS:
        raise DoctorError(f"expected {EXPECTED_RTK_HOOKS}, found {actual}")
    return "PreToolUse -> repository adapter"


def _rtk_version_health(output: str) -> str:
    parts = output.split()
    found = parts[1] if len(parts) >= 2 and parts[0].lower() == "rtk" else "<unknown>"
    if found != RTK_VERSION:
        raise DoctorError(f"expected {RTK_VERSION}, found {found}")
    return found


def _rtk_identity_health(output: str) -> str:
    if "high-performance CLI proxy" not in output:
        raise DoctorError("resolved executable is not Rust Token Killer")
    return "Rust Token Killer"


def _rtk_recall_health(output: str) -> str:
    mode, effective = recall_status(output)
    if not effective:
        raise DoctorError("recovery disabled by RTK_RECALL=0 or RTK_TEE=0")
    if mode != "sqlite":
        raise DoctorError(f"expected sqlite recall mode, found {mode}")
    return "sqlite"


def diagnose(
    *,
    root: Path = ROOT,
    cwd: Path | None = None,
    home: Path | None = None,
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
    for tool in ("uv", "git", "cargo", "gh", "codex"):
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
        "uv version",
        ("uv", "--version"),
        lambda output: _uv_version_health(output, root),
    )
    command_check(
        "Codex config",
        ("codex", "doctor", "--summary", "--json"),
        _codex_config_health,
    )
    try:
        results.extend(_check_routing(root))
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        results.append(Diagnostic(False, "configured roles", str(error)))
    try:
        results.append(Diagnostic(True, "RTK Codex hook", _rtk_hook_health(root)))
    except (DoctorError, OSError, TypeError) as error:
        results.append(Diagnostic(False, "RTK Codex hook", str(error)))
    command_check(
        "MCP configuration",
        ("codex", "mcp", "list"),
        lambda output: (
            "CodeGraph enabled"
            if "codegraph" in output and "enabled" in output
            else _raise("CodeGraph MCP is not enabled")
        ),
    )
    try:
        codegraph = resolve_codegraph(which=which, home=home)
    except LauncherError as error:
        results.append(Diagnostic(False, "codegraph", str(error)))
    else:
        results.append(Diagnostic(True, "codegraph", codegraph))
        try:
            version = runner((codegraph, "--version"), root).strip().removeprefix("v")
            if version != CODEGRAPH_VERSION:
                raise DoctorError(f"expected {CODEGRAPH_VERSION}, found {version}")
        except DoctorError as error:
            results.append(Diagnostic(False, "CodeGraph version", str(error)))
        else:
            results.append(Diagnostic(True, "CodeGraph version", version))
        try:
            health = _codegraph_health(runner((codegraph, "status", "--json"), root))
        except DoctorError as error:
            results.append(Diagnostic(False, "CodeGraph index", str(error)))
        else:
            results.append(Diagnostic(True, "CodeGraph index", health))
    try:
        rtk = resolve_rtk(which=which, home=home)
    except RtkError as error:
        results.append(Diagnostic(False, "RTK", str(error)))
    else:
        results.append(Diagnostic(True, "RTK", rtk))
        for name, command, validate in (
            ("RTK version", (rtk, "--version"), _rtk_version_health),
            ("RTK identity", (rtk, "--help"), _rtk_identity_health),
            ("RTK recall", (rtk, "config", "recall"), _rtk_recall_health),
        ):
            try:
                detail = validate(runner(command, root))
            except (DoctorError, OSError) as error:
                results.append(Diagnostic(False, name, str(error)))
            else:
                results.append(Diagnostic(True, name, detail))
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
                "    uv run --locked python scripts/agent/bootstrap.py",
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
