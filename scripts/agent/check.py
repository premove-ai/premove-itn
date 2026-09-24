"""Run deterministic repository validation profiles."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

try:
    from scripts.agent.rtk_hook import RtkError, resolve_rtk
except ModuleNotFoundError:  # Direct script execution from scripts/agent.
    from rtk_hook import RtkError, resolve_rtk

ROOT = Path(__file__).resolve().parents[2]
RECALL_HINT = re.compile(
    r"^\[full output: rtk recall ([0-9a-f]{12,64})\]$", re.MULTILINE
)
CaptureRunner = Callable[[Sequence[str], Path, dict[str, str] | None], str]
ValidationRunner = Callable[[Sequence[str], Path, dict[str, str] | None], None]


class CheckError(RuntimeError):
    """A validation command or invariant failed."""


def _print_failure(output: str | None, error: str | None) -> None:
    if output:
        print(output, end="")
    if error:
        print(error, end="", file=sys.stderr)


def _run_capture(
    command: Sequence[str], root: Path, environment: dict[str, str] | None = None
) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        _print_failure(error.stdout, error.stderr)
        raise CheckError(f"Command failed: {' '.join(command)}") from error
    except OSError as error:
        raise CheckError(f"Command failed: {' '.join(command)}") from error
    return result.stdout


def _replay_script(directory: Path) -> Path:
    if os.name == "nt":
        script = directory / "replay.cmd"
        script.write_text('@type "%~1"\r\n@exit /b 1\r\n', encoding="utf-8")
        return script
    script = directory / "replay"
    script.write_text('#!/bin/sh\ncat "$1"\nexit 1\n', encoding="utf-8")
    script.chmod(0o700)
    return script


def _print_compact_failure(
    command: Sequence[str],
    output: str | None,
    error: str | None,
    root: Path,
    environment: dict[str, str] | None,
) -> bool:
    raw = f"{output or ''}{error or ''}"
    if not raw or RECALL_HINT.search(raw):
        return False
    try:
        rtk = resolve_rtk()
    except RtkError:
        return False
    is_test = "pytest" in command or (
        Path(command[0]).name == "cargo" and command[1:2] == ("test",)
    )
    try:
        with tempfile.TemporaryDirectory(prefix="premove-failure-") as temporary:
            directory = Path(temporary)
            failure_log = directory / "output.log"
            failure_log.write_text(raw, encoding="utf-8")
            replay = _replay_script(directory)
            replay_command = replay.name if os.name == "nt" else f"./{replay.name}"
            result = subprocess.run(
                (
                    rtk,
                    "test" if is_test else "err",
                    replay_command,
                    failure_log.name,
                ),
                cwd=directory,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            compact = f"{result.stdout}{result.stderr}"
            handles = RECALL_HINT.findall(compact)
            if len(handles) != 1:
                return False
        recalled = subprocess.run(
            (rtk, "recall", handles[0], "--full"),
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    if recalled.returncode or recalled.stderr or recalled.stdout != raw:
        return False
    _print_failure(result.stdout, result.stderr)
    return True


def _command_label(command: Sequence[str]) -> str:
    executable = Path(command[0]).name
    if executable == "uv" and "ruff" in command:
        action = command[command.index("ruff") + 1]
        return f"ruff {action}"
    if executable == "uv" and "pytest" in command:
        return "pytest"
    if executable == "uv" and "python" in command:
        script = command[command.index("python") + 1]
        return Path(script).stem.removeprefix("check_").replace("_", " ")
    if executable == "uv" and len(command) > 1:
        return f"uv {command[1]}"
    if executable == "cargo" and len(command) > 1:
        return f"cargo {command[1]}"
    if executable == "git" and len(command) > 1:
        return f"git {' '.join(command[1:3])}"
    if "-c" in command:
        return f"{executable} smoke"
    if command[-1] in {"--help", "--version"}:
        return f"{executable} {command[-1]}"
    if len(command) > 1 and command[1].endswith(".py"):
        return Path(command[1]).stem.replace("_", " ")
    return executable


def _run_validation(
    command: Sequence[str], root: Path, environment: dict[str, str] | None = None
) -> None:
    try:
        subprocess.run(
            command,
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        if not _print_compact_failure(
            command, error.stdout, error.stderr, root, environment
        ):
            _print_failure(error.stdout, error.stderr)
        raise CheckError(f"Command failed: {' '.join(command)}") from error
    except OSError as error:
        raise CheckError(f"Command failed: {' '.join(command)}") from error
    print(f"PASS {_command_label(command)}", flush=True)


def _commands(
    runner: ValidationRunner, commands: Sequence[Sequence[str]], root: Path
) -> None:
    for command in commands:
        runner(command, root, None)


def check_python(
    *, root: Path = ROOT, runner: ValidationRunner = _run_validation
) -> None:
    """Run the complete Python and repository-policy gate."""
    _commands(
        runner,
        (
            ("uv", "run", "--locked", "ruff", "format", "--check", "."),
            ("uv", "run", "--locked", "ruff", "check", "."),
            ("uv", "run", "--locked", "pytest"),
            ("uv", "run", "--locked", "python", "scripts/check_local_links.py"),
            ("uv", "run", "--locked", "python", "scripts/check_frozen_boundaries.py"),
        ),
        root,
    )


def check_rust(
    *, root: Path = ROOT, runner: ValidationRunner = _run_validation
) -> None:
    """Run Rust formatting and tests."""
    _commands(
        runner,
        (
            ("cargo", "fmt", "--manifest-path", "rust/Cargo.toml", "--", "--check"),
            ("cargo", "test", "--manifest-path", "rust/Cargo.toml"),
        ),
        root,
    )


def _artifacts(directory: Path) -> tuple[Path, Path]:
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise CheckError(
            "Package build must produce exactly one wheel and one source archive."
        )
    return wheels[0], sdists[0]


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_cli(venv: Path) -> Path:
    return venv / ("Scripts/premove-itn.exe" if os.name == "nt" else "bin/premove-itn")


SMOKE_CODE = """\
from pathlib import Path
import premove_itn
from premove_itn import _rust

package_path = Path(premove_itn.__file__).resolve()
if "site-packages" not in package_path.parts:
    raise SystemExit(f"package was not imported from site-packages: {package_path}")
build = _rust.build_info()
if build["profile"] != "release" or build["debug_assertions"] is not False:
    raise SystemExit(f"wheel contains a non-release Rust extension: {build}")
print(f"verified installed package: {package_path}")
print(f"verified release Rust build: {build}")
"""


def check_package(
    *,
    root: Path = ROOT,
    runner: ValidationRunner = _run_validation,
    smoke: bool = False,
) -> None:
    """Build and inspect fresh release artifacts, with an optional clean install."""
    with tempfile.TemporaryDirectory(prefix="premove-package-") as temporary:
        temporary_root = Path(temporary)
        artifacts = temporary_root / "dist"
        runner(("uv", "lock", "--check"), root, None)
        runner(("uv", "build", "--out-dir", str(artifacts)), root, None)
        wheel, sdist = _artifacts(artifacts)
        runner(
            (
                sys.executable,
                "scripts/inspect_release_artifact.py",
                str(wheel),
                str(sdist),
            ),
            root,
            None,
        )
        if not smoke:
            return

        venv = temporary_root / "venv"
        runner(("uv", "venv", "--python", "3.11", str(venv)), root, None)
        python = _venv_python(venv)
        runner(
            ("uv", "pip", "install", "--python", str(python), str(wheel)),
            root,
            None,
        )
        smoke_root = temporary_root / "smoke"
        smoke_root.mkdir()
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        runner((str(python), "-c", SMOKE_CODE), smoke_root, environment)
        cli = _venv_cli(venv)
        runner((str(cli), "--help"), smoke_root, environment)
        runner((str(cli), "--version"), smoke_root, environment)


def _changed_files(root: Path, runner: CaptureRunner) -> tuple[str, ...]:
    tracked = runner(("git", "diff", "--name-only", "HEAD"), root, None)
    untracked = runner(
        ("git", "ls-files", "--others", "--exclude-standard"), root, None
    )
    return tuple(sorted(set(tracked.splitlines()) | set(untracked.splitlines())))


def check_focused(
    pytest_targets: Sequence[str],
    *,
    root: Path = ROOT,
    runner: ValidationRunner = _run_validation,
    capture_runner: CaptureRunner = _run_capture,
) -> None:
    """Run mechanical checks plus caller-selected semantic Python tests."""
    changed = _changed_files(root, capture_runner)
    existing = [path for path in changed if (root / path).is_file()]
    python_files = [path for path in existing if path.endswith(".py")]
    rust_changed = any(path.startswith("rust/") for path in changed)
    docs_changed = any(
        path.endswith((".md", ".mdx"))
        or path == "docs.json"
        or path.startswith(("docs/", "itn/"))
        for path in changed
    )
    package_changed = any(
        path in {"pyproject.toml", "uv.lock", "rust/Cargo.toml", "rust/Cargo.lock"}
        for path in changed
    )

    if python_files:
        runner(
            ("uv", "run", "--locked", "ruff", "format", "--check", *python_files),
            root,
            None,
        )
        runner(("uv", "run", "--locked", "ruff", "check", *python_files), root, None)
    if rust_changed:
        check_rust(root=root, runner=runner)
    if docs_changed:
        runner(
            ("uv", "run", "--locked", "python", "scripts/check_local_links.py"),
            root,
            None,
        )
        runner(
            ("uv", "run", "--locked", "pytest", "-q", "tests/test_docs_routes.py"),
            root,
            None,
        )
    if package_changed:
        check_package(root=root, runner=runner)
    if pytest_targets:
        runner(("uv", "run", "--locked", "pytest", *pytest_targets), root, None)
    elif any(path.startswith("src/") and path.endswith(".py") for path in changed):
        print(
            "WARNING: production Python changed without an explicit semantic pytest "
            "target. Use --pytest <test>.",
            file=sys.stderr,
        )


def check_full(
    *, root: Path = ROOT, runner: ValidationRunner = _run_validation
) -> None:
    """Run the canonical local completion gate."""
    check_python(root=root, runner=runner)
    check_rust(root=root, runner=runner)
    check_package(root=root, runner=runner)
    runner(("git", "diff", "--check"), root, None)
    runner(("git", "diff", "--cached", "--check"), root, None)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="profile", required=True)
    focused = subparsers.add_parser("focused")
    focused.add_argument("--pytest", action="append", default=[], dest="pytest_targets")
    subparsers.add_parser("python")
    subparsers.add_parser("rust")
    package = subparsers.add_parser("package")
    package.add_argument("--smoke", action="store_true")
    subparsers.add_parser("full")
    arguments = parser.parse_args(argv)

    try:
        if arguments.profile == "focused":
            check_focused(arguments.pytest_targets)
        elif arguments.profile == "python":
            check_python()
        elif arguments.profile == "rust":
            check_rust()
        elif arguments.profile == "package":
            check_package(smoke=arguments.smoke)
        else:
            check_full()
    except CheckError as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
