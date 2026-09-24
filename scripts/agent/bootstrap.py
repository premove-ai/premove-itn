"""Bootstrap the repository-local coding-agent harness."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

CODEGRAPH_VERSION = "1.6.0"
RTK_VERSION = "0.49.0"
ROOT = Path(__file__).resolve().parents[2]
INSTALLER_BASE_URL = (
    f"https://raw.githubusercontent.com/colbymchenry/codegraph/v{CODEGRAPH_VERSION}"
)
Runner = Callable[[Sequence[str], Path], str]
Which = Callable[[str], str | None]
Installer = Callable[[Path], str]
RTK_RELEASE_URL = f"https://github.com/rtk-ai/rtk/releases/download/v{RTK_VERSION}"
RTK_ASSETS = {
    ("Darwin", "arm64"): (
        "rtk-aarch64-apple-darwin.tar.gz",
        "bbbfebabb22686993a80da731aa4d5d35116fb8ae24abb00608efa028e13ae01",
    ),
    ("Darwin", "x86_64"): (
        "rtk-x86_64-apple-darwin.tar.gz",
        "d297388f4a8a786e79abe5f55b80451725bfe8c5835b4736c05d7cff4d68f627",
    ),
    ("Linux", "aarch64"): (
        "rtk-aarch64-unknown-linux-gnu.tar.gz",
        "c8ea4b6560841e73157c134fd4a3293914c6ede42e786ee985cf491fde691ba7",
    ),
    ("Linux", "x86_64"): (
        "rtk-x86_64-unknown-linux-musl.tar.gz",
        "7278231dfd7e6a730a4ab7f847b195bcf02289c2d57622b0dab75a6411100c8f",
    ),
    ("Windows", "AMD64"): (
        "rtk-x86_64-pc-windows-msvc.zip",
        "cb971046598f0e8bd51f6c27780fcdd2c39a4c459a811bd95b0d77ba8c0d7c9f",
    ),
}

try:
    from scripts.agent.codegraph_mcp import standalone_codegraph_path
    from scripts.agent.rtk_hook import recall_status, standalone_rtk_path
except ModuleNotFoundError:  # Direct script execution puts scripts/agent on sys.path.
    from codegraph_mcp import standalone_codegraph_path
    from rtk_hook import recall_status, standalone_rtk_path


class BootstrapError(RuntimeError):
    """A deterministic agent bootstrap failure."""


def _run(command: Sequence[str], root: Path) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise BootstrapError(
            f"Command failed: {' '.join(command)}\n{detail.strip()}"
        ) from error
    return result.stdout


def _require_repository_root(root: Path, cwd: Path) -> None:
    if cwd.resolve() != root.resolve():
        raise BootstrapError(
            "Run this command from the Premove ITN repository root:\n"
            "    uv run --locked python scripts/agent/bootstrap.py"
        )
    if not (root / "pyproject.toml").is_file() or not (root / "AGENTS.md").is_file():
        raise BootstrapError(f"Not a Premove ITN repository root: {root}")


def _install_codegraph(root: Path) -> str:
    """Run CodeGraph's pinned standalone installer and return its executable."""
    windows = platform.system() == "Windows"
    script_name = "install.ps1" if windows else "install.sh"
    url = f"{INSTALLER_BASE_URL}/{script_name}"
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            script = Path(temporary_directory) / script_name
            urllib.request.urlretrieve(url, script)
            environment = os.environ.copy()
            environment["CODEGRAPH_VERSION"] = f"v{CODEGRAPH_VERSION}"
            command = (
                (
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    script,
                )
                if windows
                else ("/bin/sh", script)
            )
            subprocess.run(command, cwd=root, env=environment, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise BootstrapError(
            f"Could not install CodeGraph {CODEGRAPH_VERSION}: {error}"
        ) from error

    destination = standalone_codegraph_path()
    if destination is None:
        raise BootstrapError("LOCALAPPDATA is unavailable for CodeGraph.")
    return str(destination)


def _rtk_asset() -> tuple[str, str]:
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin":
        machine = {"aarch64": "arm64", "amd64": "x86_64"}.get(machine.lower(), machine)
    elif system == "Windows":
        machine = {"amd64": "AMD64", "x86_64": "AMD64"}.get(machine.lower(), machine)
    else:
        machine = {"arm64": "aarch64", "amd64": "x86_64"}.get(machine.lower(), machine)
    key = (system, machine)
    try:
        return RTK_ASSETS[key]
    except KeyError as error:
        raise BootstrapError(
            f"RTK {RTK_VERSION} does not support {key[0]} {key[1]}."
        ) from error


def _install_rtk(_root: Path) -> str:
    """Install the pinned, checksum-verified RTK release asset."""
    asset, expected_sha = _rtk_asset()
    destination = standalone_rtk_path()
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / asset
            urllib.request.urlretrieve(f"{RTK_RELEASE_URL}/{asset}", archive)
            actual_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                raise BootstrapError(
                    f"RTK {RTK_VERSION} checksum mismatch for {asset}."
                )
            executable_name = "rtk.exe" if asset.endswith(".zip") else "rtk"
            if asset.endswith(".zip"):
                with zipfile.ZipFile(archive) as package:
                    member = next(
                        name
                        for name in package.namelist()
                        if Path(name).name == executable_name
                    )
                    executable = package.read(member)
            else:
                with tarfile.open(archive, "r:gz") as package:
                    member = next(
                        item
                        for item in package.getmembers()
                        if Path(item.name).name == executable_name and item.isfile()
                    )
                    extracted = package.extractfile(member)
                    if extracted is None:
                        raise BootstrapError(f"RTK executable is missing from {asset}.")
                    executable = extracted.read()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(executable)
            if os.name != "nt":
                destination.chmod(0o755)
    except BootstrapError:
        raise
    except (OSError, StopIteration, tarfile.TarError, zipfile.BadZipFile) as error:
        raise BootstrapError(f"Could not install RTK {RTK_VERSION}: {error}") from error
    return str(destination)


def _ensure_codegraph(
    root: Path,
    which: Which,
    runner: Runner,
    installer: Installer,
) -> str:
    codegraph = which("codegraph")
    if codegraph is None:
        codegraph = installer(root)
        installed_binary = Path(codegraph)
        if not installed_binary.is_file() or (
            platform.system() != "Windows" and not os.access(installed_binary, os.X_OK)
        ):
            raise BootstrapError(
                "CodeGraph installer completed, but its executable could not be "
                f"resolved at {codegraph}."
            )

    found_version = runner((codegraph, "--version"), root).strip().removeprefix("v")
    if found_version != CODEGRAPH_VERSION:
        raise BootstrapError(
            "CodeGraph version mismatch.\n\n"
            f"Required by this repository: {CODEGRAPH_VERSION}\n"
            f"Found: {found_version or '<unknown>'}\n\n"
            "Remove or replace the existing CodeGraph installation, then rerun:\n"
            "    uv run --locked python scripts/agent/bootstrap.py"
        )
    print(f"✓ CodeGraph {CODEGRAPH_VERSION}")
    return codegraph


def _ensure_rtk(
    root: Path,
    which: Which,
    runner: Runner,
    installer: Installer,
) -> str:
    rtk = which("rtk")
    if rtk is None:
        rtk = installer(root)
        installed_binary = Path(rtk)
        if not installed_binary.is_file() or (
            os.name != "nt" and not os.access(installed_binary, os.X_OK)
        ):
            raise BootstrapError(
                "RTK installer completed, but its executable could not be "
                f"resolved at {rtk}."
            )
    version_output = runner((rtk, "--version"), root).strip().split()
    found_version = (
        version_output[1]
        if len(version_output) >= 2 and version_output[0].lower() == "rtk"
        else "<unknown>"
    )
    identity = runner((rtk, "--help"), root)
    if "high-performance CLI proxy" not in identity:
        raise BootstrapError(
            "The resolved rtk executable is not Rust Token Killer. Remove the "
            "name-collision binary and rerun bootstrap."
        )
    if found_version != RTK_VERSION:
        raise BootstrapError(
            "RTK version mismatch.\n\n"
            f"Required by this repository: {RTK_VERSION}\n"
            f"Found: {found_version}\n\n"
            "Remove or replace the existing RTK installation, then rerun:\n"
            "    uv run --locked python scripts/agent/bootstrap.py"
        )
    print(f"✓ RTK {RTK_VERSION}")
    return rtk


def _ensure_rtk_recall(root: Path, rtk: str, runner: Runner) -> None:
    recall = runner((rtk, "config", "recall"), root)
    mode, effective = recall_status(recall)
    if not effective:
        raise BootstrapError(
            "RTK recovery is disabled by RTK_RECALL=0 or RTK_TEE=0.\n"
            "Remove the override before using the Premove agent harness."
        )
    if mode != "sqlite":
        raise BootstrapError(
            "RTK SQLite recall is required for the Premove agent harness.\n\n"
            f"Current mode: {mode}\n\n"
            "Enable it explicitly:\n"
            "    rtk config recall sqlite"
        )
    print("✓ RTK recall: sqlite")


def _read_status(root: Path, codegraph: str, runner: Runner) -> dict[str, object]:
    try:
        status = json.loads(runner((codegraph, "status", "--json"), root))
    except json.JSONDecodeError as error:
        raise BootstrapError(
            "CodeGraph returned invalid health status JSON."
        ) from error
    if not isinstance(status, dict):
        raise BootstrapError("CodeGraph returned invalid health status JSON.")
    return status


def _ensure_index(root: Path, codegraph: str, runner: Runner) -> None:
    status = _read_status(root, codegraph, runner)
    if status.get("initialized") is not True:
        runner((codegraph, "init", "--yes"), root)
        print("+ initialized CodeGraph")
        status = _read_status(root, codegraph, runner)
    else:
        print("✓ CodeGraph index exists")

    counts = (
        status.get("fileCount", 0),
        status.get("nodeCount", 0),
        status.get("edgeCount", 0),
    )
    if (
        status.get("initialized") is not True
        or status.get("index", {}).get("state") != "complete"
        or status.get("index", {}).get("builtWithVersion") != CODEGRAPH_VERSION
        or status.get("index", {}).get("pendingRefs") != 0
        or status.get("index", {}).get("reindexRecommended") is not False
        or not all(isinstance(count, int) and count > 0 for count in counts)
    ):
        raise BootstrapError("CodeGraph index is not healthy.")
    print(f"✓ {counts[0]} indexed files")
    print(f"✓ {counts[1]} nodes")
    print(f"✓ {counts[2]} edges")


def bootstrap(
    *,
    root: Path = ROOT,
    cwd: Path | None = None,
    which: Which = shutil.which,
    runner: Runner = _run,
    installer: Installer = _install_codegraph,
    rtk_installer: Installer = _install_rtk,
) -> None:
    """Install, initialize, and verify the pinned agent harness."""
    print("Premove ITN agent bootstrap\n")
    _require_repository_root(root, cwd or Path.cwd())
    print("✓ repository root")
    codegraph = _ensure_codegraph(root, which, runner, installer)
    rtk = _ensure_rtk(root, which, runner, rtk_installer)
    _ensure_index(root, codegraph, runner)
    _ensure_rtk_recall(root, rtk, runner)
    print("\nAgent harness ready.")


def main() -> int:
    """Run bootstrap and convert expected failures to a nonzero exit."""
    try:
        bootstrap()
    except BootstrapError as error:
        print(f"Agent bootstrap failed:\n{error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
