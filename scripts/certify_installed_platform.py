"""Record and verify one clean-installed release wheel environment."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import premove_itn
from premove_itn import _rust


def normalized_architecture(value: str) -> str:
    """Return the architecture spelling used by the compatibility matrix."""
    aliases = {"AMD64": "x86_64", "aarch64": "arm64"}
    return aliases.get(value, value)


def collect_platform_evidence(wheel: Path) -> dict[str, object]:
    """Collect evidence from the interpreter that imported the wheel."""
    package_path = Path(premove_itn.__file__).resolve()
    if "site-packages" not in package_path.parts:
        raise RuntimeError(
            f"package was not imported from site-packages: {package_path}"
        )
    build = _rust.build_info()
    if build["profile"] != "release" or build["debug_assertions"] is not False:
        raise RuntimeError(f"wheel contains a non-release Rust extension: {build}")

    cli = Path(sys.executable).with_name("premove-itn")
    help_result = subprocess.run(
        [str(cli), "--help"], capture_output=True, text=True, check=False
    )
    version_result = subprocess.run(
        [str(cli), "--version"], capture_output=True, text=True, check=False
    )
    if help_result.returncode or version_result.returncode:
        raise RuntimeError("installed CLI smoke test failed")

    import torch
    import transformers

    return {
        "schema_version": 1,
        "result": "pass",
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": normalized_architecture(platform.machine()),
        "python": platform.python_version(),
        "package_version": version("premove-itn"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "rust_target": build["target"],
        "rustc": build["rustc_version"],
        "rust_profile": build["profile"],
        "rust_debug_assertions": build["debug_assertions"],
        "wheel": wheel.name,
        "package_path": str(package_path),
        "cli_version": version_result.stdout.strip(),
    }


def verify_expected_platform(
    evidence: dict[str, object],
    *,
    expected_os: str,
    expected_arch: str,
    expected_python: str,
) -> None:
    """Fail if a matrix cell ran on a different environment."""
    actual_python = ".".join(str(evidence["python"]).split(".")[:2])
    expected = {
        "os": expected_os,
        "architecture": expected_arch,
        "python": expected_python,
    }
    actual = {
        "os": evidence["os"],
        "architecture": evidence["architecture"],
        "python": actual_python,
    }
    if actual != expected:
        raise RuntimeError(f"platform mismatch: expected {expected}, got {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--expected-os", required=True)
    parser.add_argument("--expected-arch", required=True)
    parser.add_argument("--expected-python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    evidence = collect_platform_evidence(arguments.wheel)
    verify_expected_platform(
        evidence,
        expected_os=arguments.expected_os,
        expected_arch=arguments.expected_arch,
        expected_python=arguments.expected_python,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(
        f"certified {arguments.expected_os}/{arguments.expected_arch} "
        f"Python {arguments.expected_python}"
    )


if __name__ == "__main__":
    main()
