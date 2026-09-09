"""Certify frozen prediction equivalence on one inference backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
from importlib.metadata import version
from pathlib import Path

import premove_itn
from premove_itn import PremoveITN, _rust
from premove_itn.contextual import DEFAULT_MODEL_ID, DEFAULT_REVISION, _resolve_device
from scripts.certify_installed_platform import (
    normalized_architecture,
    verify_wheel_platform,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _verify_dataset(dataset: Path, frozen_digest: Path) -> str:
    expected_digest, expected_name = frozen_digest.read_text(encoding="utf-8").split()
    actual_digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if expected_name != dataset.name or actual_digest != expected_digest:
        raise RuntimeError("frozen dataset identity mismatch")
    return actual_digest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frozen-digest", type=Path, required=True)
    parser.add_argument("--reference-records", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--expected-wheel-platform", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--unavailable-device",
        action="append",
        choices=("cpu", "mps", "cuda"),
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    verify_wheel_platform(arguments.wheel, arguments.expected_wheel_platform)

    dataset_digest = _verify_dataset(arguments.dataset, arguments.frozen_digest)
    wheel_digest = _sha256(arguments.wheel)
    reference_digest = _sha256(arguments.reference_records)
    rows = _read_jsonl(arguments.dataset)
    references = _read_jsonl(arguments.reference_records)
    if len(rows) != 1500 or len(references) != 1500:
        raise RuntimeError("compatibility certification requires exactly 1,500 rows")
    if len({str(row["id"]) for row in rows}) != 1500:
        raise RuntimeError("frozen dataset contains duplicate IDs")
    expected_by_id = {str(row["id"]): row for row in references}
    if len(expected_by_id) != 1500:
        raise RuntimeError("reference predictions contain duplicate IDs")
    for row in rows:
        reference = expected_by_id.get(str(row["id"]))
        if reference is None or reference.get("text") != row.get("text"):
            raise RuntimeError(f"reference input mismatch for {row.get('id')}")

    auto_device, torch = _resolve_device("auto")
    if str(auto_device) != arguments.device:
        raise RuntimeError(
            f"device='auto' selected {auto_device}, expected {arguments.device}"
        )
    explicit_device, _ = _resolve_device(arguments.device)
    if str(explicit_device) != arguments.device:
        raise RuntimeError("explicit device selection mismatch")
    for unavailable_device in arguments.unavailable_device:
        try:
            PremoveITN.from_pretrained(device=unavailable_device)
        except RuntimeError as exc:
            if "not available" not in str(exc):
                raise RuntimeError(
                    f"unexpected {unavailable_device} failure: {exc}"
                ) from exc
        else:
            raise RuntimeError(f"{unavailable_device} was expected to be unavailable")

    itn = PremoveITN.from_pretrained(device=arguments.device)
    mismatches = []
    for row in rows:
        row_id = str(row["id"])
        actual = itn.normalize(str(row["text"]))
        expected = str(expected_by_id[row_id]["prediction"])
        if actual != expected:
            mismatches.append({"id": row_id, "expected": expected, "actual": actual})

    package_path = Path(premove_itn.__file__).resolve()
    if "site-packages" not in package_path.parts:
        raise RuntimeError(
            f"package was not imported from site-packages: {package_path}"
        )
    build = _rust.build_info()
    if build["profile"] != "release" or build["debug_assertions"] is not False:
        raise RuntimeError(f"wheel contains a non-release Rust extension: {build}")
    evidence = {
        "schema_version": 1,
        "result": "pass" if not mismatches else "fail",
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": normalized_architecture(platform.machine()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": arguments.device,
        "auto_device": str(auto_device),
        "unavailable_devices": arguments.unavailable_device,
        "package_version": version("premove-itn"),
        "model_repository": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_REVISION,
        "dataset_sha256": dataset_digest,
        "reference": "first-evaluation/premove-itn/records.jsonl",
        "reference_predictions_sha256": reference_digest,
        "wheel": arguments.wheel.name,
        "wheel_sha256": wheel_digest,
        "source_commit": arguments.source_commit,
        "rows": len(rows),
        "identical_rows": len(rows) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "peak_rss_bytes": _peak_rss_bytes(),
        "rust_target": build["target"],
        "rust_profile": build["profile"],
        "rust_debug_assertions": build["debug_assertions"],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    if mismatches:
        raise RuntimeError(f"{len(mismatches)} frozen predictions differ")
    print(f"certified {len(rows)}/{len(rows)} exact predictions on {arguments.device}")


if __name__ == "__main__":
    main()
