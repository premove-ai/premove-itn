"""Build one compact index of all generated evaluation results."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

HISTORY_PATH = (
    Path(__file__).resolve().parents[1] / "docs/evaluations/evaluation-history.json"
)
EXPOSURE_RE = re.compile(r"(?:checkpoint_|_)(\d{4,6})(?:\.pt)?")
DATASET_NAMES = (
    "google_validation",
    "conversational_validation",
    "golden",
    "numb3rs",
    "polynorm",
)
METRIC_KEYS = (
    "accuracy",
    "correct",
    "exact",
    "total",
    "overall",
    "keep",
    "positive",
    "multi",
    "candidate_bearing",
    "candidate_bearing_keep",
    "per_kind",
    "per_source",
    "per_category",
    "oracle",
    "error_categories",
    "semantic",
    "all",
    "reachable_only",
)


def _compact_metrics(block: object) -> dict[str, object] | None:
    if not isinstance(block, dict):
        return None
    return {key: block[key] for key in METRIC_KEYS if key in block}


def _dataset_name(path: Path, payload: dict[str, object]) -> str:
    for value in (path.parent.name, path.name, str(payload.get("dataset", ""))):
        for name in DATASET_NAMES:
            if name in value:
                return name
    return path.parent.name


def _exposure(path: Path, payload: dict[str, object]) -> int | None:
    metadata = payload.get("checkpoint_metadata")
    if isinstance(metadata, dict) and metadata.get("checkpoint_exposure") is not None:
        return int(metadata["checkpoint_exposure"])
    checkpoint = payload.get("checkpoint")
    match = EXPOSURE_RE.search(str(checkpoint or path))
    return int(match.group(1)) if match else None


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _entry(root: Path, path: Path, payload: dict[str, object]) -> dict[str, object]:
    metadata = payload.get("checkpoint_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    entry: dict[str, object] = {
        "source_file": _relative(root, path),
        "metrics_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dataset": _dataset_name(path, payload),
        "checkpoint_exposure": _exposure(path, payload),
        "checkpoint": metadata.get("checkpoint", payload.get("checkpoint")),
        "records": payload.get("records"),
        "benchmark_role": payload.get("benchmark_role"),
        "evaluation_only": payload.get("evaluation_only"),
        "trained": _compact_metrics(payload.get("trained")),
        "strict_exact": _compact_metrics(payload.get("strict_exact")),
        "deterministic_baseline": _compact_metrics(
            payload.get("deterministic_baseline")
        ),
    }
    return entry


def build_history(root: Path) -> dict[str, object]:
    """Return a deterministic compact index of generated evaluation JSONs."""
    entries: list[dict[str, object]] = []
    for path in sorted(root.glob("data/generated/**/*.json")):
        if path == HISTORY_PATH:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or "trained" not in payload:
            continue
        entries.append(_entry(root, path, payload))
    entries.sort(key=lambda item: str(item["source_file"]))
    return {
        "schema_version": 1,
        "description": (
            "Compact index of generated evaluation results. Raw metric JSONs "
            "remain the source of detailed predictions and error samples."
        ),
        "entries": entries,
    }


def write_history(path: Path, history: dict[str, object]) -> None:
    """Atomically write the consolidated evaluation-history artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(history, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    root = HISTORY_PATH.parents[2]
    write_history(HISTORY_PATH, build_history(root))
    print(HISTORY_PATH)


if __name__ == "__main__":
    main()
