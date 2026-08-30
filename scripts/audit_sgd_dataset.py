"""Audit a compiled Schema-Guided Dialogue ITN artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from premove_itn import build_gold_graph

try:
    from scripts.build_google_dataset_1 import _quality_issues
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from build_google_dataset_1 import _quality_issues


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sample(rows: list[dict[str, object]], limit: int):
    selected: dict[str, list[tuple[str, dict[str, object]]]] = {}
    for row in rows:
        digest = hashlib.sha256(
            f"{row['text']}\0{row['expected_text']}".encode()
        ).hexdigest()
        for kind in row["kinds"] or ["KEEP"]:
            selected.setdefault(kind, []).append((digest, row))
    pairs = {
        (row["text"], row["expected_text"]): row
        for values in selected.values()
        for _, row in sorted(values)[:limit]
    }
    return list(pairs.values())


def audit(path: Path, sample_per_kind: int) -> int:
    manifest = json.loads((path / "manifest.json").read_text())
    rows = [json.loads(line) for line in (path / "dataset.jsonl").open()]
    provenance_count = sum(1 for _ in (path / "provenance.jsonl").open())
    errors = Counter()
    keys = []
    for row in rows:
        if set(row) != {"text", "expected_text", "partition", "kind", "kinds"}:
            errors["schema"] += 1
            continue
        keys.append((row["text"], row["expected_text"]))
        if not row["text"] or not row["expected_text"]:
            errors["empty"] += 1
        expected_kind = (
            "KEEP"
            if not row["kinds"]
            else row["kinds"][0]
            if len(row["kinds"]) == 1
            else "MULTI"
        )
        if row["kind"] != expected_kind:
            errors["kind"] += 1
        if row["partition"] not in {"train", "validation", "test"}:
            errors["partition"] += 1
        for reason in (
            *_quality_issues(row["text"]),
            *_quality_issues(row["expected_text"]),
        ):
            errors[f"quality:{reason}"] += 1
    errors["duplicate"] += len(keys) - len(set(keys))
    sample = _sample(rows, sample_per_kind)
    failures = sum(
        build_gold_graph(row["text"], row["expected_text"]) is None for row in sample
    )
    errors["oracle"] += failures
    result = {
        "records": len(rows),
        "manifest_records": manifest.get("records"),
        "sampled_for_full_oracle": len(sample),
        "full_oracle_sample_passed": failures == 0,
        "errors": {key: value for key, value in sorted(errors.items()) if value},
    }
    if manifest.get("records") != len(rows):
        errors["manifest_records"] += 1
        result["errors"]["manifest_records"] = 1
    if provenance_count != len(rows):
        errors["provenance_records"] += 1
        result["errors"]["provenance_records"] = 1
    distribution = Counter(row["kind"] for row in rows)
    if manifest.get("distribution") != dict(sorted(distribution.items())):
        errors["manifest_distribution"] += 1
        result["errors"]["manifest_distribution"] = 1
    for name, artifact in manifest.get("artifacts", {}).items():
        artifact_path = path / name
        if (
            not artifact_path.is_file()
            or artifact.get("bytes") != artifact_path.stat().st_size
            or artifact.get("sha256") != _sha256(artifact_path)
        ):
            errors[f"artifact:{name}"] += 1
            result["errors"][f"artifact:{name}"] = 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(any(errors.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--sample-per-kind", type=int, default=100)
    args = parser.parse_args()
    raise SystemExit(audit(args.dataset, args.sample_per_kind))


if __name__ == "__main__":
    main()
