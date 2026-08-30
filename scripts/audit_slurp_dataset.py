"""Audit a compiled SLURP GoldGraph dataset and its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from premove_itn import build_gold_graph


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path) -> int:
    manifest = json.loads((root / "manifest.json").read_text())
    rows = [json.loads(line) for line in (root / "dataset.jsonl").open()]
    provenance = [json.loads(line) for line in (root / "provenance.jsonl").open()]
    errors: Counter[str] = Counter()
    keys = []
    distribution: Counter[str] = Counter()
    partitions: Counter[str] = Counter()
    for row in rows:
        if set(row) != {"text", "expected_text", "partition", "kind", "kinds"}:
            errors["schema"] += 1
            continue
        if not row["text"] or not row["expected_text"]:
            errors["empty"] += 1
        if row["partition"] not in {"train", "validation", "test"}:
            errors["partition"] += 1
        expected_kind = (
            "KEEP"
            if not row["kinds"]
            else row["kinds"][0]
            if len(row["kinds"]) == 1
            else "MULTI"
        )
        if row["kind"] != expected_kind:
            errors["kind"] += 1
        if build_gold_graph(row["text"], row["expected_text"]) is None:
            errors["oracle"] += 1
        keys.append((row["text"], row["expected_text"]))
        partitions[row["partition"]] += 1
        distribution.update(row["kinds"] or ["KEEP"])
    if len(keys) != len(set(keys)):
        errors["duplicate"] += len(keys) - len(set(keys))
    if len(rows) != len(provenance):
        errors["provenance_length"] += 1
    if manifest["records"] != len(rows):
        errors["manifest_records"] += 1
    if manifest["distribution"] != dict(sorted(distribution.items())):
        errors["manifest_distribution"] += 1
    if manifest["partition_distribution"] != dict(sorted(partitions.items())):
        errors["manifest_partitions"] += 1
    for name, expected_hash in manifest["artifacts"].items():
        if _sha256(root / name) != expected_hash:
            errors["artifact_hash"] += 1
    report = {
        "records": len(rows),
        "provenance_records": len(provenance),
        "distribution": dict(sorted(distribution.items())),
        "partitions": dict(sorted(partitions.items())),
        "errors": dict(sorted(errors.items())),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(bool(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    raise SystemExit(audit(parser.parse_args().dataset))


if __name__ == "__main__":
    main()
