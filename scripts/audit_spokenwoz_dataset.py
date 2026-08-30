"""Audit a compiled SpokenWOZ GoldGraph dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from premove_itn import build_gold_graph


def audit(root: Path) -> int:
    manifest = json.loads((root / "manifest.json").read_text())
    rows = [json.loads(line) for line in (root / "dataset.jsonl").open()]
    provenance = [json.loads(line) for line in (root / "provenance.jsonl").open()]
    errors: Counter[str] = Counter()
    keys: set[tuple[str, str]] = set()
    distribution: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if set(row) != {"text", "expected_text", "partition", "kind", "kinds"}:
            errors["schema"] += 1
            continue
        key = (row["text"], row["expected_text"])
        if key in keys:
            errors["duplicate"] += 1
        keys.add(key)
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
        if build_gold_graph(*key) is None:
            errors["oracle"] += 1
        distribution[row["kind"]] += 1
        if index >= len(provenance) or provenance[index].get("record_index") != index:
            errors["provenance"] += 1
    if len(provenance) != len(rows):
        errors["provenance_length"] += 1
    if manifest.get("records") != len(rows):
        errors["manifest_records"] += 1
    if manifest.get("distribution") != dict(sorted(distribution.items())):
        errors["manifest_distribution"] += 1
    print(
        json.dumps(
            {
                "records": len(rows),
                "distribution": dict(sorted(distribution.items())),
                "errors": dict(sorted(errors.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return bool(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    raise SystemExit(audit(args.dataset))


if __name__ == "__main__":
    main()
