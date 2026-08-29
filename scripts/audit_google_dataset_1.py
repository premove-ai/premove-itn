"""Audit a compiled Google TN Dataset 1 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from premove_itn import build_gold_graph

try:
    from scripts.build_google_dataset_1 import _quality_issues
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from build_google_dataset_1 import _quality_issues

# Rust's measurement realizer intentionally writes a space before ``%``.
# Treat that form as canonical; flag other accidental spaces around
# punctuation and repeated whitespace.
SUSPICIOUS_SPACING = re.compile(r"\s{2,}|\s+[.,;:!?\)\]}]|[\(\[]\s")


def _audit_graph(values: tuple[str, str]) -> bool:
    return build_gold_graph(*values) is not None


def _sample(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}
    for row in rows:
        for kind in row["kinds"] or ["KEEP"]:
            selected.setdefault(kind, {})
            selected[kind].setdefault(
                hashlib.sha256(
                    (row["text"] + "\0" + row["expected_text"]).encode()
                ).hexdigest(),
                row,
            )
    output = []
    for candidates in selected.values():
        output.extend(row for _, row in sorted(candidates.items())[:limit])
    return list({(row["text"], row["expected_text"]): row for row in output}.values())


def audit(path: Path, sample_per_kind: int, workers: int) -> int:
    root = path
    manifest = json.loads((root / "manifest.json").read_text())
    rows = [json.loads(line) for line in (root / "dataset.jsonl").open()]
    keys = [(row["text"], row["expected_text"]) for row in rows]
    errors = []
    for row in rows:
        if set(row) != {"text", "expected_text", "partition", "kind", "kinds"}:
            errors.append("schema")
        if not row["text"] or not row["expected_text"]:
            errors.append("empty")
        if row["kind"] != (
            "KEEP"
            if not row["kinds"]
            else row["kinds"][0]
            if len(row["kinds"]) == 1
            else "MULTI"
        ):
            errors.append("kind")
        if SUSPICIOUS_SPACING.search(row["text"]):
            errors.append("source_spacing")
        if SUSPICIOUS_SPACING.search(row["expected_text"]):
            errors.append("target_spacing")
        errors.extend(_quality_issues(row["text"]))
        errors.extend(_quality_issues(row["expected_text"]))
    if len(set(keys)) != len(keys):
        errors.append("duplicate")
    sample = _sample(rows, sample_per_kind)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(
                _audit_graph,
                ((row["text"], row["expected_text"]) for row in sample),
                chunksize=32,
            )
        )
    if not all(results):
        errors.append("oracle")
    print(
        json.dumps(
            {
                "records": len(rows),
                "sampled_for_full_oracle": len(sample),
                "full_oracle_sample_passed": all(results),
                "suspicious_quality_flags": {
                    name: errors.count(name)
                    for name in sorted(set(errors))
                    if name not in {"schema", "empty", "kind", "duplicate", "oracle"}
                },
                "manifest_records": manifest["records"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--sample-per-kind", type=int, default=100)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    raise SystemExit(audit(args.dataset, args.sample_per_kind, args.workers))


if __name__ == "__main__":
    main()
