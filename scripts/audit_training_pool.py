"""Audit every record and artifact in a unified training pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from premove_itn import build_gold_graph
from premove_itn.model_inputs import MODEL_MAX_TOKENS

PARTITIONS = ("train", "validation", "test")
RECORD_KEYS = {"text", "expected_text", "partition", "kind", "kinds"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit(path: Path, *, tokenizer=None) -> int:
    manifest = json.loads((path / "manifest.json").read_text())
    rows = [json.loads(line) for line in (path / "dataset.jsonl").open()]
    provenance = [json.loads(line) for line in (path / "provenance.jsonl").open()]
    errors: Counter[str] = Counter()
    distribution: Counter[str] = Counter()
    partition_distribution: Counter[str] = Counter()
    pairs = set()
    text_partitions: dict[str, set[str]] = {}
    target_partitions: dict[str, set[str]] = {}

    for index, row in enumerate(rows):
        if set(row) != RECORD_KEYS:
            errors["schema"] += 1
            continue
        text, target, partition = (
            row["text"],
            row["expected_text"],
            row["partition"],
        )
        if not isinstance(text, str) or not text:
            errors["text"] += 1
            continue
        if not isinstance(target, str) or not target:
            errors["expected_text"] += 1
            continue
        if partition not in PARTITIONS:
            errors["partition"] += 1
            continue
        kinds = row["kinds"]
        if not isinstance(kinds, list) or not all(
            isinstance(kind, str) for kind in kinds
        ):
            errors["kinds"] += 1
            continue
        expected_kind = (
            "KEEP" if not kinds else kinds[0] if len(kinds) == 1 else "MULTI"
        )
        if row["kind"] != expected_kind:
            errors["kind"] += 1
        pair = (text, target)
        if pair in pairs:
            errors["duplicate"] += 1
        pairs.add(pair)
        text_partitions.setdefault(text, set()).add(partition)
        target_partitions.setdefault(target, set()).add(partition)
        distribution[row["kind"]] += 1
        partition_distribution[partition] += 1
        if build_gold_graph(text, target) is None:
            errors["oracle"] += 1
        if tokenizer is not None:
            encoding = tokenizer(text, add_special_tokens=True, truncation=False)
            if len(encoding["input_ids"]) > MODEL_MAX_TOKENS:
                errors["model_token_limit"] += 1
        if index >= len(provenance) or provenance[index].get("record_index") != index:
            errors["provenance_alignment"] += 1

    if len(provenance) != len(rows):
        errors["provenance_count"] += 1
    errors["text_partition_leakage"] += sum(
        len(partitions) > 1 for partitions in text_partitions.values()
    )
    errors["target_partition_leakage"] += sum(
        len(partitions) > 1 for partitions in target_partitions.values()
    )
    if manifest.get("records") != len(rows):
        errors["manifest_records"] += 1
    if manifest.get("distribution") != dict(sorted(distribution.items())):
        errors["manifest_distribution"] += 1
    if manifest.get("partition_distribution") != dict(
        sorted(partition_distribution.items())
    ):
        errors["manifest_partition_distribution"] += 1

    for partition in PARTITIONS:
        split_rows = [json.loads(line) for line in (path / f"{partition}.jsonl").open()]
        if split_rows != [row for row in rows if row.get("partition") == partition]:
            errors[f"split:{partition}"] += 1
    for name, expected in manifest.get("artifacts", {}).items():
        artifact = path / name
        if (
            not artifact.is_file()
            or artifact.stat().st_size != expected.get("bytes")
            or _sha256(artifact) != expected.get("sha256")
        ):
            errors[f"artifact:{name}"] += 1

    token_gate = manifest.get("model_token_gate", {})
    if token_gate.get("enforced") and tokenizer is None:
        for item in provenance:
            count = item.get("model_token_count")
            if not isinstance(count, int) or count > MODEL_MAX_TOKENS:
                errors["recorded_model_token_count"] += 1

    licensing = manifest.get("licensing")
    if not isinstance(licensing, dict) or not isinstance(
        licensing.get("sources"), list
    ):
        errors["licensing"] += 1
    else:
        licenses = {
            source.get("license")
            for source in licensing["sources"]
            if isinstance(source, dict)
        }
        if len(licensing["sources"]) != len(manifest.get("inputs", [])):
            errors["license_source_count"] += 1
        if (
            licenses
            & {
                "CC-BY-NC-4.0",
                "CC-BY-SA-4.0",
                "LicenseRef-Kaggle-Competition-Rules",
            }
            and licensing.get("use_scope") != "non_commercial_research"
        ):
            errors["restrictive_license_scope"] += 1

    report = {
        "records": len(rows),
        "all_records_oracle_checked": len(rows),
        "model_token_gate_rechecked": tokenizer is not None,
        "errors": {key: value for key, value in sorted(errors.items()) if value},
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(any(errors.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    raise SystemExit(audit(args.dataset))


if __name__ == "__main__":
    main()
