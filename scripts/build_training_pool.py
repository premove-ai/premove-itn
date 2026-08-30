"""Combine compiled corpora into one leakage-safe GoldGraph training pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from premove_itn import build_gold_graph
from premove_itn.model_inputs import (
    MODEL_MAX_TOKENS,
    MODEL_NAME,
    MODEL_REVISION,
    load_model_tokenizer,
)

SEED = "premove-itn/training-pool/v1"
PARTITIONS = ("train", "validation", "test")
PARTITION_ORDER = {name: index for index, name in enumerate(PARTITIONS)}
RECORD_KEYS = {"text", "expected_text", "partition", "kind", "kinds"}
LICENSE_ALIASES = {
    "APACHE-2.0": "Apache-2.0",
    "CC-BY-3.0": "CC-BY-3.0",
    "CC-BY-4.0": "CC-BY-4.0",
    "CC-BY-NC-4.0": "CC-BY-NC-4.0",
    "CC-BY-SA-4.0": "CC-BY-SA-4.0",
    "LICENSEREF-KAGGLE-COMPETITION-RULES": "LicenseRef-Kaggle-Competition-Rules",
    "MIT": "MIT",
}
LICENSE_RESTRICTIONS = {
    "Apache-2.0": ("attribution", "notice_and_license_preservation"),
    "CC-BY-3.0": ("attribution",),
    "CC-BY-4.0": ("attribution", "change_indication"),
    "CC-BY-NC-4.0": ("attribution", "change_indication", "non_commercial"),
    "CC-BY-SA-4.0": ("attribution", "change_indication", "share_alike"),
    "LicenseRef-Kaggle-Competition-Rules": (
        "competition_rules",
        "no_redistribution_without_permission",
    ),
    "MIT": ("copyright_and_license_preservation",),
}
RESTRICTIVE_LICENSES = {
    "CC-BY-NC-4.0",
    "CC-BY-SA-4.0",
    "LicenseRef-Kaggle-Competition-Rules",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(f"{SEED}\0{value}".encode()).hexdigest()


def _partition(component_key: str) -> str:
    bucket = int(_digest(component_key), 16) % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def _source_name(root: Path, manifest: dict[str, object]) -> str:
    value = manifest.get("kind") or manifest.get("dataset") or root.name
    if not isinstance(value, str) or not value:
        raise ValueError(f"{root}/manifest.json has no usable dataset identity")
    return value


def _source_license(
    source: str,
    manifest: dict[str, object],
    overrides: dict[str, str],
) -> str:
    value = overrides.get(source, manifest.get("license"))
    if value is None and isinstance(manifest.get("source"), dict):
        value = manifest["source"].get("license")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{source}: missing license metadata; provide --source-license "
            f"{source}=SPDX-ID"
        )
    normalized = LICENSE_ALIASES.get(value.strip().upper())
    if normalized is None:
        raise ValueError(f"{source}: unsupported or ambiguous license {value!r}")
    return normalized


def _read_source(root: Path):
    manifest_path = root / "manifest.json"
    dataset_path = root / "dataset.jsonl"
    provenance_path = root / "provenance.jsonl"
    for path in (manifest_path, dataset_path, provenance_path):
        if not path.is_file():
            raise ValueError(f"missing compiled source artifact: {path}")
    manifest = json.loads(manifest_path.read_text())
    name = _source_name(root, manifest)
    records = [json.loads(line) for line in dataset_path.open()]
    provenance = [json.loads(line) for line in provenance_path.open()]
    if len(records) != len(provenance):
        raise ValueError(f"{root}: dataset/provenance length mismatch")
    if manifest.get("records") != len(records):
        raise ValueError(f"{root}: manifest record count mismatch")
    return name, manifest_path, dataset_path, provenance_path, records, provenance


def _certified_source_oracle(
    source: str,
    manifest: dict[str, object],
    dataset_path: Path,
    provenance_path: Path,
) -> None:
    if not (
        manifest.get("all_selected_oracle_reachable") is True
        or manifest.get("all_records_gold_graph_reachable") is True
    ):
        raise ValueError(f"{source}: source manifest does not certify every record")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"{source}: certified source has no artifact hashes")
    for path in (dataset_path, provenance_path):
        metadata = artifacts.get(path.name)
        if not isinstance(metadata, dict) or metadata.get("sha256") != _sha256(path):
            raise ValueError(f"{source}: certified artifact hash mismatch: {path.name}")


def _validate_record(record: dict[str, object], location: str) -> None:
    if set(record) != RECORD_KEYS:
        raise ValueError(f"{location}: invalid record schema")
    if not isinstance(record["text"], str) or not record["text"]:
        raise ValueError(f"{location}: invalid text")
    if not isinstance(record["expected_text"], str) or not record["expected_text"]:
        raise ValueError(f"{location}: invalid expected_text")
    kinds = record["kinds"]
    if not isinstance(kinds, list) or not all(isinstance(kind, str) for kind in kinds):
        raise ValueError(f"{location}: invalid kinds")
    expected_kind = "KEEP" if not kinds else kinds[0] if len(kinds) == 1 else "MULTI"
    if record["kind"] != expected_kind:
        raise ValueError(f"{location}: inconsistent kind")
    if record["partition"] not in PARTITIONS:
        raise ValueError(f"{location}: invalid source partition")


def _oracle_chunk(
    pairs: list[tuple[str, str]],
) -> tuple[int, tuple[str, str] | None]:
    for pair in pairs:
        if build_gold_graph(*pair) is None:
            return len(pairs), pair
    return len(pairs), None


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _progress(label: str, completed: int, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    rate = completed / elapsed if elapsed else 0.0
    eta = (total - completed) / rate if rate else 0.0
    print(
        f"{label}: {completed}/{total} ({completed / total:.1%}), "
        f"{rate:.1f} records/s, elapsed {_duration(elapsed)}, "
        f"ETA {_duration(eta)}",
        file=sys.stderr,
        flush=True,
    )


def _token_count(tokenizer, text: str) -> int:
    encoding = tokenizer(text, add_special_tokens=True, truncation=False)
    return len(encoding["input_ids"])


def build_training_pool(
    inputs: list[Path],
    output: Path,
    *,
    tokenizer=None,
    non_commercial_research: bool = False,
    source_licenses: dict[str, str] | None = None,
    workers: int = min(7, os.cpu_count() or 1),
    trust_source_oracle: bool = False,
) -> dict[str, object]:
    if not inputs:
        raise ValueError("at least one compiled dataset root is required")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if workers < 1:
        raise ValueError("workers must be at least one")

    occurrences: dict[tuple[str, str], list[dict[str, object]]] = {}
    union_find = _UnionFind()
    input_reports = []
    source_names = set()
    license_reports = []
    source_licenses = source_licenses or {}
    for root in sorted(inputs, key=lambda path: str(path.resolve())):
        (
            source,
            manifest_path,
            dataset_path,
            provenance_path,
            records,
            provenance,
        ) = _read_source(root)
        if source in source_names:
            raise ValueError(f"duplicate compiled source identity: {source}")
        source_names.add(source)
        source_manifest = json.loads(manifest_path.read_text())
        license_id = _source_license(source, source_manifest, source_licenses)
        if trust_source_oracle:
            _certified_source_oracle(
                source, source_manifest, dataset_path, provenance_path
            )
        if license_id in RESTRICTIVE_LICENSES and not non_commercial_research:
            raise ValueError(
                f"{source}: {license_id} requires explicit "
                "non-commercial research opt-in"
            )
        license_reports.append(
            {
                "source": source,
                "license": license_id,
                "restrictions": list(LICENSE_RESTRICTIONS[license_id]),
            }
        )
        input_reports.append(
            {
                "source": source,
                "root": str(root),
                "records": len(records),
                "manifest_sha256": _sha256(manifest_path),
                "dataset_sha256": _sha256(dataset_path),
                "provenance_sha256": _sha256(provenance_path),
            }
        )
        for index, (record, source_provenance) in enumerate(
            zip(records, provenance, strict=True)
        ):
            _validate_record(record, f"{root}/dataset.jsonl:{index + 1}")
            text, target = record["text"], record["expected_text"]
            union_find.union(text, target)
            occurrences.setdefault((text, target), []).append(
                {
                    "source": source,
                    "source_record_index": index,
                    "source_partition": record["partition"],
                    "kinds": record["kinds"],
                    "provenance": source_provenance,
                }
            )

    component_members: dict[str, list[str]] = {}
    for value in union_find.parent:
        component_members.setdefault(union_find.find(value), []).append(value)
    component_key = {root: min(members) for root, members in component_members.items()}

    selected = []
    excluded = []
    ordered_pairs = sorted(occurrences, key=lambda pair: _digest("\0".join(pair)))
    token_started = time.monotonic()
    for pair_index, (text, target) in enumerate(ordered_pairs, 1):
        sources = sorted(
            occurrences[(text, target)],
            key=lambda item: (
                item["source"],
                item["source_record_index"],
            ),
        )
        kinds = sorted({kind for source in sources for kind in source["kinds"]})
        if text == target:
            kinds = []
        kind = "KEEP" if not kinds else kinds[0] if len(kinds) == 1 else "MULTI"
        partition = _partition(component_key[union_find.find(text)])
        token_count = _token_count(tokenizer, text) if tokenizer is not None else None
        value = {
            "text": text,
            "expected_text": target,
            "partition": partition,
            "kind": kind,
            "kinds": kinds,
        }
        provenance = {"sources": sources}
        if token_count is not None:
            provenance["model_token_count"] = token_count
        if token_count is not None and token_count > MODEL_MAX_TOKENS:
            excluded.append(
                {
                    **value,
                    **provenance,
                    "reason": "model_token_limit",
                    "limit": MODEL_MAX_TOKENS,
                }
            )
        else:
            selected.append((value, provenance))
        if pair_index % 50_000 == 0 or pair_index == len(ordered_pairs):
            _progress("token gate", pair_index, len(ordered_pairs), token_started)

    oracle_groups: dict[str, list[dict[str, object]]] = {}
    identity_records = 0
    for record, provenance in selected:
        if record["text"] == record["expected_text"]:
            identity_records += 1
            continue
        source = provenance["sources"][0]["source"]
        oracle_groups.setdefault(source, []).append(record)
    print(
        f"oracle: {identity_records} identity records proven directly",
        file=sys.stderr,
        flush=True,
    )
    for source in (() if trust_source_oracle else sorted(oracle_groups)):
        records = oracle_groups[source]
        print(
            f"oracle: checking {source}: {len(records)} records",
            file=sys.stderr,
            flush=True,
        )
        chunks = [
            [
                (record["text"], record["expected_text"])
                for record in records[index : index + 64]
            ]
            for index in range(0, len(records), 64)
        ]
        started = time.monotonic()
        completed = 0
        next_report = started + 5
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_oracle_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                count, failed_pair = future.result()
                completed += count
                if failed_pair is not None:
                    raise ValueError(
                        "final deduplicated record is not GoldGraph-reachable: "
                        f"{failed_pair[0]!r} -> {failed_pair[1]!r}"
                    )
                now = time.monotonic()
                if now >= next_report or completed == len(records):
                    _progress(f"oracle {source}", completed, len(records), started)
                    next_report = now + 5
        print(f"oracle: passed {source}", file=sys.stderr, flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    handles = {
        partition: (staging / f"{partition}.jsonl").open("w", encoding="utf-8")
        for partition in PARTITIONS
    }
    distribution: Counter[str] = Counter()
    partition_distribution: Counter[str] = Counter()
    source_distribution: Counter[str] = Counter()
    source_occurrence_distribution: Counter[str] = Counter()
    source_kind_occurrences: Counter[str] = Counter()
    with (
        (staging / "dataset.jsonl").open("w", encoding="utf-8") as dataset_handle,
        (staging / "provenance.jsonl").open("w", encoding="utf-8") as provenance_handle,
    ):
        try:
            for index, (record, provenance) in enumerate(
                sorted(
                    selected,
                    key=lambda item: (
                        PARTITION_ORDER[item[0]["partition"]],
                        _digest(f"{item[0]['text']}\0{item[0]['expected_text']}"),
                    ),
                )
            ):
                line = json.dumps(record, ensure_ascii=False) + "\n"
                dataset_handle.write(line)
                handles[record["partition"]].write(line)
                provenance_handle.write(
                    json.dumps(
                        {"record_index": index, **provenance}, ensure_ascii=False
                    )
                    + "\n"
                )
                distribution[record["kind"]] += 1
                partition_distribution[record["partition"]] += 1
                source_distribution[provenance["sources"][0]["source"]] += 1
                for source in provenance["sources"]:
                    source_occurrence_distribution[source["source"]] += 1
                    for kind in record["kinds"] or ["KEEP"]:
                        source_kind_occurrences[f"{source['source']}:{kind}"] += 1
        finally:
            for handle in handles.values():
                handle.close()
    with (staging / "excluded.jsonl").open("w", encoding="utf-8") as handle:
        for value in excluded:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    artifacts = [
        staging / name
        for name in (
            "dataset.jsonl",
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
            "provenance.jsonl",
            "excluded.jsonl",
        )
    ]
    manifest = {
        "kind": "unified_training_pool",
        "schema_version": 1,
        "inputs": input_reports,
        "licensing": {
            "use_scope": (
                "non_commercial_research"
                if any(
                    report["license"] in RESTRICTIVE_LICENSES
                    for report in license_reports
                )
                else "license_compliant"
            ),
            "sources": license_reports,
            "effective_restrictions": sorted(
                {
                    restriction
                    for report in license_reports
                    for restriction in report["restrictions"]
                }
            ),
        },
        "selection": {
            "seed": SEED,
            "partition_method": "seeded_sha256_of_text_target_connected_component",
            "partitions": {"train": 90, "validation": 5, "test": 5},
            "unique_by": ["text", "expected_text"],
            "leakage_grouped_by": ["text", "expected_text"],
        },
        "model_token_gate": {
            "enforced": tokenizer is not None,
            "model": MODEL_NAME,
            "revision": MODEL_REVISION,
            "max_tokens": MODEL_MAX_TOKENS,
            "excluded": len(excluded),
        },
        "records_before_deduplication": sum(
            len(values) for values in occurrences.values()
        ),
        "duplicates": sum(len(values) - 1 for values in occurrences.values()),
        "records": len(selected),
        "distribution": dict(sorted(distribution.items())),
        "partition_distribution": dict(sorted(partition_distribution.items())),
        "source_distribution": dict(sorted(source_distribution.items())),
        "source_occurrence_distribution": dict(
            sorted(source_occurrence_distribution.items())
        ),
        "source_kind_occurrences": dict(sorted(source_kind_occurrences.items())),
        "all_records_gold_graph_reachable": True,
        "oracle_gate": {
            "method": (
                "verified_source_manifest_and_artifact_hashes"
                if trust_source_oracle
                else "exhaustive_final_gold_graph"
            ),
            "non_identity_records": sum(map(len, oracle_groups.values())),
        },
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in artifacts
        },
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    staging.replace(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--enforce-model-max-tokens", action="store_true")
    parser.add_argument("--non-commercial-research", action="store_true")
    parser.add_argument("--workers", type=int, default=min(7, os.cpu_count() or 1))
    parser.add_argument("--trust-source-oracle", action="store_true")
    parser.add_argument(
        "--source-license",
        action="append",
        default=[],
        metavar="SOURCE=SPDX-ID",
    )
    args = parser.parse_args()
    overrides = {}
    for value in args.source_license:
        if "=" not in value:
            parser.error("--source-license must use SOURCE=SPDX-ID")
        source, license_id = value.split("=", 1)
        overrides[source] = license_id
    tokenizer = load_model_tokenizer() if args.enforce_model_max_tokens else None
    manifest = build_training_pool(
        args.inputs,
        args.output,
        tokenizer=tokenizer,
        non_commercial_research=args.non_commercial_research,
        source_licenses=overrides,
        workers=args.workers,
        trust_source_oracle=args.trust_source_oracle,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
