"""Compile Schema-Guided Dialogue turns into oracle-reachable ITN records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from premove_itn import SpanKind, build_gold_graph, realize_options
from premove_itn import representations_equivalent as equivalent

try:
    from scripts.build_google_dataset_1 import _quality_issues
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from build_google_dataset_1 import _quality_issues

SOURCE_REVISION = "e852981ae34990f4358979625854259302feaa78"
SAMPLING_SEED = "premove-itn/sgd/v1"
SPLITS = {"train": "train", "dev": "validation", "test": "test"}
EQUIVALENCE_KINDS = frozenset(
    {
        SpanKind.CARDINAL,
        SpanKind.DATE,
        SpanKind.DECIMAL,
        SpanKind.DIGIT_SEQUENCE,
        SpanKind.MEASUREMENT,
        SpanKind.MONEY,
        SpanKind.ORDINAL,
        SpanKind.PHONE,
        SpanKind.TIME,
    }
)
DEFAULT_MAX_PER_KIND = 100_000


@dataclass(frozen=True, slots=True)
class Replacement:
    start: int
    end: int
    source: str
    target: str
    kind: SpanKind
    service: str
    slot: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_digest(text: str, expected_text: str) -> str:
    return hashlib.sha256(
        f"{SAMPLING_SEED}\0{text}\0{expected_text}".encode()
    ).hexdigest()


def _matching_realization(source: str, canonical: str):
    """Return a deterministic Rust realization agreeing with SGD semantics."""
    matches = []
    for kind in SpanKind:
        for replacement in realize_options(kind, source):
            if replacement == source:
                continue
            if replacement == canonical or (
                kind in EQUIVALENCE_KINDS and equivalent(kind, canonical, replacement)
            ):
                matches.append((kind, replacement))
    if not matches:
        return None
    return min(matches, key=lambda value: (list(SpanKind).index(value[0]), value[1]))


def _compile_turn(
    turn: dict[str, object],
    *,
    split: str,
    shard: str,
    dialogue_id: str,
    turn_index: int,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    text = turn.get("utterance")
    reasons: list[dict[str, object]] = []
    replacements: dict[tuple[int, int], Replacement] = {}
    observed_slots = 0

    if not isinstance(text, str) or not text or "\x00" in text:
        reasons.append({"reason": "invalid_utterance"})
        text = text if isinstance(text, str) else ""

    frames = turn.get("frames")
    if not isinstance(frames, list):
        reasons.append({"reason": "invalid_frames"})
        frames = []

    for frame in frames:
        if not isinstance(frame, dict):
            reasons.append({"reason": "invalid_frame"})
            continue
        service = frame.get("service")
        actions = frame.get("actions")
        slots = frame.get("slots")
        if (
            not isinstance(service, str)
            or not isinstance(actions, list)
            or not isinstance(slots, list)
        ):
            reasons.append({"reason": "invalid_frame"})
            continue

        action_values: dict[str, list[tuple[str, str]]] = {}
        for action in actions:
            if not isinstance(action, dict):
                reasons.append({"reason": "invalid_action"})
                continue
            slot_name = action.get("slot")
            values = action.get("values", [])
            canonical_values = action.get("canonical_values", [])
            if (
                not isinstance(values, list)
                or not isinstance(canonical_values, list)
                or len(values) != len(canonical_values)
            ):
                reasons.append({"reason": "invalid_action_values", "service": service})
                continue
            if not isinstance(slot_name, str):
                continue
            for value, canonical in zip(values, canonical_values, strict=True):
                if isinstance(value, str) and isinstance(canonical, str):
                    action_values.setdefault(slot_name, []).append((value, canonical))
                else:
                    reasons.append(
                        {
                            "reason": "invalid_action_values",
                            "service": service,
                            "slot": slot_name,
                        }
                    )

        for slot_span in slots:
            observed_slots += 1
            if not isinstance(slot_span, dict):
                reasons.append({"reason": "invalid_slot_span", "service": service})
                continue
            slot_name = slot_span.get("slot")
            start = slot_span.get("start")
            end = slot_span.get("exclusive_end")
            if (
                not isinstance(slot_name, str)
                or not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end <= start
                or end > len(text)
            ):
                reasons.append(
                    {
                        "reason": "invalid_slot_span",
                        "service": service,
                        "slot": slot_name,
                    }
                )
                continue
            source = text[start:end]
            pairs = sorted(set(action_values.get(slot_name, ())))
            exact = [
                (value, canonical) for value, canonical in pairs if value == source
            ]
            if not exact:
                reasons.append(
                    {
                        "reason": "slot_action_mismatch",
                        "service": service,
                        "slot": slot_name,
                        "source": source,
                    }
                )
                continue
            canonicals = sorted({canonical for _, canonical in exact})
            if len(canonicals) != 1:
                reasons.append(
                    {
                        "reason": "ambiguous_canonical_value",
                        "service": service,
                        "slot": slot_name,
                        "source": source,
                    }
                )
                continue
            canonical = canonicals[0]
            if canonical == source:
                continue
            match = _matching_realization(source, canonical)
            if match is None:
                reasons.append(
                    {
                        "reason": "unsupported_canonicalization",
                        "service": service,
                        "slot": slot_name,
                        "source": source,
                        "canonical": canonical,
                    }
                )
                continue
            kind, target = match
            replacement = Replacement(
                start, end, source, target, kind, service, slot_name
            )
            previous = replacements.get((start, end))
            if previous is not None and previous != replacement:
                reasons.append(
                    {
                        "reason": "conflicting_frame_annotation",
                        "service": service,
                        "slot": slot_name,
                        "source": source,
                    }
                )
            else:
                replacements[(start, end)] = replacement

    ordered = sorted(replacements.values(), key=lambda item: (item.start, item.end))
    if any(
        left.end > right.start
        for left, right in zip(ordered, ordered[1:], strict=False)
    ):
        reasons.append({"reason": "overlapping_replacements"})

    provenance = {
        "source_split": split,
        "shard": shard,
        "dialogue_id": dialogue_id,
        "turn_index": turn_index,
        "speaker": turn.get("speaker"),
        "services": sorted(
            {
                frame.get("service")
                for frame in frames
                if isinstance(frame, dict) and isinstance(frame.get("service"), str)
            }
        ),
        "annotated_slot_spans": observed_slots,
    }
    audit = {**provenance, "text": text or None, "reasons": reasons}
    if reasons:
        return None, audit

    pieces = []
    cursor = 0
    for replacement in ordered:
        pieces.extend((text[cursor : replacement.start], replacement.target))
        cursor = replacement.end
    pieces.append(text[cursor:])
    expected_text = "".join(pieces)
    quality_issues = dict.fromkeys(
        (*_quality_issues(text), *_quality_issues(expected_text))
    )
    if quality_issues:
        audit["expected_text"] = expected_text
        audit["reasons"] = [{"reason": reason} for reason in quality_issues]
        return None, audit
    kinds = sorted({replacement.kind.value for replacement in ordered})
    kind = "KEEP" if not kinds else kinds[0] if len(kinds) == 1 else "MULTI"
    record = {
        "text": text,
        "expected_text": expected_text,
        "partition": SPLITS[split],
        "kind": kind,
        "kinds": kinds,
    }
    if build_gold_graph(text, expected_text) is None:
        audit["expected_text"] = expected_text
        audit["reasons"] = [{"reason": "gold_graph_unreachable"}]
        return None, audit
    return record, {
        **provenance,
        "replacements": [
            {
                "start": item.start,
                "exclusive_end": item.end,
                "source": item.source,
                "target": item.target,
                "kind": item.kind.value,
                "service": item.service,
                "slot": item.slot,
            }
            for item in ordered
        ],
    }


def build_dataset(
    source_root: Path,
    output_directory: Path,
    *,
    max_per_kind: int = DEFAULT_MAX_PER_KIND,
    quarantine_sample_limit: int = 100,
) -> dict[str, object]:
    if max_per_kind < 1:
        raise ValueError("max_per_kind must be at least one")
    if quarantine_sample_limit < 0:
        raise ValueError("quarantine_sample_limit cannot be negative")
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    inputs = [
        (split, path)
        for split in SPLITS
        for path in sorted((source_root / split).glob("dialogues_*.json"))
    ]
    if not inputs:
        raise ValueError(f"no SGD dialogue shards found under {source_root}")
    output_directory.mkdir(parents=True)

    eligible: list[tuple[str, dict[str, object], dict[str, object]]] = []
    quarantine: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    quarantined_count = 0
    input_reports = []
    turns = 0
    for split, path in inputs:
        dialogues = json.loads(path.read_text())
        if not isinstance(dialogues, list):
            raise ValueError(f"{path}: expected a JSON list")
        input_reports.append(
            {
                "path": str(path.relative_to(source_root)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "dialogues": len(dialogues),
            }
        )
        for dialogue in dialogues:
            dialogue_id = (
                dialogue.get("dialogue_id") if isinstance(dialogue, dict) else None
            )
            dialogue_turns = (
                dialogue.get("turns") if isinstance(dialogue, dict) else None
            )
            if not isinstance(dialogue_id, str) or not isinstance(dialogue_turns, list):
                raise ValueError(f"{path}: invalid dialogue object")
            for turn_index, turn in enumerate(dialogue_turns):
                turns += 1
                if not isinstance(turn, dict):
                    record, audit = (
                        None,
                        {
                            "source_split": split,
                            "shard": path.name,
                            "dialogue_id": dialogue_id,
                            "turn_index": turn_index,
                            "text": None,
                            "reasons": [{"reason": "invalid_turn"}],
                        },
                    )
                else:
                    record, audit = _compile_turn(
                        turn,
                        split=split,
                        shard=path.name,
                        dialogue_id=dialogue_id,
                        turn_index=turn_index,
                    )
                if record is None:
                    quarantined_count += 1
                    names = [reason["reason"] for reason in audit["reasons"]]
                    reasons.update(names)
                    if any(
                        sum(
                            1
                            for row in quarantine
                            for reason in row["reasons"]
                            if reason["reason"] == name
                        )
                        < quarantine_sample_limit
                        for name in names
                    ):
                        quarantine.append(audit)
                else:
                    eligible.append(
                        (
                            _selection_digest(record["text"], record["expected_text"]),
                            record,
                            audit,
                        )
                    )

    # Deduplicate globally before quota selection. A pair present in two official
    # splits is assigned to the earliest split to prevent evaluation leakage.
    partition_order = {"train": 0, "validation": 1, "test": 2}
    unique: dict[tuple[str, str], tuple[str, dict[str, object], dict[str, object]]] = {}
    duplicates = []
    for item in sorted(
        eligible, key=lambda value: (partition_order[value[1]["partition"]], value[0])
    ):
        key = (item[1]["text"], item[1]["expected_text"])
        if key in unique:
            duplicates.append(item[2])
        else:
            unique[key] = item

    limits = {
        "train": max_per_kind,
        "validation": max(1, (max_per_kind + 17) // 18),
        "test": max(1, (max_per_kind + 17) // 18),
    }
    quota_counts: Counter[str] = Counter()
    selected = []
    for digest, record, provenance in sorted(
        unique.values(),
        key=lambda value: (partition_order[value[1]["partition"]], value[0]),
    ):
        quota_kinds = record["kinds"] or ["KEEP"]
        if any(
            quota_counts[f"{record['partition']}:{kind}"] < limits[record["partition"]]
            for kind in quota_kinds
        ):
            selected.append((digest, record, provenance))
            for kind in quota_kinds:
                quota_counts[f"{record['partition']}:{kind}"] += 1

    distribution: Counter[str] = Counter()
    partition_distribution: Counter[str] = Counter()
    kind_occurrences: Counter[str] = Counter()
    with (
        (output_directory / "dataset.jsonl").open(
            "w", encoding="utf-8"
        ) as dataset_handle,
        (output_directory / "provenance.jsonl").open(
            "w", encoding="utf-8"
        ) as provenance_handle,
    ):
        for index, (_, record, provenance) in enumerate(selected):
            dataset_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            provenance_handle.write(
                json.dumps({"record_index": index, **provenance}, ensure_ascii=False)
                + "\n"
            )
            distribution[record["kind"]] += 1
            partition_distribution[record["partition"]] += 1
            kind_occurrences.update(record["kinds"] or ["KEEP"])
    for name, rows in (
        ("quarantine.jsonl", quarantine),
        ("duplicates.jsonl", duplicates),
    ):
        with (output_directory / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "kind": "schema_guided_dialogue",
        "schema_version": 1,
        "source": {
            "repository": "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue",
            "revision": SOURCE_REVISION,
            "license": "CC-BY-SA-4.0",
        },
        "selection": {
            "method": "official_split_then_lowest_seeded_sha256_per_contained_kind",
            "seed": SAMPLING_SEED,
            "train_max_per_kind": max_per_kind,
            "validation_max_per_kind": limits["validation"],
            "test_max_per_kind": limits["test"],
            "unique_by": ["text", "expected_text"],
            "partitions": SPLITS,
        },
        "quality_gates": [
            "valid_dialogue_and_turn_schema",
            "exact_slot_span_to_action_value_alignment",
            "unambiguous_canonical_value",
            "rust_sgd_semantic_agreement",
            "non_overlapping_replacements",
            "constructive_gold_derivation",
            "high_confidence_english_quality_filter",
            "cross_partition_deduplication",
        ],
        "inputs": input_reports,
        "turns_processed": turns,
        "eligible_before_deduplication": len(eligible),
        "duplicates": len(duplicates),
        "quarantined": quarantined_count,
        "quarantine_reasons": dict(sorted(reasons.items())),
        "records": len(selected),
        "distribution": dict(sorted(distribution.items())),
        "partition_distribution": dict(sorted(partition_distribution.items())),
        "kind_occurrences": dict(sorted(kind_occurrences.items())),
        "quota_selected": dict(sorted(quota_counts.items())),
        "all_selected_oracle_reachable": True,
    }
    artifacts = [
        output_directory / name
        for name in (
            "dataset.jsonl",
            "provenance.jsonl",
            "quarantine.jsonl",
            "duplicates.jsonl",
        )
    ]
    manifest["artifacts"] = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in artifacts
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--max-per-kind", type=int, default=DEFAULT_MAX_PER_KIND)
    parser.add_argument("--quarantine-sample-limit", type=int, default=100)
    args = parser.parse_args()
    manifest = build_dataset(
        args.source_root,
        args.output_directory,
        max_per_kind=args.max_per_kind,
        quarantine_sample_limit=args.quarantine_sample_limit,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
