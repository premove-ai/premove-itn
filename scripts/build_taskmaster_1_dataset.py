"""Compile Taskmaster-1 spoken user turns into conservative ITN records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from premove_itn import SpanKind, build_candidate_graph, build_gold_graph

try:
    from scripts.build_google_dataset_1 import _quality_issues
except ModuleNotFoundError:
    from build_google_dataset_1 import _quality_issues

SOURCE_COMMIT = "d92cb6af3005f1dc09c39e75e7daf4a04905e00b"
SOURCE_LICENSE = "CC-BY-4.0"
SAMPLING_SEED = "premove-itn/taskmaster-1-spoken/v1"
PARTITIONS = ("train", "validation", "test")
SLOT_KIND_SUFFIXES = {
    "date.appt": SpanKind.DATE,
    "num.drink": SpanKind.CARDINAL,
    "num.guests": SpanKind.CARDINAL,
    "num.people": SpanKind.CARDINAL,
    "num.tickets": SpanKind.CARDINAL,
    "price.estimate": SpanKind.MONEY,
    "price.ticket": SpanKind.MONEY,
    "time.appt": SpanKind.TIME,
    "time.dropoff": SpanKind.TIME,
    "time.pickup": SpanKind.TIME,
    "time.reservation": SpanKind.TIME,
    "time.start": SpanKind.TIME,
    "year.vehicle": SpanKind.CARDINAL,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _slot_kind(names: list[str]) -> SpanKind | None:
    kinds = {
        kind
        for name in names
        for suffix, kind in SLOT_KIND_SUFFIXES.items()
        if name.removesuffix(".accept").removesuffix(".reject").endswith(suffix)
    }
    return next(iter(kinds)) if len(kinds) == 1 else None


def _canonicalize_segment(text: str, kind: SpanKind):
    choices: dict[tuple[int, int], set[str]] = {}
    for candidate in build_candidate_graph(text):
        if kind in candidate.kinds:
            choices.setdefault((candidate.char_start, candidate.char_end), set()).add(
                candidate.replacement
            )
    if not choices or any(len(values) != 1 for values in choices.values()):
        return None
    replacements = [
        (start, end, next(iter(values))) for (start, end), values in choices.items()
    ]
    replacements.sort()
    if any(
        left[1] > right[0]
        for left, right in zip(replacements, replacements[1:], strict=False)
    ):
        return None
    output = text
    for start, end, replacement in reversed(replacements):
        output = output[:start] + replacement + output[end:]
    return output if output != text else None


def _compile_utterance(utterance: dict[str, object]):
    text = utterance.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "invalid_text"
    issues = _quality_issues(text)
    if issues:
        return None, issues[0]
    edits: list[tuple[int, int, str, SpanKind, str]] = []
    for segment in utterance.get("segments", []):
        names = [item.get("name", "") for item in segment.get("annotations", [])]
        kind = _slot_kind(names)
        if kind is None:
            continue
        start, end = segment.get("start_index"), segment.get("end_index")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(text)
        ):
            return None, "invalid_segment_bounds"
        source = text[start:end]
        if segment.get("text") != source:
            return None, "segment_text_mismatch"
        target = _canonicalize_segment(source, kind)
        if target is not None:
            edits.append((start, end, target, kind, names[0]))
    if not edits:
        return None, "no_unambiguous_itn_edit"
    edits.sort()
    if any(left[1] > right[0] for left, right in zip(edits, edits[1:], strict=False)):
        return None, "overlapping_annotated_edits"
    expected = text
    for start, end, replacement, _, _ in reversed(edits):
        expected = expected[:start] + replacement + expected[end:]
    graph = build_gold_graph(text, expected)
    if graph is None:
        return None, "unreachable_expected_text"
    kinds = sorted({kind.value for *_, kind, _ in edits})
    return {
        "text": text,
        "expected_text": expected,
        "kind": kinds[0] if len(kinds) == 1 else "MULTI",
        "kinds": kinds,
        "edits": [
            {"start": start, "end": end, "kind": kind.value, "annotation": name}
            for start, end, _, kind, name in edits
        ],
    }, None


def _digest(record: dict[str, object]) -> str:
    value = f"{SAMPLING_SEED}\0{record['text']}\0{record['expected_text']}"
    return hashlib.sha256(value.encode()).hexdigest()


def _partition(digest: str) -> str:
    bucket = int(digest, 16) % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"


def _partition_limit(partition: str, train_limit: int) -> int:
    return train_limit if partition == "train" else max(1, math.ceil(train_limit / 18))


def build_dataset(source: Path, output: Path, *, max_per_kind: int = 100_000):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if max_per_kind < 1:
        raise ValueError("max_per_kind must be at least one")
    dialogs = json.loads(source.read_text(encoding="utf-8"))
    candidates: dict[str, dict[str, object]] = {}
    quarantine: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    processed = 0
    for dialog in dialogs:
        for utterance in dialog.get("utterances", []):
            if utterance.get("speaker") != "USER":
                continue
            processed += 1
            record, reason = _compile_utterance(utterance)
            provenance = {
                "conversation_id": dialog.get("conversation_id"),
                "utterance_index": utterance.get("index"),
            }
            if record is None:
                reasons[reason] += 1
                if len(quarantine) < 1000:
                    quarantine.append({"reason": reason, **provenance})
                continue
            digest = _digest(record)
            record["digest"] = digest
            record["partition"] = _partition(digest)
            record["provenance"] = provenance
            candidates.setdefault(digest, record)

    selected: set[str] = set()
    quota_selected: Counter[str] = Counter()
    for partition in PARTITIONS:
        for kind in (item.value for item in SpanKind):
            matches = sorted(
                (
                    digest
                    for digest, row in candidates.items()
                    if row["partition"] == partition and kind in row["kinds"]
                )
            )[: _partition_limit(partition, max_per_kind)]
            selected.update(matches)
            quota_selected[f"{partition}:{kind}"] += len(matches)

    output.mkdir(parents=True)
    distribution: Counter[str] = Counter()
    kind_occurrences: Counter[str] = Counter()
    partition_distribution: Counter[str] = Counter()
    dataset_lines, provenance_lines = [], []
    for index, digest in enumerate(
        sorted(
            selected,
            key=lambda key: (PARTITIONS.index(candidates[key]["partition"]), key),
        )
    ):
        row = candidates[digest]
        assert build_gold_graph(row["text"], row["expected_text"]) is not None
        public = {
            key: row[key]
            for key in ("text", "expected_text", "partition", "kind", "kinds")
        }
        dataset_lines.append(json.dumps(public, ensure_ascii=False))
        provenance_lines.append(
            json.dumps(
                {"record_index": index, **row["provenance"], "edits": row["edits"]},
                ensure_ascii=False,
            )
        )
        distribution[row["kind"]] += 1
        partition_distribution[row["partition"]] += 1
        kind_occurrences.update(row["kinds"])
    (output / "dataset.jsonl").write_text(
        "\n".join(dataset_lines) + ("\n" if dataset_lines else ""), encoding="utf-8"
    )
    (output / "provenance.jsonl").write_text(
        "\n".join(provenance_lines) + ("\n" if provenance_lines else ""),
        encoding="utf-8",
    )
    (output / "quarantine.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in quarantine)
        + ("\n" if quarantine else ""),
        encoding="utf-8",
    )
    manifest = {
        "kind": "taskmaster_1_spoken",
        "schema_version": 1,
        "source": {
            "commit": SOURCE_COMMIT,
            "sha256": _sha256(source),
            "license": SOURCE_LICENSE,
            "file": source.name,
        },
        "selection": {
            "seed": SAMPLING_SEED,
            "partitions": {"train": 90, "validation": 5, "test": 5},
            "train_max_per_kind": max_per_kind,
        },
        "quality_gates": [
            "spoken_user_turn_only",
            "valid_character_annotations",
            "unambiguous_rust_realization",
            "high_confidence_english_quality_filter",
            "constructive_gold_derivation_for_every_selected_record",
        ],
        "utterances_processed": processed,
        "eligible_unique": len(candidates),
        "records": len(selected),
        "all_selected_oracle_reachable": True,
        "quarantine_reasons": dict(sorted(reasons.items())),
        "distribution": dict(sorted(distribution.items())),
        "kind_occurrences": dict(sorted(kind_occurrences.items())),
        "partition_distribution": dict(sorted(partition_distribution.items())),
        "quota_selected": dict(sorted(quota_selected.items())),
    }
    artifacts = (
        output / "dataset.jsonl",
        output / "provenance.jsonl",
        output / "quarantine.jsonl",
    )
    manifest["artifacts"] = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in artifacts
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-per-kind", type=int, default=100_000)
    args = parser.parse_args()
    print(
        json.dumps(
            build_dataset(args.source, args.output, max_per_kind=args.max_per_kind),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
