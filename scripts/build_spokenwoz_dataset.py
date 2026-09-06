"""Compile official SpokenWOZ text releases into GoldGraph-reachable records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from premove_itn import build_gold_graph, realize_options
from premove_itn.labels import SPAN_KINDS

try:
    from scripts.build_google_dataset_1 import _quality_issues
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from build_google_dataset_1 import _quality_issues

SOURCE_URLS = {
    "train_dev": "https://huggingface.co/datasets/ssz1111/SpokenWOZ-Train-Text",
    "test": "https://huggingface.co/datasets/ssz1111/SpokenWOZ-Test-Text-Fixed",
}
SOURCE_REVISIONS = {
    "train_dev": "d3aad10f2e5a37e7e1e84375f0db368b5872a044",
    "test": "d6c2d9e53b1005e327db582d327b69311092eb65",
}
SPACING = re.compile(r"\s+([,.;:!?])")
MULTISPACE = re.compile(r"\s{2,}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_spacing(text: str) -> str:
    return SPACING.sub(r"\1", MULTISPACE.sub(" ", text.strip()))


def _whitespace_spans(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((match.start(), match.end()) for match in re.finditer(r"\S+", text))


def _read_id_list(path: Path) -> set[str]:
    value = path.read_text()
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {line.strip() for line in value.splitlines() if line.strip()}
    if not isinstance(decoded, list) or not all(
        isinstance(item, str) for item in decoded
    ):
        raise ValueError(f"{path} must contain dialogue IDs")
    return set(decoded)


def _derive_target(text: str, span_info: object) -> str:
    """Use only annotated values that a current Rust realizer can construct."""
    if not isinstance(span_info, list):
        raise ValueError("invalid_span_info")
    words = _whitespace_spans(text)
    edits: list[tuple[int, int, str, str]] = []
    for annotation in span_info:
        if not isinstance(annotation, list) or len(annotation) != 5:
            raise ValueError("invalid_span_annotation")
        target, first, last = annotation[2], annotation[3], annotation[4]
        if (
            not isinstance(target, str)
            or not isinstance(first, int)
            or not isinstance(last, int)
        ):
            raise ValueError("invalid_span_annotation")
        if first < 0 or last < first or last >= len(words):
            raise ValueError("span_out_of_bounds")
        annotation_start, annotation_end = words[first][0], words[last][1]
        matches: list[tuple[int, int, str, str]] = []
        for token_start in range(first, last + 1):
            for token_end in range(token_start, last + 1):
                start, end = words[token_start][0], words[token_end][1]
                source = text[start:end]
                for kind in SPAN_KINDS:
                    if target in realize_options(kind, source):
                        matches.append((start, end, target, kind.value))
        if matches:
            # Prefer the widest annotated source phrase, then the declared kind order.
            edits.append(
                max(
                    matches,
                    key=lambda item: (
                        item[1] - item[0],
                        -next(
                            index
                            for index, kind in enumerate(SPAN_KINDS)
                            if kind.value == item[3]
                        ),
                    ),
                )
            )
        elif annotation_start == annotation_end:
            raise AssertionError("unreachable empty annotation")
    edits.sort()
    if any(left[1] > right[0] for left, right in zip(edits, edits[1:], strict=False)):
        raise ValueError("overlapping_itn_spans")
    expected = text
    for start, end, replacement, _ in reversed(edits):
        expected = expected[:start] + replacement + expected[end:]
    return _normalize_spacing(expected)


def _record(
    dialogue_id: str, turn_index: int, turn: object, partition: str
) -> dict[str, object]:
    if not isinstance(turn, dict):
        raise ValueError("invalid_turn")
    if turn.get("tag") != "user":
        raise ValueError("non_user_turn")
    text = turn.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty_text")
    normalized_text = _normalize_spacing(text)
    expected = _derive_target(text, turn.get("span_info"))
    quality_issues = (
        *_quality_issues(normalized_text),
        *_quality_issues(expected),
    )
    if quality_issues:
        raise ValueError(quality_issues[0])
    graph = build_gold_graph(normalized_text, expected)
    if graph is None:
        raise ValueError("oracle_unreachable")
    graph_kinds = tuple(
        kind.value
        for kind in SPAN_KINDS
        if any(kind in edge.candidate.kinds for edge in graph.candidate_transitions)
    )
    kinds = graph_kinds
    return {
        "text": normalized_text,
        "expected_text": expected,
        "partition": partition,
        "kind": "KEEP" if not kinds else kinds[0] if len(kinds) == 1 else "MULTI",
        "kinds": list(kinds),
        "provenance": {"dialogue_id": dialogue_id, "turn_index": turn_index},
    }


def build_dataset(
    train_dev_path: Path,
    validation_ids_path: Path,
    test_path: Path,
    output_directory: Path,
    *,
    max_per_partition_kind: int = 100_000,
) -> dict[str, object]:
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    if max_per_partition_kind < 1:
        raise ValueError("max_per_partition_kind must be at least one")
    validation_ids = _read_id_list(validation_ids_path)
    sources = ((train_dev_path, validation_ids, "train"), (test_path, set(), "test"))
    accepted: dict[tuple[str, str], dict[str, object]] = {}
    quarantine: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    turns_seen = 0
    for source_path, dev_ids, default_partition in sources:
        dialogues = json.loads(source_path.read_text())
        if not isinstance(dialogues, dict):
            raise ValueError(f"{source_path} must contain a JSON object")
        for dialogue_id in sorted(dialogues):
            dialogue = dialogues[dialogue_id]
            partition = "validation" if dialogue_id in dev_ids else default_partition
            log = dialogue.get("log") if isinstance(dialogue, dict) else None
            if not isinstance(log, list):
                reasons["invalid_dialogue"] += 1
                continue
            for turn_index, turn in enumerate(log):
                turns_seen += 1
                try:
                    row = _record(dialogue_id, turn_index, turn, partition)
                except ValueError as error:
                    reason = str(error)
                    reasons[reason] += 1
                    if reason != "non_user_turn" and len(quarantine) < 500:
                        quarantine.append(
                            {
                                "reason": reason,
                                "dialogue_id": dialogue_id,
                                "turn_index": turn_index,
                            }
                        )
                    continue
                key = (row["text"], row["expected_text"])
                accepted.setdefault(key, row)

    selected: list[dict[str, object]] = []
    counts: Counter[tuple[str, str]] = Counter()
    for _key, row in sorted(
        accepted.items(),
        key=lambda item: hashlib.sha256(
            (item[0][0] + "\0" + item[0][1]).encode()
        ).hexdigest(),
    ):
        quota_kinds = row["kinds"] or ["KEEP"]
        if any(
            counts[(row["partition"], kind)] >= max_per_partition_kind
            for kind in quota_kinds
        ):
            continue
        selected.append(row)
        counts.update((row["partition"], kind) for kind in quota_kinds)
    selected.sort(
        key=lambda row: (
            row["partition"],
            row["provenance"]["dialogue_id"],
            row["provenance"]["turn_index"],
        )
    )
    output_directory.mkdir(parents=True)
    dataset_path = output_directory / "dataset.jsonl"
    dataset_rows = []
    provenance_rows = []
    for record_index, row in enumerate(selected):
        dataset_rows.append(
            {key: value for key, value in row.items() if key != "provenance"}
        )
        provenance_rows.append({"record_index": record_index, **row["provenance"]})
    dataset_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in dataset_rows)
    )
    provenance_path = output_directory / "provenance.jsonl"
    provenance_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in provenance_rows)
    )
    quarantine_path = output_directory / "quarantine.jsonl"
    quarantine_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in quarantine)
    )
    distribution = Counter(row["kind"] for row in selected)
    manifest = {
        "kind": "spokenwoz",
        "schema_version": 1,
        "license": "CC-BY-NC-4.0",
        "quality_gates": [
            "official_user_turn_only",
            "valid_annotated_token_spans",
            "exact_rust_annotation_agreement",
            "non_overlapping_replacements",
            "high_confidence_english_quality_filter",
            "constructive_gold_derivation_for_every_record",
            "cross_partition_deduplication",
        ],
        "all_records_gold_graph_reachable": True,
        "source_revisions": SOURCE_REVISIONS,
        "source_sha256": {
            "train_dev": _sha256(train_dev_path),
            "validation_ids": _sha256(validation_ids_path),
            "test": _sha256(test_path),
        },
        "turns_seen": turns_seen,
        "records": len(selected),
        "distribution": dict(sorted(distribution.items())),
        "partition_kind_distribution": {
            f"{partition}:{kind}": count
            for (partition, kind), count in sorted(counts.items())
        },
        "quarantine_reasons": dict(sorted(reasons.items())),
        "artifact_sha256": {
            "dataset.jsonl": _sha256(dataset_path),
            "provenance.jsonl": _sha256(provenance_path),
            "quarantine.jsonl": _sha256(quarantine_path),
        },
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_dev", type=Path)
    parser.add_argument("validation_ids", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--max-per-partition-kind", type=int, default=100_000)
    args = parser.parse_args()
    print(
        json.dumps(
            build_dataset(
                args.train_dev,
                args.validation_ids,
                args.test,
                args.output_directory,
                max_per_partition_kind=args.max_per_partition_kind,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
