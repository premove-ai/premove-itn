"""Compile real SLURP text into conservative GoldGraph training records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from premove_itn import SpanKind, build_gold_graph, realize_options

try:
    from scripts.build_google_dataset_1 import _quality_issues
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from build_google_dataset_1 import _quality_issues

SOURCE_REVISION = "8eb16545762be97ace75334109d73824217311f1"
SOURCE_FILES = {
    "train": "train.jsonl",
    "validation": "devel.jsonl",
    "test": "test.jsonl",
}
ENTITY_KIND_TIERS = {
    "date": ((SpanKind.DATE,),),
    "time": ((SpanKind.TIME,), (SpanKind.CARDINAL,)),
    "general_frequency": ((SpanKind.MEASUREMENT,),),
    "change_amount": ((SpanKind.CARDINAL,),),
    "email_address": ((SpanKind.ELECTRONIC,),),
}
TIME_DURATION = re.compile(
    r"\b(?:second|seconds|minute|minutes|hour|hours|day|days|week|weeks)\b",
    re.IGNORECASE,
)
TIME_RANGE = re.compile(r"\b(?:and|to|through|until)\b", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _token_offsets(
    sentence: str, tokens: list[dict[str, object]]
) -> list[tuple[int, int]]:
    offsets = []
    cursor = 0
    for expected_id, token in enumerate(tokens):
        if token.get("id") != expected_id or not isinstance(token.get("surface"), str):
            raise ValueError("invalid_token_sequence")
        match = re.search(re.escape(token["surface"]), sentence[cursor:], re.IGNORECASE)
        if match is None:
            raise ValueError("token_alignment")
        start = cursor + match.start()
        end = cursor + match.end()
        if sentence[cursor:start].strip():
            raise ValueError("token_alignment")
        offsets.append((start, end))
        cursor = end
    if sentence[cursor:].strip():
        raise ValueError("token_alignment")
    return offsets


def _replacement(text: str, entity_type: str) -> tuple[str, tuple[str, ...]] | None:
    if entity_type == "time" and TIME_RANGE.search(text):
        return None
    tiers = ENTITY_KIND_TIERS.get(entity_type, ())
    if entity_type == "time" and TIME_DURATION.search(text):
        tiers = ((SpanKind.MEASUREMENT,),)
    if entity_type == "change_amount" and re.search(
        r"\b(?:point|dot)\b", text, re.IGNORECASE
    ):
        tiers = ((SpanKind.DECIMAL,),)
    for kinds in tiers:
        options_by_kind = {
            kind: {
                replacement
                for replacement in realize_options(kind, text)
                if replacement != text
            }
            for kind in kinds
        }
        options = set().union(*options_by_kind.values())
        if len(options) == 1:
            replacement = options.pop()
            replacement_kinds = tuple(
                kind.value
                for kind, kind_options in options_by_kind.items()
                if replacement in kind_options
            )
            return replacement, replacement_kinds
        if options:
            return None
    return None


def _canonicalize(record: dict[str, object]) -> tuple[str, tuple[str, ...]]:
    sentence = record.get("sentence")
    tokens = record.get("tokens")
    entities = record.get("entities")
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("empty_sentence")
    if not isinstance(tokens, list) or not isinstance(entities, list):
        raise ValueError("invalid_schema")
    offsets = _token_offsets(sentence, tokens)
    edits: list[tuple[int, int, str]] = []
    selected_kinds: set[str] = set()
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError("invalid_entity")
        span = entity.get("span")
        entity_type = entity.get("type")
        if (
            not isinstance(span, list)
            or not span
            or not all(isinstance(index, int) for index in span)
            or span != list(range(span[0], span[-1] + 1))
            or span[0] < 0
            or span[-1] >= len(offsets)
            or not isinstance(entity_type, str)
        ):
            raise ValueError("invalid_entity")
        start, end = offsets[span[0]][0], offsets[span[-1]][1]
        result = _replacement(sentence[start:end], entity_type)
        if result is not None:
            replacement, replacement_kinds = result
            edits.append((start, end, replacement))
            selected_kinds.update(replacement_kinds)
    edits.sort()
    if any(left[1] > right[0] for left, right in zip(edits, edits[1:], strict=False)):
        raise ValueError("overlapping_entities")
    expected = sentence
    for start, end, replacement in reversed(edits):
        expected = expected[:start] + replacement + expected[end:]
    quality_issues = (*_quality_issues(sentence), *_quality_issues(expected))
    if quality_issues:
        raise ValueError(quality_issues[0])
    graph = build_gold_graph(sentence, expected)
    if graph is None:
        raise ValueError("oracle_miss")
    kinds = tuple(sorted(selected_kinds))
    return expected, kinds


def build_dataset(source: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    accepted = []
    provenance = []
    quarantine = []
    seen: set[tuple[str, str]] = set()
    source_files = []
    reason_counts: Counter[str] = Counter()
    partition_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    for partition, filename in SOURCE_FILES.items():
        path = source / filename
        source_files.append({"path": filename, "sha256": _sha256(path)})
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record: object = None
                try:
                    record = json.loads(line)
                    expected, kinds = _canonicalize(record)
                    key = (record["sentence"], expected)
                    if key in seen:
                        raise ValueError("duplicate")
                    seen.add(key)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    reason = str(error) if str(error) else type(error).__name__
                    reason_counts[reason] += 1
                    quarantine.append(
                        {
                            "partition": partition,
                            "line_number": line_number,
                            "slurp_id": (
                                record.get("slurp_id")
                                if isinstance(record, dict)
                                else None
                            ),
                            "reason": reason,
                        }
                    )
                    continue
                kind = "KEEP" if not kinds else kinds[0] if len(kinds) == 1 else "MULTI"
                accepted.append(
                    {
                        "text": record["sentence"],
                        "expected_text": expected,
                        "partition": partition,
                        "kind": kind,
                        "kinds": list(kinds),
                    }
                )
                provenance.append(
                    {
                        "partition": partition,
                        "line_number": line_number,
                        "slurp_id": record["slurp_id"],
                        "scenario": record["scenario"],
                        "intent": record["intent"],
                    }
                )
                partition_counts[partition] += 1
                if kinds:
                    kind_counts.update(kinds)
                else:
                    kind_counts["KEEP"] += 1
    dataset_path = output / "dataset.jsonl"
    provenance_path = output / "provenance.jsonl"
    quarantine_path = output / "quarantine.jsonl"
    for path, rows in (
        (dataset_path, accepted),
        (provenance_path, provenance),
        (quarantine_path, quarantine),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "dataset": "SLURP real textual annotations",
        "source_repository": "https://github.com/pswietojanski/slurp.git",
        "source_revision": SOURCE_REVISION,
        "license": "CC-BY-4.0",
        "synthetic_excluded": True,
        "quality_gates": [
            "real_spoken_partition_only",
            "valid_token_and_entity_alignment",
            "unambiguous_rust_realization",
            "high_confidence_english_quality_filter",
            "constructive_gold_derivation_for_every_record",
            "cross_partition_deduplication",
        ],
        "all_records_gold_graph_reachable": True,
        "records": len(accepted),
        "quarantined": len(quarantine),
        "partition_distribution": dict(sorted(partition_counts.items())),
        "distribution": dict(sorted(kind_counts.items())),
        "quarantine_reasons": dict(sorted(reason_counts.items())),
        "source_files": source_files,
        "artifacts": {
            path.name: _sha256(path)
            for path in (dataset_path, provenance_path, quarantine_path)
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
