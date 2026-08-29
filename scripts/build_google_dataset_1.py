"""Compile Google TN CSV shards into an oracle-reachable Dataset 1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from premove_itn import (
    SpanKind,
    realize_options,
)
from premove_itn import representations_equivalent as equivalent

HEADER = ("Semiotic Class", "Input Token", "Output Token")
EOS = "<eos>"
SELF = "<self>"
SILENCE = "sil"
CONTEXT_CLASSES = frozenset({"PLAIN", "PUNCT"})
GOOGLE_KIND_OPTIONS = {
    "CARDINAL": (SpanKind.CARDINAL,),
    "DATE": (SpanKind.DATE,),
    "DECIMAL": (SpanKind.DECIMAL,),
    "DIGIT": (SpanKind.DIGIT_SEQUENCE,),
    "ELECTRONIC": (SpanKind.ELECTRONIC,),
    "LETTERS": (SpanKind.WHITELIST, SpanKind.WORD),
    "MEASURE": (SpanKind.MEASUREMENT,),
    "MONEY": (SpanKind.MONEY,),
    "ORDINAL": (SpanKind.ORDINAL,),
    "PUNCTUATION": (SpanKind.PUNCTUATION,),
    "TELEPHONE": (SpanKind.PHONE,),
    "TIME": (SpanKind.TIME,),
    "VERBATIM": (SpanKind.PUNCTUATION,),
    "WHITELIST": (SpanKind.WHITELIST,),
    "WORD": (SpanKind.WORD,),
}
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
NO_SPACE_BEFORE = frozenset({".", ",", ";", ":", "!", "?", "%", ")", "]", "}"})
NO_SPACE_AFTER = frozenset({"(", "[", "{"})
FUNCTION_WORD_DUPLICATE = re.compile(
    r"\b(the|a|an|of|in|to|for|with|on|at|from|by|and|or|is|are|was|were)"
    r"\s+\1\b",
    re.IGNORECASE,
)
INVALID_GRAMMAR_SEQUENCE = re.compile(
    r"\bthe\s+(?:is|are|was|were|be|been|being)\b"
    r"|\b(?:of|in|to|for|with|on|at|from|by)\s+the\s+"
    r"(?:is|are|was|were|be|been|being)\b",
    re.IGNORECASE,
)
INVALID_CURRENCY_PLACEMENT = re.compile(r"\b\d+(?:[.,]\d+)?\s*[$€£](?!\w)")
ADJACENT_PUNCTUATION = re.compile(r"[,;:]\s*[.!?,;:]")
DOUBLE_PERIOD = re.compile(r"(?<!\.)\.\.(?!\.)")
CONCATENATED_FUNCTION_WORD = re.compile(r"(?<=[a-z])(?:The|As)(?=$|[\s,.;:!?])")
SPLIT_YEAR = re.compile(r"(?<!\d)(?:19|20)\s+\d{2}(?!\d)")
LEADING_STRAY_APOSTROPHE = re.compile(r"^['’]\s+")
DEFAULT_MAX_PER_KIND = 100_000
SAMPLING_SEED = "premove-itn/google-tn-dataset-1/v1"
SHARD_NUMBER_PATTERN = re.compile(r"(\d+)(?=\.[^.]+$)")
PARTITIONS = ("train", "validation", "test")
PARTITION_ORDER = {name: index for index, name in enumerate(PARTITIONS)}


@dataclass(frozen=True, slots=True)
class SourceRow:
    record_number: int
    source_class: str
    written: str
    spoken: str


@dataclass(frozen=True, slots=True)
class SourceSentence:
    shard: str
    sentence_number: int
    rows: tuple[SourceRow, ...]
    structural_errors: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class PartReport:
    shard: str
    source_sha256: str
    sentences: int
    eligible: int
    reservoir_records: int
    quarantined: int
    eligible_kind_occurrences: dict[str, int]
    eligible_partition_kind_occurrences: dict[str, int]
    rejected_kind_occurrences: dict[str, int]
    rejection_reason_kind_occurrences: dict[str, int]
    reasons: dict[str, int]
    accepted_path: str
    quarantine_path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_sentences(path: Path, limit: int | None = None):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = tuple(next(reader, ()))
        if header != HEADER:
            raise ValueError(f"{path}: expected header {HEADER!r}; got {header!r}")

        rows: list[SourceRow] = []
        structural_errors: list[dict[str, object]] = []
        sentence_number = 0
        while True:
            try:
                fields = next(reader)
            except StopIteration:
                break
            except csv.Error as error:
                structural_errors.append(
                    {
                        "reason": "malformed_csv_record",
                        "record_number": reader.line_num,
                        "detail": str(error),
                    }
                )
                continue

            record_number = reader.line_num
            if len(fields) != 3:
                structural_errors.append(
                    {
                        "reason": "malformed_csv_record",
                        "record_number": record_number,
                        "field_count": len(fields),
                    }
                )
                if fields and fields[0].strip().lower() == EOS:
                    sentence_number += 1
                    yield SourceSentence(
                        path.name,
                        sentence_number,
                        tuple(rows),
                        tuple(structural_errors),
                    )
                    rows.clear()
                    structural_errors.clear()
                    if limit is not None and sentence_number >= limit:
                        return
                continue
            source_class, written, spoken = fields
            if source_class == EOS:
                if written != EOS or spoken:
                    structural_errors.append(
                        {
                            "reason": "malformed_eos_record",
                            "record_number": record_number,
                        }
                    )
                sentence_number += 1
                yield SourceSentence(
                    path.name,
                    sentence_number,
                    tuple(rows),
                    tuple(structural_errors),
                )
                rows.clear()
                structural_errors.clear()
                if limit is not None and sentence_number >= limit:
                    return
                continue
            rows.append(SourceRow(record_number, source_class, written, spoken))

        if rows or structural_errors:
            structural_errors.append(
                {"reason": "missing_eos", "record_number": reader.line_num}
            )
            yield SourceSentence(
                path.name,
                sentence_number + 1,
                tuple(rows),
                tuple(structural_errors),
            )


def _needs_space(previous: str | None, current: str) -> bool:
    return (
        previous is not None
        and current not in NO_SPACE_BEFORE
        and previous not in NO_SPACE_AFTER
    )


def _join_pieces(pieces: list[str]) -> str:
    output: list[str] = []
    previous = None
    quote_open = False
    for raw_piece in pieces:
        piece = raw_piece.strip()
        if not piece:
            continue
        needs_space = _needs_space(previous, piece)
        if (piece == '"' and quote_open) or (quote_open and previous == '"'):
            needs_space = False
        if needs_space:
            output.append(" ")
        output.append(piece)
        previous = piece
        if piece == '"':
            quote_open = not quote_open
    return "".join(output)


def _quality_issues(text: str) -> tuple[str, ...]:
    """Return only high-confidence sentence quality failures."""
    issues: list[str] = []
    if FUNCTION_WORD_DUPLICATE.search(text):
        issues.append("duplicate_function_word")
    if INVALID_GRAMMAR_SEQUENCE.search(text):
        issues.append("invalid_grammar_sequence")
    if INVALID_CURRENCY_PLACEMENT.search(text):
        issues.append("invalid_currency_placement")
    if ADJACENT_PUNCTUATION.search(text):
        issues.append("adjacent_punctuation")
    if DOUBLE_PERIOD.search(text):
        issues.append("double_period")
    if CONCATENATED_FUNCTION_WORD.search(text):
        issues.append("concatenated_function_word")
    if SPLIT_YEAR.search(text):
        issues.append("split_year")
    if LEADING_STRAY_APOSTROPHE.search(text):
        issues.append("leading_stray_apostrophe")
    if text.count('"') % 2 or any(
        text.count(opening) != text.count(closing)
        for opening, closing in (("(", ")"), ("[", "]"), ("{", "}"))
    ):
        issues.append("unbalanced_delimiter")
    return tuple(issues)


def _matching_realization(
    kinds: tuple[SpanKind, ...], written: str, spoken: str
) -> tuple[SpanKind | None, str | None, bool]:
    produced_options = False
    for kind in kinds:
        options = realize_options(kind, spoken)
        produced_options |= bool(options)
        for replacement in options:
            if replacement == written:
                return kind, replacement, True
            if kind in EQUIVALENCE_KINDS and equivalent(kind, written, replacement):
                return kind, replacement, True
    return None, None, produced_options


def _compile_sentence(
    sentence: SourceSentence,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    reasons = list(sentence.structural_errors)
    source_pieces: list[str] = []
    target_pieces: list[str] = []
    google_target_pieces: list[str] = []
    sentence_kinds: set[SpanKind] = set()
    observed_kinds: set[SpanKind] = set()
    segments: list[dict[str, str | None]] = []

    if not sentence.rows:
        reasons.append({"reason": "empty_sentence"})

    for row in sentence.rows:
        source_class = row.source_class.strip().upper()
        segment_kind = None
        if (
            not source_class
            or not row.written
            or "\x00" in row.written
            or "\x00" in row.spoken
        ):
            reasons.append(
                {"reason": "malformed_row", "record_number": row.record_number}
            )
            continue

        if source_class == "PLAIN":
            if row.spoken != SELF:
                reasons.append(
                    {
                        "reason": "invalid_plain_output",
                        "record_number": row.record_number,
                        "spoken": row.spoken,
                    }
                )
                continue
            source_piece = target_piece = row.written
        elif source_class == "PUNCT":
            if row.spoken not in {SILENCE, SELF}:
                reasons.append(
                    {
                        "reason": "invalid_punctuation_output",
                        "record_number": row.record_number,
                        "spoken": row.spoken,
                    }
                )
                continue
            source_piece = target_piece = row.written
        elif source_class in GOOGLE_KIND_OPTIONS:
            candidate_kinds = GOOGLE_KIND_OPTIONS[source_class]
            observed_kinds.update(candidate_kinds)
            if not row.spoken or row.spoken in {SELF, SILENCE}:
                reasons.append(
                    {
                        "reason": "malformed_semantic_output",
                        "record_number": row.record_number,
                    }
                )
                continue
            kind, target_piece, produced_options = _matching_realization(
                candidate_kinds, row.written, row.spoken
            )
            if target_piece is None:
                reason = (
                    "annotation_mismatch" if produced_options else "realizer_rejected"
                )
                reasons.append(
                    {
                        "reason": reason,
                        "record_number": row.record_number,
                        "source_class": source_class,
                        "kinds": [kind.value for kind in candidate_kinds],
                        "written": row.written,
                        "spoken": row.spoken,
                    }
                )
                continue
            source_piece = row.spoken
            assert kind is not None
            sentence_kinds.add(kind)
            segment_kind = kind.value
        else:
            reasons.append(
                {
                    "reason": "unsupported_class",
                    "record_number": row.record_number,
                    "source_class": source_class,
                }
            )
            continue

        source_pieces.append(source_piece)
        target_pieces.append(target_piece)
        google_target_pieces.append(row.written)
        segments.append(
            {
                "source": source_piece,
                "target": target_piece,
                "kind": segment_kind,
            }
        )

    text = _join_pieces(source_pieces)
    expected_text = _join_pieces(target_pieces)
    google_expected_text = _join_pieces(google_target_pieces)
    if not reasons and (not text or not expected_text):
        reasons.append({"reason": "empty_compiled_sentence"})
    if not reasons:
        quality_reasons = dict.fromkeys(
            (*_quality_issues(text), *_quality_issues(expected_text))
        )
        reasons.extend({"reason": reason} for reason in quality_reasons)
    provenance = {
        "shard": sentence.shard,
        "sentence_number": sentence.sentence_number,
        "first_record_number": (
            sentence.rows[0].record_number if sentence.rows else None
        ),
        "last_record_number": (
            sentence.rows[-1].record_number if sentence.rows else None
        ),
    }
    audit = {
        **provenance,
        "text": text or None,
        "expected_text": expected_text or None,
        "google_expected_text": google_expected_text or None,
        "source_kinds": sorted(kind.value for kind in observed_kinds),
        "reasons": reasons,
    }
    if reasons:
        return None, audit
    kinds = sorted(kind.value for kind in sentence_kinds)
    if not kinds:
        distribution_kind = "KEEP"
    elif len(kinds) == 1:
        distribution_kind = kinds[0]
    else:
        distribution_kind = "MULTI"
    return {
        "text": text,
        "expected_text": expected_text,
        "kind": distribution_kind,
        "kinds": kinds,
        "segments": segments,
    }, provenance


def _process_shard(arguments: tuple[str, str, int | None, int, int]) -> PartReport:
    input_name, part_directory_name, limit, max_per_kind, quarantine_sample_limit = (
        arguments
    )
    path = Path(input_name)
    part_directory = Path(part_directory_name)
    accepted_path = part_directory / f"{path.stem}.accepted.jsonl"
    quarantine_path = part_directory / f"{path.stem}.quarantine.jsonl"
    counts: Counter[str] = Counter()
    with accepted_path.open("w", encoding="utf-8") as accepted_handle:
        quarantine_handle = quarantine_path.open("w", encoding="utf-8")
        try:
            _write_shard_records(
                path,
                accepted_handle,
                quarantine_handle,
                counts,
                limit,
                max_per_kind,
                quarantine_sample_limit,
            )
        finally:
            quarantine_handle.close()

    sentences = counts.pop("_sentences")
    eligible = counts.pop("_eligible")
    reservoir_records = counts.pop("_reservoir_records")
    quarantined = counts.pop("_quarantined")
    eligible_kind_occurrences = {
        key.removeprefix("_eligible_kind_occurrences:"): counts.pop(key)
        for key in tuple(counts)
        if key.startswith("_eligible_kind_occurrences:")
    }
    eligible_partition_kind_occurrences = {
        key.removeprefix("_eligible_partition_kind_occurrences:"): counts.pop(key)
        for key in tuple(counts)
        if key.startswith("_eligible_partition_kind_occurrences:")
    }
    rejected_kind_occurrences = {
        key.removeprefix("_rejected_kind_occurrences:"): counts.pop(key)
        for key in tuple(counts)
        if key.startswith("_rejected_kind_occurrences:")
    }
    rejection_reason_kind_occurrences = {
        key.removeprefix("_rejection_reason_kind_occurrences:"): counts.pop(key)
        for key in tuple(counts)
        if key.startswith("_rejection_reason_kind_occurrences:")
    }

    return PartReport(
        path.name,
        _sha256(path),
        sentences,
        eligible,
        reservoir_records,
        quarantined,
        dict(sorted(eligible_kind_occurrences.items())),
        dict(sorted(eligible_partition_kind_occurrences.items())),
        dict(sorted(rejected_kind_occurrences.items())),
        dict(sorted(rejection_reason_kind_occurrences.items())),
        dict(sorted(counts.items())),
        str(accepted_path),
        str(quarantine_path),
    )


def _write_shard_records(
    path: Path,
    accepted_handle,
    quarantine_handle,
    counts: Counter[str],
    limit: int | None,
    max_per_kind: int,
    quarantine_sample_limit: int,
) -> None:
    sentences = eligible = quarantined = 0
    reservoirs: dict[str, list[tuple[int, str, str]]] = {}
    selected_digests: dict[str, set[str]] = {}
    quarantine_samples: Counter[str] = Counter()
    try:
        for sentence in _iter_sentences(path, limit):
            sentences += 1
            record, audit = _compile_sentence(sentence)
            if record is None:
                quarantined += 1
                reasons = [reason["reason"] for reason in audit["reasons"]]
                counts.update(reasons)
                for kind in audit["source_kinds"]:
                    counts[f"_rejected_kind_occurrences:{kind}"] += 1
                    for reason in set(reasons):
                        counts[
                            f"_rejection_reason_kind_occurrences:{reason}:{kind}"
                        ] += 1
                if any(
                    quarantine_samples[reason] < quarantine_sample_limit
                    for reason in reasons
                ):
                    quarantine_handle.write(
                        json.dumps(audit, ensure_ascii=False) + "\n"
                    )
                    quarantine_samples.update(reasons)
            else:
                eligible += 1
                digest = _selection_digest(record)
                partition = _partition_for_digest(digest)
                quota_kinds = record["kinds"] or ["KEEP"]
                for kind in quota_kinds:
                    counts[f"_eligible_kind_occurrences:{kind}"] += 1
                    counts[
                        f"_eligible_partition_kind_occurrences:{partition}:{kind}"
                    ] += 1
                    _offer_reservoir_record(
                        reservoirs,
                        selected_digests,
                        {
                            **record,
                            "partition": partition,
                            "quota_kind": kind,
                            "provenance": audit,
                        },
                        _partition_limit(partition, max_per_kind),
                        f"{partition}:{kind}",
                    )
            if sentences % 100_000 == 0:
                print(
                    f"{path.name}: processed={sentences} "
                    f"eligible={eligible} "
                    f"quarantined={quarantined}",
                    file=sys.stderr,
                    flush=True,
                )

    finally:
        reservoir_records = 0
        for distribution_kind in sorted(reservoirs):
            for _, _, line in sorted(
                reservoirs[distribution_kind], key=lambda item: item[1]
            ):
                accepted_handle.write(line + "\n")
                reservoir_records += 1
        counts["_sentences"] = sentences
        counts["_eligible"] = eligible
        counts["_reservoir_records"] = reservoir_records
        counts["_quarantined"] = quarantined


def _offer_reservoir_record(
    reservoirs: dict[str, list[tuple[int, str, str]]],
    selected_digests: dict[str, set[str]],
    value: dict[str, object],
    limit: int,
    reservoir_key: str,
) -> str:
    digest = _selection_digest(value)
    digests = selected_digests.setdefault(reservoir_key, set())
    if digest in digests:
        return "duplicate"
    line = json.dumps({**value, "selection_rank": digest}, ensure_ascii=False)
    item = (-int(digest, 16), digest, line)
    reservoir = reservoirs.setdefault(reservoir_key, [])
    if len(reservoir) < limit:
        heapq.heappush(reservoir, item)
        digests.add(digest)
        return "selected"
    if item <= reservoir[0]:
        return "discarded"
    removed = heapq.heapreplace(reservoir, item)
    digests.remove(removed[1])
    digests.add(digest)
    return "selected"


def _selection_digest(value: dict[str, object]) -> str:
    canonical = f"{SAMPLING_SEED}\0{value['text']}\0{value['expected_text']}"
    return hashlib.sha256(canonical.encode()).hexdigest()


def _partition_for_digest(digest: str) -> str:
    bucket = int(digest, 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def _partition_limit(partition: str, train_limit: int) -> int:
    if partition == "train":
        return train_limit
    return max(1, math.ceil(train_limit / 18))


def _publish_dataset(
    reports: list[PartReport],
    output_directory: Path,
    max_per_kind: int,
) -> dict[str, object]:
    dataset_path = output_directory / "dataset.jsonl"
    provenance_path = output_directory / "provenance.jsonl"
    quarantine_path = output_directory / "quarantine.jsonl"
    duplicates_path = output_directory / "duplicates.jsonl"

    with dataset_path.open("w", encoding="utf-8") as dataset_handle:
        provenance_handle = provenance_path.open("w", encoding="utf-8")
        quarantine_handle = quarantine_path.open("w", encoding="utf-8")
        duplicates_handle = duplicates_path.open("w", encoding="utf-8")
        try:
            merge_report = _merge_parts(
                reports,
                dataset_handle,
                provenance_handle,
                quarantine_handle,
                duplicates_handle,
                max_per_kind,
            )
        finally:
            provenance_handle.close()
            quarantine_handle.close()
            duplicates_handle.close()
    reason_counts: Counter[str] = Counter()
    eligible_kind_occurrences: Counter[str] = Counter()
    eligible_partition_kind_occurrences: Counter[str] = Counter()
    rejected_kind_occurrences: Counter[str] = Counter()
    rejection_reason_kind_occurrences: Counter[str] = Counter()
    for report in reports:
        reason_counts.update(report.reasons)
        eligible_kind_occurrences.update(report.eligible_kind_occurrences)
        eligible_partition_kind_occurrences.update(
            report.eligible_partition_kind_occurrences
        )
        rejected_kind_occurrences.update(report.rejected_kind_occurrences)
        rejection_reason_kind_occurrences.update(
            report.rejection_reason_kind_occurrences
        )
    input_reports = []
    for report in reports:
        value = asdict(report)
        value.pop("accepted_path")
        value.pop("quarantine_path")
        input_reports.append(value)
    artifacts = (dataset_path, provenance_path, quarantine_path, duplicates_path)
    manifest = {
        "kind": "google_tn_dataset_1",
        "schema_version": 1,
        "selection": {
            "method": "lowest_seeded_sha256_per_partition_and_contained_kind",
            "seed": SAMPLING_SEED,
            "train_max_per_kind": max_per_kind,
            "validation_max_per_kind": _partition_limit("validation", max_per_kind),
            "test_max_per_kind": _partition_limit("test", max_per_kind),
            "unique_by": ["text", "expected_text"],
            "partitions": {"train": 90, "validation": 5, "test": 5},
        },
        "quality_gates": [
            "structurally_valid_csv_sentence",
            "supported_source_classes_only",
            "valid_context_annotations",
            "rust_google_semantic_agreement",
            "high_confidence_english_quality_filter",
            "constructive_gold_derivation_for_every_selected_record",
        ],
        "inputs": input_reports,
        "sentences_processed": sum(report.sentences for report in reports),
        "eligible_before_oracle": sum(report.eligible for report in reports),
        "part_reservoir_records": sum(report.reservoir_records for report in reports),
        "all_selected_oracle_reachable": True,
        "quarantined": sum(report.quarantined for report in reports),
        "quarantine_reasons": dict(sorted(reason_counts.items())),
        "eligible_kind_occurrences": dict(sorted(eligible_kind_occurrences.items())),
        "eligible_partition_kind_occurrences": dict(
            sorted(eligible_partition_kind_occurrences.items())
        ),
        "rejected_kind_occurrences": dict(sorted(rejected_kind_occurrences.items())),
        "rejection_reason_kind_occurrences": dict(
            sorted(rejection_reason_kind_occurrences.items())
        ),
        "kind_report": _build_kind_report(
            max_per_kind,
            eligible_kind_occurrences,
            eligible_partition_kind_occurrences,
            rejected_kind_occurrences,
            rejection_reason_kind_occurrences,
            merge_report,
        ),
        **merge_report,
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in artifacts
        },
    }
    return manifest


def _build_kind_report(
    max_per_kind: int,
    eligible: Counter[str],
    eligible_by_partition: Counter[str],
    rejected: Counter[str],
    rejection_reasons: Counter[str],
    merge_report: dict[str, object],
) -> dict[str, object]:
    quota_selected = merge_report["quota_selected"]
    final_occurrences = merge_report["partition_kind_occurrences"]
    report = {}
    for kind in ("KEEP", *(span_kind.value for span_kind in SpanKind)):
        reasons = {
            key.split(":", 1)[0]: count
            for key, count in rejection_reasons.items()
            if key.endswith(f":{kind}")
        }
        report[kind] = {
            "requested": {
                partition: _partition_limit(partition, max_per_kind)
                for partition in PARTITIONS
            },
            "eligible_occurrences_before_oracle": eligible[kind],
            "rejected_sentence_occurrences": rejected[kind],
            "rejections_by_reason": dict(sorted(reasons.items())),
            "eligible_occurrences_by_partition": {
                partition: eligible_by_partition[f"{partition}:{kind}"]
                for partition in PARTITIONS
            },
            "selected_for_quota": {
                partition: quota_selected.get(f"{partition}:{kind}", 0)
                for partition in PARTITIONS
            },
            "final_unique_occurrences": {
                partition: final_occurrences.get(f"{partition}:{kind}", 0)
                for partition in PARTITIONS
            },
        }
    return report


def _merge_parts(
    reports,
    dataset_handle,
    provenance_handle,
    quarantine_handle,
    duplicates_handle,
    max_per_kind: int,
) -> dict[str, object]:
    duplicates = 0
    reservoirs: dict[str, list[tuple[int, str, str]]] = {}
    selected_digests: dict[str, set[str]] = {}
    for report in reports:
        with Path(report.quarantine_path).open(encoding="utf-8") as handle:
            for line in handle:
                quarantine_handle.write(line)
        with Path(report.accepted_path).open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                reservoir_key = f"{value['partition']}:{value['quota_kind']}"
                result = _offer_reservoir_record(
                    reservoirs,
                    selected_digests,
                    value,
                    _partition_limit(value["partition"], max_per_kind),
                    reservoir_key,
                )
                if result == "duplicate":
                    duplicates += 1
                    duplicates_handle.write(line)

    quota_selected: Counter[str] = Counter()
    selected_by_digest: dict[str, dict[str, object]] = {}
    for reservoir_key in sorted(reservoirs):
        for _, _, line in sorted(reservoirs[reservoir_key], key=lambda item: item[1]):
            value = json.loads(line)
            quota_selected[reservoir_key] += 1
            digest = _selection_digest(value)
            existing = selected_by_digest.get(digest)
            if existing is None:
                selected_by_digest[digest] = value
                continue
            combined_kinds = sorted(set(existing["kinds"]) | set(value["kinds"]))
            existing["kinds"] = combined_kinds
            existing["kind"] = (
                combined_kinds[0] if len(combined_kinds) == 1 else "MULTI"
            )

    distribution: Counter[str] = Counter()
    kind_occurrences: Counter[str] = Counter()
    partition_distribution: Counter[str] = Counter()
    partition_kind_occurrences: Counter[str] = Counter()
    ordered_records = sorted(
        selected_by_digest.items(),
        key=lambda item: (
            PARTITION_ORDER[item[1]["partition"]],
            item[0],
        ),
    )
    for _, value in ordered_records:
        if not _verify_constructive_gold_record(value):
            raise RuntimeError(
                "selected record failed the final constructive gold audit: "
                f"{value['text']!r} -> {value['expected_text']!r}"
            )

    for record_index, (_, value) in enumerate(ordered_records):
        quota_kinds = value["kinds"] or ["KEEP"]
        for kind in quota_kinds:
            kind_occurrences[kind] += 1
            partition_kind_occurrences[f"{value['partition']}:{kind}"] += 1
        distribution[value["kind"]] += 1
        partition_distribution[value["partition"]] += 1
        dataset_handle.write(
            json.dumps(
                {
                    "text": value["text"],
                    "expected_text": value["expected_text"],
                    "partition": value["partition"],
                    "kind": value["kind"],
                    "kinds": value["kinds"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        provenance_handle.write(
            json.dumps(
                {"record_index": record_index, **value["provenance"]},
                ensure_ascii=False,
            )
            + "\n"
        )

    return {
        "records": len(ordered_records),
        "selected_oracle_checked": len(ordered_records),
        "reservoir_duplicates": duplicates,
        "quota_selected": dict(sorted(quota_selected.items())),
        "distribution": dict(sorted(distribution.items())),
        "partition_distribution": dict(sorted(partition_distribution.items())),
        "kind_occurrences": dict(sorted(kind_occurrences.items())),
        "partition_kind_occurrences": dict(sorted(partition_kind_occurrences.items())),
    }


def _verify_constructive_gold_record(value: dict[str, object]) -> bool:
    segments = value["segments"]
    source = _join_pieces([segment["source"] for segment in segments])
    target = _join_pieces([segment["target"] for segment in segments])
    if source != value["text"] or target != value["expected_text"]:
        return False
    for segment in segments:
        segment_kind = segment["kind"]
        if segment_kind is None:
            if segment["source"] != segment["target"]:
                return False
            continue
        kind = SpanKind(segment_kind)
        options = realize_options(kind, segment["source"])
        if segment["target"] in options:
            continue
        if kind not in EQUIVALENCE_KINDS or not any(
            equivalent(kind, segment["target"], option) for option in options
        ):
            return False
    return True


def build_dataset(
    inputs: list[Path],
    output_directory: Path,
    *,
    workers: int,
    limit_per_shard: int | None = None,
    max_per_kind: int = DEFAULT_MAX_PER_KIND,
    quarantine_sample_limit: int = 100,
) -> dict[str, object]:
    if not inputs:
        raise ValueError("at least one input shard is required")
    if workers < 1:
        raise ValueError("workers must be at least one")
    if max_per_kind < 1:
        raise ValueError("max_per_kind must be at least one")
    if quarantine_sample_limit < 0:
        raise ValueError("quarantine_sample_limit cannot be negative")
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    output_directory.mkdir(parents=True)
    part_directory = output_directory / "parts"
    part_directory.mkdir()
    ordered_inputs = sorted(inputs, key=_shard_order)
    arguments = [
        (
            str(path),
            str(part_directory),
            limit_per_shard,
            max_per_kind,
            quarantine_sample_limit,
        )
        for path in ordered_inputs
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        reports = list(executor.map(_process_shard, arguments))
    manifest = _publish_dataset(reports, output_directory, max_per_kind)
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    shutil.rmtree(part_directory)
    return manifest


def _shard_order(path: Path) -> tuple[int, str]:
    match = SHARD_NUMBER_PATTERN.search(path.name)
    return (int(match.group(1)) if match else sys.maxsize, path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--workers", type=int, default=min(7, os.cpu_count() or 1))
    parser.add_argument("--limit-per-shard", type=int)
    parser.add_argument("--max-per-kind", type=int, default=DEFAULT_MAX_PER_KIND)
    parser.add_argument("--quarantine-sample-limit", type=int, default=100)
    args = parser.parse_args()
    manifest = build_dataset(
        args.inputs,
        args.output_directory,
        workers=args.workers,
        limit_per_shard=args.limit_per_shard,
        max_per_kind=args.max_per_kind,
        quarantine_sample_limit=args.quarantine_sample_limit,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
