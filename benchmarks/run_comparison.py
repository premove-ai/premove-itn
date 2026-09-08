"""Run the frozen VoiceAgent ITN benchmark against three backends.

The headline adapters receive only the dataset's ``text`` field.  The script
stores one JSON record per input, aggregate JSON metrics, and a human-readable
report.  Thutmose runs in a separate interpreter because its official 1.9
artifact is not importable from the current NeMo 2.x installation.
"""

# The report embeds wide Markdown tables and intentionally keeps those rows on
# one line. Runtime code remains formatted by Ruff.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval/voice_agent_itn/voice_agent_eval.jsonl"
DEFAULT_THUTMOSE_ARTIFACT = (
    Path.home() / "Documents/Git Repositories/premove/.artifacts/thutmose-cache/nemo/"
    "itn_en_thutmose_bert/bf3a085f3b78525c3b3abe09bd0c62c8/"
    "itn_en_thutmose_bert.nemo"
)
DEFAULT_THUTMOSE_PYTHON = (
    Path.home() / "Documents/Git Repositories/premove/.venv-thutmose/bin/python"
)
SURFACE_TOKEN = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)


def _surface_tokens(text: str) -> tuple[str, ...]:
    """Tokenize only for context alignment, never for semantic values."""
    return tuple(
        token.casefold().replace("’", "'").replace("“", '"').replace("”", '"')
        for token in SURFACE_TOKEN.findall(text)
    )


def _tokens_match(left: tuple[str, ...], right: tuple[str, ...], start: int) -> bool:
    return tuple(right[start : start + len(left)]) == left


def _value_variants(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Offer spacing variants to the semantic parser, not to exact scoring."""
    spaced = " ".join(tokens)
    compact_punctuation = re.sub(r"\s+([.,:%;!?/@#$\-+])", r"\1", spaced)
    compact = "".join(tokens)
    return tuple(dict.fromkeys((spaced, compact_punctuation, compact)))


def _structured_value(value: str) -> str:
    """Canonicalize formatting that speech cannot encode for structured IDs."""
    return "".join(_surface_tokens(value))


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 1_500:
        raise ValueError(f"expected 1500 benchmark rows, found {len(rows)}")
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("benchmark row IDs are not unique")
    for row in rows:
        if not isinstance(row.get("text"), str) or not isinstance(
            row.get("expected_text"), str
        ):
            raise ValueError(f"row {row.get('id')} has invalid text fields")
        if not isinstance(row.get("spans", []), list):
            raise ValueError(f"row {row.get('id')} has invalid spans")
    return rows


def row_kind(row: dict[str, Any]) -> str:
    spans = row.get("spans", [])
    if not spans:
        return "KEEP"
    if len(spans) > 1:
        return "MULTI"
    return str(spans[0]["category"])


def semantic_kind(row: dict[str, Any], span: dict[str, Any]) -> str | None:
    # The dataset deliberately leaves PERCENT unmapped because Premove has no
    # public PERCENT SpanKind.  Structured and electronic values are scored by
    # their canonical form (case-insensitive for identifiers).
    value = span.get("premove_span_kind")
    if value in {
        "CARDINAL",
        "DATE",
        "DECIMAL",
        "DIGIT_SEQUENCE",
        "MEASUREMENT",
        "MONEY",
        "ORDINAL",
        "PHONE",
        "TIME",
    }:
        return str(value)
    return None


def _expected_entity_parts(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return expected replacements and exact context between replacements."""
    expected = str(row["expected_text"])
    replacements: list[str] = []
    contexts: list[str] = []
    cursor = 0
    for span in sorted(row.get("spans", []), key=lambda item: int(item["start"])):
        replacement = str(span["replacement"])
        index = expected.find(replacement, cursor)
        if index < 0:
            raise ValueError(f"replacement is absent from expected text: {row['id']}")
        contexts.append(expected[cursor:index])
        replacements.append(replacement)
        cursor = index + len(replacement)
    contexts.append(expected[cursor:])
    return replacements, contexts


def _predicted_entity_values(row: dict[str, Any], prediction: str) -> list[str] | None:
    """Split a prediction using a formatting-tolerant context skeleton.

    Context matching is token based. This ignores whitespace, case, and the
    placement of harmless punctuation spaces, while the extracted entity text
    is still passed to the category-aware semantic parser. Exact sentence
    scoring remains a separate, strict metric.
    """
    replacements, contexts = _expected_entity_parts(row)
    if not replacements:
        return [] if prediction == str(row["text"]) else None
    expected_contexts = [_surface_tokens(context) for context in contexts]
    predicted_tokens = _surface_tokens(prediction)

    def search(
        entity_index: int, cursor: int, values: list[tuple[str, ...]]
    ) -> list[tuple[str, ...]] | None:
        context = expected_contexts[entity_index]
        if not _tokens_match(context, predicted_tokens, cursor):
            return None
        cursor += len(context)
        suffix = expected_contexts[entity_index + 1]
        if not suffix:
            if entity_index + 1 != len(replacements):
                return search(entity_index + 1, cursor, values + [()])
            return values + [predicted_tokens[cursor:]]
        for end in range(cursor, len(predicted_tokens) - len(suffix) + 1):
            if not _tokens_match(suffix, predicted_tokens, end):
                continue
            next_values = values + [predicted_tokens[cursor:end]]
            if entity_index + 1 == len(replacements):
                if end + len(suffix) == len(predicted_tokens):
                    return next_values
                continue
            result = search(entity_index + 1, end, next_values)
            if result is not None:
                return result
        return None

    values = search(0, 0, [])
    if values is None or len(values) != len(replacements):
        return None
    return [" ".join(value) for value in values]


def _semantic_equal(
    kind: str | None, expected: str, observed: str, category: str
) -> bool:
    if category == "PERCENT":

        def number(value: str) -> Decimal | None:
            raw = re.sub(r"[^0-9.+-]", "", value)
            try:
                return Decimal(raw)
            except InvalidOperation:
                return None

        expected_number = number(expected)
        observed_number = number(observed)
        return (
            expected_number is not None
            and observed_number is not None
            and expected_number == observed_number
        )
    if category in {
        "EMAIL",
        "FLIGHT_ID",
        "IP",
        "ORDER_ID",
        "REFERENCE_ID",
        "URL",
        "VERSION",
    }:
        return _structured_value(expected) == _structured_value(observed)
    if kind is None:
        return expected == observed
    from premove_itn import SpanKind, representations_equivalent

    try:
        return any(
            representations_equivalent(SpanKind(kind), expected, variant)
            for variant in _value_variants(_surface_tokens(observed))
        )
    except (ValueError, RuntimeError):
        return False


def score_row(row: dict[str, Any], prediction: str) -> dict[str, Any]:
    spans = list(row.get("spans", []))
    expected = str(row["expected_text"])
    source = str(row["text"])
    exact = prediction == expected
    values = _predicted_entity_values(row, prediction)
    entity_results: list[bool] = []
    if values is not None and len(values) == len(spans):
        for span, observed in zip(spans, values, strict=True):
            entity_results.append(
                _semantic_equal(
                    semantic_kind(row, span),
                    str(span["replacement"]),
                    observed,
                    str(span["category"]),
                )
            )
    entity_total = len(spans)
    entity_correct = sum(entity_results)
    positive = bool(spans)
    decision_correct = prediction != source if positive else prediction == source
    if exact:
        error_type = None
    elif not positive:
        error_type = "false_normalization"
    elif prediction == source:
        error_type = "missed_edit"
    else:
        error_type = "wrong_rewrite"
    return {
        "strict_exact": exact,
        "semantic_entities_correct": entity_correct,
        "semantic_entities_total": entity_total,
        "semantic_entity_correct": bool(
            entity_total and entity_correct == entity_total
        ),
        "normalization_decision_correct": decision_correct,
        "error_type": error_type,
        "predicted_entities": values,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def latency_metrics(
    records: list[dict[str, Any]], field: str = "latency_ms"
) -> dict[str, Any]:
    values = [
        float(record[field]) for record in records if record.get(field) is not None
    ]
    if not values:
        return {"count": 0}
    total = sum(values)
    return {
        "count": len(values),
        "total_ms": total,
        "mean_ms": statistics.fmean(values),
        "stddev_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
        "records_per_second": len(values) / (total / 1000) if total else None,
    }


def subset_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"records": 0}
    strict = sum(bool(record["strict_exact"]) for record in records)
    semantic_correct = sum(
        int(record["semantic_entities_correct"]) for record in records
    )
    semantic_total = sum(int(record["semantic_entities_total"]) for record in records)
    entity_rows = [record for record in records if record["semantic_entities_total"]]
    all_entities = sum(
        bool(record["semantic_entity_correct"]) for record in entity_rows
    )
    decisions = sum(
        bool(record["normalization_decision_correct"]) for record in records
    )
    errors = Counter(
        str(record["error_type"])
        for record in records
        if record.get("error_type") is not None
    )
    return {
        "records": len(records),
        "strict_exact_correct": strict,
        "strict_exact_accuracy": strict / len(records),
        "semantic_entity_correct": semantic_correct,
        "semantic_entity_total": semantic_total,
        "semantic_entity_accuracy": (
            semantic_correct / semantic_total if semantic_total else None
        ),
        "semantic_entity_rows": len(entity_rows),
        "all_entities_correct": all_entities,
        "all_entities_accuracy": all_entities / len(entity_rows)
        if entity_rows
        else None,
        "normalization_decision_correct": decisions,
        "normalization_decision_accuracy": decisions / len(records),
        "errors": dict(sorted(errors.items())),
        "latency": latency_metrics(records),
    }


def aggregate_metrics(
    rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, record in zip(rows, records, strict=True):
        by_kind[row_kind(row)].append(record)
        by_group[str(row["group"])].append(record)
        by_domain[str(row["domain"])].append(record)
        by_difficulty[str(row["difficulty"])].append(record)
        for span in row.get("spans", []):
            by_category_entity[str(span["category"])].append(record)

    collisions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, record in zip(rows, records, strict=True):
        pair = row.get("collision_pair_id")
        if pair:
            collisions[str(pair)].append(record)
    collision_pairs = [subset_metrics(items) for items in collisions.values()]
    collision_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, record in zip(rows, records, strict=True):
        family = row.get("collision")
        if family:
            collision_family[str(family)].append(record)

    negative = [
        record
        for row, record in zip(rows, records, strict=True)
        if row["group"] == "negative"
    ]
    no_span = [
        record
        for row, record in zip(rows, records, strict=True)
        if not row.get("spans")
    ]
    voice_agent = [
        record
        for row, record in zip(rows, records, strict=True)
        if row["group"] == "voice_agent"
    ]
    multi = [
        record
        for row, record in zip(rows, records, strict=True)
        if len(row.get("spans", [])) > 1
    ]
    return {
        "overall": subset_metrics(records),
        "per_kind": {key: subset_metrics(by_kind[key]) for key in sorted(by_kind)},
        "per_group": {key: subset_metrics(by_group[key]) for key in sorted(by_group)},
        "per_domain": {
            key: subset_metrics(by_domain[key]) for key in sorted(by_domain)
        },
        "per_difficulty": {
            key: subset_metrics(by_difficulty[key]) for key in sorted(by_difficulty)
        },
        "per_entity_category": {
            key: subset_metrics(by_category_entity[key])
            for key in sorted(by_category_entity)
        },
        "collision": {
            "pairs": len(collision_pairs),
            "pair_metrics": {
                "strict_all_rows_correct": sum(
                    metric["strict_exact_accuracy"] == 1.0 for metric in collision_pairs
                ),
                "semantic_all_rows_correct": sum(
                    metric["all_entities_accuracy"] == 1.0 for metric in collision_pairs
                ),
                "strict_pair_accuracy": (
                    sum(
                        metric["strict_exact_accuracy"] == 1.0
                        for metric in collision_pairs
                    )
                    / len(collision_pairs)
                    if collision_pairs
                    else None
                ),
                "semantic_pair_accuracy": (
                    sum(
                        metric["all_entities_accuracy"] == 1.0
                        for metric in collision_pairs
                    )
                    / len(collision_pairs)
                    if collision_pairs
                    else None
                ),
            },
            "per_family": {
                key: subset_metrics(collision_family[key])
                for key in sorted(collision_family)
            },
        },
        "multi_entity": subset_metrics(multi),
        "voice_agent": subset_metrics(voice_agent),
        "standalone_negative_keep": subset_metrics(negative),
        "all_no_span_keep": subset_metrics(no_span),
    }


class Backend:
    name = ""
    description = ""

    def initialize(self) -> dict[str, Any]:
        return {}

    def warmup(self) -> float | None:
        return None

    def normalize(self, text: str) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class TextProcessingRSBackend(Backend):
    name = "text-processing-rs"
    description = (
        "Compiled upstream English sentence normalizer exposed by premove_itn."
    )

    def initialize(self) -> dict[str, Any]:
        from premove_itn import normalize_sentence

        self._normalize_sentence = normalize_sentence
        return {"runtime": "premove_itn._rust.baseline_normalize_sentence"}

    def warmup(self) -> None:
        self._normalize_sentence("warm up at five thirty")

    def normalize(self, text: str) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter_ns()
        output = self._normalize_sentence(text)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        return output, {"backend_latency_ms": elapsed}


class PremoveITNBackend(Backend):
    name = "premove-itn"
    description = (
        "Production structured-value 20k contextual candidate scorer and decoder."
    )

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint

    def initialize(self) -> dict[str, Any]:
        import torch

        from premove_itn.candidate_scorer import load_candidate_scorer
        from premove_itn.model_inputs import load_model_tokenizer
        from premove_itn.training import load_checkpoint

        self.torch = torch
        self.tokenizer = load_model_tokenizer()
        self.model = load_candidate_scorer()
        load_checkpoint(self.checkpoint, self.model, map_location="cpu")
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model.to(self.device).eval()
        return {
            "checkpoint": str(self.checkpoint.relative_to(ROOT)),
            "device": str(self.device),
            "model_name": "microsoft/deberta-v3-large",
        }

    def warmup(self) -> None:
        self.normalize("warm up at five thirty")

    def normalize(self, text: str) -> tuple[str, dict[str, Any]]:
        from premove_itn.candidate_scorer import collate_candidate_batch
        from premove_itn.candidates import build_candidate_graph
        from premove_itn.decoder import decode_candidates
        from premove_itn.model_inputs import encode_candidates

        total_started = time.perf_counter_ns()
        started = time.perf_counter_ns()
        candidates = build_candidate_graph(text)
        graph_ms = (time.perf_counter_ns() - started) / 1_000_000
        if not candidates:
            return text, {
                "backend_latency_ms": (time.perf_counter_ns() - total_started)
                / 1_000_000,
                "candidate_count": 0,
                "candidate_graph_ms": graph_ms,
                "encode_ms": 0.0,
                "model_ms": 0.0,
                "decode_ms": 0.0,
                "device": str(self.device),
            }
        started = time.perf_counter_ns()
        encoded = encode_candidates(text, candidates, self.tokenizer)
        batch = collate_candidate_batch(
            [(encoded, candidates)], pad_token_id=self.tokenizer.pad_token_id
        ).to(self.device)
        encode_ms = (time.perf_counter_ns() - started) / 1_000_000
        started = time.perf_counter_ns()
        with self.torch.inference_mode():
            scores = self.model(batch)
        if self.device.type == "mps":
            self.torch.mps.synchronize()
        model_ms = (time.perf_counter_ns() - started) / 1_000_000
        started = time.perf_counter_ns()
        decoded = decode_candidates(text, candidates, scores)
        decode_ms = (time.perf_counter_ns() - started) / 1_000_000
        return decoded.text, {
            "backend_latency_ms": (time.perf_counter_ns() - total_started) / 1_000_000,
            "candidate_count": len(candidates),
            "candidate_graph_ms": graph_ms,
            "encode_ms": encode_ms,
            "model_ms": model_ms,
            "decode_ms": decode_ms,
            "decode_score": decoded.score,
            "selected_candidate_count": len(decoded.selected_candidates),
            "device": str(self.device),
        }

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if getattr(self, "device", None) is not None and self.device.type == "mps":
            self.torch.mps.empty_cache()


class ThutmoseBackend(Backend):
    name = "thutmose"
    description = "Official NVIDIA itn_en_thutmose_bert artifact in an isolated worker."

    def __init__(self, python: Path, artifact: Path, script: Path) -> None:
        self.python = python
        self.artifact = artifact
        self.script = script
        self.process: subprocess.Popen[str] | None = None

    def initialize(self) -> dict[str, Any]:
        if not self.python.is_file():
            raise FileNotFoundError(f"Thutmose Python runtime not found: {self.python}")
        if not self.artifact.is_file():
            raise FileNotFoundError(
                f"Thutmose model artifact not found: {self.artifact}"
            )
        started = time.perf_counter_ns()
        self.process = subprocess.Popen(
            [
                str(self.python),
                str(self.script),
                "--thutmose-worker",
                "--thutmose-artifact",
                str(self.artifact),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"Thutmose worker exited during startup: {stderr}")
        ready = json.loads(line)
        if ready.get("type") != "ready":
            raise RuntimeError(f"Thutmose worker was not ready: {ready}")
        self.worker_warmup_ms = float(ready.get("warmup_ms", 0.0))
        return {
            "artifact": str(self.artifact),
            "worker_python": str(self.python),
            "worker_startup_ms": (time.perf_counter_ns() - started) / 1_000_000,
            "worker_warmup_ms": self.worker_warmup_ms,
            "warmup_in_initialization_ms": self.worker_warmup_ms,
            "device": ready.get("device"),
            "nemo_version": ready.get("nemo_version"),
            "model_target": ready.get("model_target"),
        }

    def warmup(self) -> float:
        # The isolated worker warms the model before emitting its ready line.
        # Return that measured interval so it is reported separately from
        # process/model initialization and excluded from record latency.
        return self.worker_warmup_ms

    def normalize(self, text: str) -> tuple[str, dict[str, Any]]:
        if (
            self.process is None
            or self.process.stdin is None
            or self.process.stdout is None
        ):
            raise RuntimeError("Thutmose worker is not initialized")
        started = time.perf_counter_ns()
        self.process.stdin.write(json.dumps({"text": text}) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        request_ms = (time.perf_counter_ns() - started) / 1_000_000
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"Thutmose worker stopped: {stderr}")
        response = json.loads(line)
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        return str(response["output"]), {
            "backend_latency_ms": float(response["backend_latency_ms"]),
            "request_latency_ms": request_ms,
            "device": response.get("device"),
        }

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process = None


def run_backend(
    backend: Backend,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    init_started = time.perf_counter_ns()
    metadata = backend.initialize()
    init_ms = (time.perf_counter_ns() - init_started) / 1_000_000
    init_ms -= float(metadata.get("warmup_in_initialization_ms", 0.0))
    init_ms = max(init_ms, 0.0)
    warmup_started = time.perf_counter_ns()
    warmup_records = []
    for text in (
        "warm up at five thirty",
        "call nine one one",
        "pay two dollars and fifty cents",
    ) * 5:
        tick = time.perf_counter_ns()
        backend.normalize(text)
        warmup_records.append(
            {"text": text, "latency_ms": (time.perf_counter_ns() - tick) / 1_000_000}
        )
    json_dump(output_dir / "warmup.json", warmup_records)
    warmup_result = None
    measured_warmup_ms = (time.perf_counter_ns() - warmup_started) / 1_000_000
    warmup_ms = (
        float(warmup_result) if warmup_result is not None else measured_warmup_ms
    )
    records: list[dict[str, Any]] = []
    record_path = output_dir / "records.jsonl"
    with record_path.open("w") as handle:
        for index, row in enumerate(rows):
            text = str(row["text"])
            started = time.perf_counter_ns()
            error = None
            details: dict[str, Any] = {}
            prediction = ""
            try:
                prediction, details = backend.normalize(text)
            except Exception as exc:  # keep the row in the audit trail
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000
            scored = (
                score_row(row, prediction)
                if error is None
                else {
                    "strict_exact": False,
                    "semantic_entities_correct": 0,
                    "semantic_entities_total": len(row.get("spans", [])),
                    "semantic_entity_correct": False,
                    "normalization_decision_correct": False,
                    "error_type": "backend_error",
                    "predicted_entities": None,
                }
            )
            record = {
                "index": index,
                "id": row["id"],
                "group": row["group"],
                "kind": row_kind(row),
                "categories": row["categories"],
                "domain": row["domain"],
                "difficulty": row["difficulty"],
                "collision": row.get("collision"),
                "collision_pair_id": row.get("collision_pair_id"),
                "text": text,
                "expected_text": row["expected_text"],
                "prediction": prediction,
                "latency_ms": latency_ms,
                **details,
                **scored,
                "error": error,
            }
            records.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            if (index + 1) % 100 == 0:
                print(
                    json.dumps({"backend": backend.name, "completed": index + 1}),
                    flush=True,
                )
    metrics = aggregate_metrics(rows, records)
    try:
        records_file = str(record_path.relative_to(ROOT))
    except ValueError:
        records_file = str(record_path)
    payload = {
        "backend": backend.name,
        "description": backend.description,
        "started_at": started_at,
        "completed_at": utc_now(),
        "initialization_ms": init_ms,
        "warmup_ms": warmup_ms,
        "records_file": records_file,
        "metrics": metrics,
        "runtime": metadata,
    }
    json_dump(output_dir / "metrics.json", payload)
    return payload


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _metrics_table(mapping: dict[str, Any]) -> list[str]:
    lines = [
        "| Slice | Rows | Strict exact | Semantic entities | All entities | Mean ms | p50 ms | p95 ms | p99 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, metric in mapping.items():
        lines.append(
            f"| `{key}` | {metric.get('records', 0)} | "
            f"{_fmt(metric.get('strict_exact_accuracy'))} | "
            f"{_fmt(metric.get('semantic_entity_accuracy'))} | "
            f"{_fmt(metric.get('all_entities_accuracy'))} | "
            f"{_fmt(metric.get('latency', {}).get('mean_ms'))} | "
            f"{_fmt(metric.get('latency', {}).get('p50_ms'))} | "
            f"{_fmt(metric.get('latency', {}).get('p95_ms'))} | "
            f"{_fmt(metric.get('latency', {}).get('p99_ms'))} |"
        )
    return lines


def write_report(
    output_root: Path,
    run_metadata: dict[str, Any],
    backend_payloads: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# VoiceAgent ITN benchmark results",
        "",
        f"Run completed: `{run_metadata['completed_at']}`  ",
        f"Dataset: `{run_metadata['dataset']}`  ",
        f"Rows: **{len(rows)}**  ",
        "",
        "## Executive summary",
        "",
        "This is a blind, backend-neutral run of the frozen 1,500-row VoiceAgent ITN dataset.",
        "Each adapter received only the row's `text` field. Gold spans, categories, domains,",
        "difficulty, and expected output were withheld from every backend. The per-record JSONL",
        "files are the source of truth for every aggregate below.",
        "",
        "The Premove ITN entry is the current production structured-value 20k checkpoint named",
        "by `data/models/production.json`. The text-processing-rs entry is the compiled upstream",
        "sentence normalizer. Thutmose is the official `itn_en_thutmose_bert` artifact, run in",
        "an isolated worker with its own Python environment.",
        "",
        "## Backend comparison",
        "",
        "| Backend | Rows | Strict exact | Semantic entity | All-entities rows | Mean ms | p50 ms | p95 ms | p99 ms | Throughput/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for payload in backend_payloads:
        overall = payload["metrics"]["overall"]
        latency = overall["latency"]
        lines.append(
            f"| **{payload['backend']}** | {overall['records']} | "
            f"{_fmt(overall.get('strict_exact_accuracy'))} | "
            f"{_fmt(overall.get('semantic_entity_accuracy'))} | "
            f"{_fmt(overall.get('all_entities_accuracy'))} | "
            f"{_fmt(latency.get('mean_ms'))} | {_fmt(latency.get('p50_ms'))} | "
            f"{_fmt(latency.get('p95_ms'))} | {_fmt(latency.get('p99_ms'))} | "
            f"{_fmt(latency.get('records_per_second'))} |"
        )
    lines += [
        "",
        "Strict exact is the canonical full-sentence match. Semantic entity accuracy is",
        "entity-level and category-aware; it preserves numeric value, currency, unit, digit",
        "order, phone dialing form, and identifier case policy. No-span rows do not enter the",
        "semantic-entity denominator. `all-entities rows` is the fraction of rows for which",
        "every declared entity was semantically correct.",
        "Structured identifiers ignore case and harmless whitespace/separator formatting for",
        "semantic scoring; strict exact accuracy still requires the complete expected sentence.",
        "",
        "## Detailed per-backend results",
        "",
    ]
    for payload in backend_payloads:
        metrics = payload["metrics"]
        lines += [
            f"### {payload['backend']}",
            "",
            payload["description"],
            "",
            f"- Records: **{metrics['overall']['records']}**",
            f"- Initialization: **{_fmt(payload['initialization_ms'])} ms**",
            f"- Warm-up: **{_fmt(payload['warmup_ms'])} ms** (excluded from per-record latency)",
            f"- Per-record output: [`{payload['records_file']}`]({payload['records_file']})",
            f"- Runtime: `{json.dumps(payload['runtime'], sort_keys=True)}`",
            "",
            "#### Per kind",
            "",
        ]
        lines += _metrics_table(metrics["per_kind"])
        lines += ["", "#### Per benchmark group", ""]
        lines += _metrics_table(metrics["per_group"])
        lines += ["", "#### Per domain", ""]
        lines += _metrics_table(metrics["per_domain"])
        lines += ["", "#### Per difficulty", ""]
        lines += _metrics_table(metrics["per_difficulty"])
        lines += ["", "#### Per declared entity category", ""]
        lines += _metrics_table(metrics["per_entity_category"])
        lines += [
            "",
            "#### Collision and compositional metrics",
            "",
            f"- Collision pairs: **{metrics['collision']['pairs']}**",
            f"- Strict pair accuracy: **{_fmt(metrics['collision']['pair_metrics']['strict_pair_accuracy'])}**",
            f"- Semantic pair accuracy: **{_fmt(metrics['collision']['pair_metrics']['semantic_pair_accuracy'])}**",
            f"- Multi-entity span accuracy: **{_fmt(metrics['multi_entity'].get('semantic_entity_accuracy'))}**",
            f"- Multi-entity all-correct accuracy: **{_fmt(metrics['multi_entity'].get('all_entities_accuracy'))}**",
            f"- Voice-agent group accuracy (strict): **{_fmt(metrics['voice_agent'].get('strict_exact_accuracy'))}**",
            f"- Standalone negative KEEP preservation: **{_fmt(metrics['standalone_negative_keep'].get('strict_exact_accuracy'))}**",
            f"- All no-span KEEP preservation: **{_fmt(metrics['all_no_span_keep'].get('strict_exact_accuracy'))}**",
            "",
            "Collision-family metrics:",
            "",
            "| Family | Rows | Strict exact | Semantic entities | Mean ms | p95 ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for family, family_metric in metrics["collision"]["per_family"].items():
            lines.append(
                f"| `{family}` | {family_metric['records']} | "
                f"{_fmt(family_metric.get('strict_exact_accuracy'))} | "
                f"{_fmt(family_metric.get('semantic_entity_accuracy'))} | "
                f"{_fmt(family_metric.get('latency', {}).get('mean_ms'))} | "
                f"{_fmt(family_metric.get('latency', {}).get('p95_ms'))} |"
            )
        lines += [
            "",
            "#### Error categories",
            "",
            f"`{json.dumps(metrics['overall'].get('errors', {}), sort_keys=True)}`",
            "",
        ]
    lines += [
        "## Latency methodology",
        "",
        "Latency is measured with a monotonic high-resolution clock around each individual",
        "record. Initialization and one explicit warm-up call are reported separately. The",
        "per-record `latency_ms` includes the adapter call; for Premove ITN the record also",
        "stores candidate enumeration, token encoding, model forward, and exact decoding",
        "components. Thutmose stores both worker inference time (`backend_latency_ms`) and the",
        "main-process request round trip (`request_latency_ms`). The comparison table uses the",
        "adapter-call `latency_ms` field so all three rows have the same boundary.",
        "For Thutmose, model warm-up happens before the worker emits `ready`; its measured",
        "warm-up interval is subtracted from initialization to avoid double counting.",
        "",
        "Percentiles use linear interpolation over the sorted per-record measurements. Throughput",
        "is the number of records divided by the sum of per-record latencies. These measurements",
        "are host-specific and should not be treated as a deployment SLA.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "uv run python scripts/run_voice_agent_benchmark.py",
        "```",
        "",
        "The command writes a new timestamped directory under",
        "`eval/voice_agent_itn/results/`. Use `--output` to select an explicit directory.",
        "The Thutmose environment and artifact can be overridden with",
        "`--thutmose-python` and `--thutmose-artifact`.",
        "",
        "## Limitations",
        "",
        "- This dataset is a balanced synthetic stress suite, not an IID production sample.",
        "- The frozen dataset status remains `pending_independent_review`; this run does not",
        "  change that dataset status or alter its gold labels.",
        "- The current Thutmose Python environment contains NeMo 2.x, while the cached artifact",
        "  targets NeMo 1.9.0rc0. The isolated worker loads the artifact weights and performs",
        "  the documented tagger inference path without passing gold fields to the model.",
        "- No local Docker image was available for this run. Thutmose therefore used the cached",
        "  official `.nemo` artifact in the persistent isolated worker; the exact artifact and",
        "  worker interpreter are recorded in its `metrics.json` runtime block.",
    ]
    (output_root / "REPORT.md").write_text("\n".join(lines) + "\n")


def run_worker(artifact: Path) -> int:
    """Run the isolated Thutmose inference worker."""
    import re

    import torch
    from transformers import BertConfig, BertModel, BertTokenizer

    started = time.perf_counter_ns()
    temp_dir = Path(tempfile.mkdtemp(prefix="thutmose-"))
    try:
        with tarfile.open(artifact) as archive:
            archive.extractall(temp_dir)
        vocab = next(temp_dir.glob("*vocab.txt"))
        label_path = next(temp_dir.glob("*label_map.txt"))
        semiotic_path = next(temp_dir.glob("*semiotic_classes.txt"))
        weights_path = temp_dir / "model_weights.ckpt"
        labels = {
            line.strip(): index
            for index, line in enumerate(label_path.read_text().splitlines())
            if line.strip()
        }
        id_to_label = {index: label for label, index in labels.items()}
        semiotics = {
            line.strip(): index
            for index, line in enumerate(semiotic_path.read_text().splitlines())
            if line.strip()
        }
        id_to_semiotic = {index: label for label, index in semiotics.items()}
        tokenizer = BertTokenizer(vocab_file=str(vocab), do_lower_case=True)
        config = BertConfig(
            vocab_size=len(tokenizer.vocab),
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            type_vocab_size=2,
        )
        encoder = BertModel(config)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(
            {
                key.removeprefix("bert_model."): value
                for key, value in checkpoint.items()
                if key.startswith("bert_model.")
            },
            strict=False,
        )
        tag_head = torch.nn.Linear(768, len(labels))
        tag_head.load_state_dict(
            {
                "weight": checkpoint["logits.mlp.layer0.weight"],
                "bias": checkpoint["logits.mlp.layer0.bias"],
            }
        )
        semiotic_head = torch.nn.Linear(768, len(semiotics))
        semiotic_head.load_state_dict(
            {
                "weight": checkpoint["semiotic_logits.mlp.layer0.weight"],
                "bias": checkpoint["semiotic_logits.mlp.layer0.bias"],
            }
        )
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        encoder.to(device).eval()
        tag_head.to(device).eval()
        semiotic_head.to(device).eval()

        def preprocess(text: str) -> str:
            text = text.casefold()
            text = text.replace("_trans", "").replace("_letter_latin", "")
            return text.replace("_letter", "")

        def realized(text: str) -> tuple[str, list[str], list[str]]:
            processed = preprocess(text)
            words = processed.split()
            pieces = ["[CLS]"]
            token_starts: list[int] = []
            for word in words:
                token_starts.append(len(pieces))
                pieces.extend(tokenizer.tokenize(word))
            pieces.append("[SEP]")
            if len(pieces) > 128:
                raise ValueError("Thutmose max sequence length is 128 wordpieces")
            ids = tokenizer.convert_tokens_to_ids(pieces)
            pad = 128 - len(ids)
            input_ids = torch.tensor(
                [ids + [tokenizer.pad_token_id] * pad], device=device
            )
            mask = torch.tensor([[1] * len(ids) + [0] * pad], device=device)
            segment = torch.zeros_like(input_ids)
            with torch.inference_mode():
                output = encoder(
                    input_ids=input_ids,
                    attention_mask=mask,
                    token_type_ids=segment,
                ).last_hidden_state[0]
                tag_ids = torch.argmax(tag_head(output), dim=-1).tolist()
                semiotic_ids = torch.argmax(semiotic_head(output), dim=-1).tolist()
            if device.type == "mps":
                torch.mps.synchronize()
            predicted_tags = [id_to_label[tag_ids[index]] for index in token_starts]
            predicted_semiotics = [
                id_to_semiotic[semiotic_ids[index]] for index in token_starts
            ]
            # This is the official Thutmose/LaserTagger detokenization rule,
            # including its short and long semiotic-span swaps.
            outputs: list[dict[str, Any]] = []
            for word, tag in zip(words, predicted_tags, strict=True):
                if "|" in tag:
                    tag_type, phrase = tag.split("|", 1)
                else:
                    tag_type, phrase = tag, ""
                if phrase:
                    out = phrase
                elif tag_type == "KEEP":
                    out = word
                else:
                    out = ""
                swap = None
                if out.endswith(">>"):
                    swap = "LONG_RIGHT"
                elif out.endswith("<<"):
                    swap = "LONG_LEFT"
                elif out.endswith(">"):
                    swap = "SHORT_RIGHT"
                elif out.endswith("<"):
                    swap = "SHORT_LEFT"
                outputs.append({"out": out, "swap": swap})
            swapped = [item["out"] for item in outputs]
            for index in range(len(outputs)):
                if outputs[index]["swap"] == "SHORT_LEFT" or (
                    outputs[index - 1]["swap"] == "SHORT_RIGHT"
                ):
                    swapped[index - 1], swapped[index] = (
                        swapped[index],
                        swapped[index - 1],
                    )
                if outputs[index]["swap"] == "LONG_LEFT":
                    item = swapped.pop(index)
                    label = predicted_semiotics[index]
                    position = index - 1
                    while position >= 0 and predicted_semiotics[position] == label:
                        position -= 1
                    swapped.insert(position + 1, item)
            output_text = " ".join(swapped).replace("<", "").replace(">", "")
            fragments = re.split(r"(_[^ ][^_]+[^ ]_)", output_text)
            output_parts = []
            for fragment in fragments:
                if fragment.startswith("_") and fragment.endswith("_"):
                    output_parts.append(fragment.replace(" ", "").replace("_", ""))
                else:
                    output_parts.append(fragment.strip().replace("_", ""))
            output_text = re.sub(r" +", " ", " ".join(output_parts)).strip()
            return output_text, predicted_tags, predicted_semiotics

        # Explicit warm-up. Its time is reported separately from model loading.
        warmup_started = time.perf_counter_ns()
        realized("warm up at five thirty")
        warmup_ms = (time.perf_counter_ns() - warmup_started) / 1_000_000
        print(
            json.dumps(
                {
                    "type": "ready",
                    "device": str(device),
                    "model_target": "nemo.collections.nlp.models.text_normalization_as_tagging.thutmose_tagger.ThutmoseTaggerModel",
                    "nemo_version": "1.9.0rc0 artifact; isolated weight-compatible loader",
                    "startup_ms": (time.perf_counter_ns() - started) / 1_000_000,
                    "warmup_ms": warmup_ms,
                }
            ),
            flush=True,
        )
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                text = request.get("text")
                if not isinstance(text, str) or not text:
                    raise ValueError("text must be a non-empty string")
                begin = time.perf_counter_ns()
                output, tags, semiotics_out = realized(text)
                elapsed = (time.perf_counter_ns() - begin) / 1_000_000
                print(
                    json.dumps(
                        {
                            "output": output,
                            "backend_latency_ms": elapsed,
                            "device": str(device),
                            "tag_count": len(tags),
                            "semiotic_count": len(semiotics_out),
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:
                print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["text-processing-rs", "premove-itn", "thutmose"],
        default=["text-processing-rs", "premove-itn", "thutmose"],
    )
    parser.add_argument("--thutmose-python", type=Path, default=DEFAULT_THUTMOSE_PYTHON)
    parser.add_argument(
        "--thutmose-artifact", type=Path, default=DEFAULT_THUTMOSE_ARTIFACT
    )
    parser.add_argument(
        "--thutmose-worker", action="store_true", help=argparse.SUPPRESS
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.thutmose_worker:
        return run_worker(args.thutmose_artifact)
    from premove_itn import _rust

    info = _rust.build_info()
    if info["debug_assertions"] or info["profile"] != "release":
        raise RuntimeError("Refusing to benchmark a non-release Rust extension")
    rows = read_rows(args.dataset)
    if args.output is None:
        args.output = (
            ROOT
            / "eval/voice_agent_itn/results"
            / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        )
    args.output = args.output.resolve()
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=False)
    run_metadata = {
        "name": "First Evaluation",
        "measurement": "release-artifact latency correction",
        "rust_build": info,
        "rust_extension": _rust.__file__,
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
        "warmup_policy": "15 sequential calls per backend; excluded from record latency",
        "started_at": utc_now(),
        "dataset": str(args.dataset.relative_to(ROOT)),
        "records": len(rows),
        "backends": args.backends,
        "host": platform.platform(),
        "python": sys.version,
        "argv": sys.argv,
    }
    json_dump(output_root / "run.json", run_metadata)
    backends: list[Backend] = []
    production = json.loads((ROOT / "data/models/production.json").read_text())
    checkpoint = ROOT / str(production["checkpoint"])
    for name in args.backends:
        if name == "text-processing-rs":
            backends.append(TextProcessingRSBackend())
        elif name == "premove-itn":
            backends.append(PremoveITNBackend(checkpoint))
        else:
            backends.append(
                ThutmoseBackend(
                    args.thutmose_python,
                    args.thutmose_artifact,
                    Path(__file__).resolve(),
                )
            )
    payloads: list[dict[str, Any]] = []
    try:
        for backend in backends:
            print(
                json.dumps({"backend": backend.name, "status": "started"}), flush=True
            )
            payload = run_backend(backend, rows, output_root / backend.name)
            payloads.append(payload)
            backend.close()
            print(
                json.dumps({"backend": backend.name, "status": "completed"}), flush=True
            )
    finally:
        for backend in reversed(backends):
            backend.close()
    run_metadata["completed_at"] = utc_now()
    run_metadata["backend_metrics"] = [payload["backend"] for payload in payloads]
    json_dump(output_root / "run.json", run_metadata)
    write_report(output_root, run_metadata, payloads, rows)
    print(json.dumps({"status": "completed", "output": str(output_root)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
