"""Evaluate every preserved full-Google checkpoint on Golden and Numb3rs."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import torch

from premove_itn import (
    SpanKind,
    normalize_sentence,
    representations_equivalent,
    target_is_reachable,
)
from premove_itn.candidate_scorer import (
    collate_candidate_batch,
    load_candidate_scorer,
)
from premove_itn.candidates import build_candidate_graph
from premove_itn.decoder import decode_candidates
from premove_itn.model_inputs import (
    MODEL_NAME,
    MODEL_REVISION,
    encode_candidates,
    load_model_tokenizer,
)
from premove_itn.training import CheckpointState, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
MILESTONES = ROOT / "data/generated/full_google_train/milestones"
OUTPUT = ROOT / "data/generated/full_google_train/evaluations"
GOLDEN = ROOT / "data/golden.json"
NUMB3RS = (
    ROOT
    / "data/external/nvidia-numb3rs"
    / "192908075e1bd293914cc7b508f4a183ba6ef2b8"
    / "metadata.jsonl"
)
INITIAL_EXAMPLES = 10_000
BATCH_SIZE = 8
ERROR_SAMPLE_LIMIT = 20
SEMANTIC_KINDS = {
    "CARDINAL": SpanKind.CARDINAL,
    "DATE": SpanKind.DATE,
    "DECIMAL": SpanKind.DECIMAL,
    "DIGIT": SpanKind.DIGIT_SEQUENCE,
    "MEASURE": SpanKind.MEASUREMENT,
    "MONEY": SpanKind.MONEY,
    "ORDINAL": SpanKind.ORDINAL,
    "TELEPHONE": SpanKind.PHONE,
    "TIME": SpanKind.TIME,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def exposure_from_path(path: Path) -> int:
    return int(path.stem.removeprefix("checkpoint_"))


def validate_state(state: CheckpointState, exposure: int) -> None:
    if exposure == 711_135:
        if state.epoch != 2 or state.batch_offset != 0:
            raise ValueError("final checkpoint does not contain a completed epoch")
        if not state.metrics or state.metrics[-1].examples != 701_135:
            raise ValueError("final checkpoint has incorrect example metadata")
        return
    actual = INITIAL_EXAMPLES + state.batch_offset * BATCH_SIZE
    if actual != exposure:
        raise ValueError(f"checkpoint exposure is {actual}; expected {exposure}")


def evaluate_predictions(
    rows: Sequence[dict[str, object]],
    predictions: Sequence[str],
    *,
    reference_field: str,
    category_field: str,
) -> dict[str, object]:
    def semantically_correct(row: dict[str, object], prediction: str) -> bool:
        expected = str(row[reference_field])
        if prediction == expected:
            return True
        kind = SEMANTIC_KINDS.get(str(row[category_field]))
        return bool(
            kind is not None and representations_equivalent(kind, expected, prediction)
        )

    categories = sorted({str(row[category_field]) for row in rows})
    per_category: dict[str, object] = {}
    for category in categories:
        indices = [
            index
            for index, row in enumerate(rows)
            if str(row[category_field]) == category
        ]
        correct = sum(
            predictions[index] == str(rows[index][reference_field]) for index in indices
        )
        per_category[category] = {
            "correct": correct,
            "total": len(indices),
            "accuracy": correct / len(indices),
        }
    errors: Counter[str] = Counter()
    samples: dict[str, list[dict[str, object]]] = {}
    for row, prediction in zip(rows, predictions, strict=True):
        expected = str(row[reference_field])
        if prediction == expected:
            continue
        category = "missed_edit" if prediction == str(row["text"]) else "wrong_rewrite"
        errors[category] += 1
        category_samples = samples.setdefault(category, [])
        if len(category_samples) < ERROR_SAMPLE_LIMIT:
            category_samples.append(
                {
                    "id": row.get("id", row.get("name")),
                    "category": row[category_field],
                    "text": row["text"],
                    "expected": expected,
                    "predicted": prediction,
                }
            )
    exact = sum(
        prediction == str(row[reference_field])
        for row, prediction in zip(rows, predictions, strict=True)
    )
    semantic_per_category: dict[str, object] = {}
    semantic_correct = 0
    for category in categories:
        indices = [
            index
            for index, row in enumerate(rows)
            if str(row[category_field]) == category
        ]
        correct = sum(
            semantically_correct(rows[index], predictions[index]) for index in indices
        )
        semantic_correct += correct
        semantic_per_category[category] = {
            "correct": correct,
            "total": len(indices),
            "accuracy": correct / len(indices),
        }
    return {
        "exact": exact,
        "total": len(rows),
        "accuracy": exact / len(rows),
        "error_categories": dict(errors),
        "error_samples": samples,
        "per_category": per_category,
        "semantic": {
            "correct": semantic_correct,
            "total": len(rows),
            "accuracy": semantic_correct / len(rows),
            "per_category": semantic_per_category,
        },
    }


def golden_kind(row: dict[str, object]) -> str:
    spans = list(row.get("spans", []))
    if not spans:
        return "KEEP"
    if len(spans) > 1:
        return "MULTI"
    return str(spans[0]["kind"])


def evaluate_model(model, tokenizer, rows: Sequence[dict[str, object]]) -> tuple:
    predictions: list[str] = []
    candidate_count = 0
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            chunk = rows[start : start + BATCH_SIZE]
            candidate_sets = []
            encoded = []
            for row in chunk:
                text = str(row["text"])
                candidates = build_candidate_graph(text)
                candidate_sets.append(candidates)
                encoded.append(
                    (encode_candidates(text, candidates, tokenizer), candidates)
                )
                candidate_count += len(candidates)
            batch = collate_candidate_batch(
                encoded, pad_token_id=tokenizer.pad_token_id
            ).to(next(model.parameters()).device)
            scores = model(batch)
            for local, (row, candidates) in enumerate(
                zip(chunk, candidate_sets, strict=True)
            ):
                low, high = batch.candidate_offsets[local : local + 2]
                predictions.append(
                    decode_candidates(
                        str(row["text"]), candidates, scores[low:high]
                    ).text
                )
            done = min(start + BATCH_SIZE, len(rows))
            if done % 800 == 0:
                print(json.dumps({"evaluated": done, "total": len(rows)}), flush=True)
    elapsed = time.monotonic() - started
    return predictions, candidate_count, elapsed


def golden_metrics(model, tokenizer, rows, checkpoint, state, exposure):
    predictions, candidate_count, elapsed = evaluate_model(model, tokenizer, rows)
    categorized = [dict(row, evaluation_kind=golden_kind(row)) for row in rows]
    summary = evaluate_predictions(
        categorized,
        predictions,
        reference_field="expected_text",
        category_field="evaluation_kind",
    )
    per_kind = summary.pop("per_category")
    positive = [
        index
        for index, row in enumerate(categorized)
        if row["evaluation_kind"] != "KEEP"
    ]
    keep = [
        index
        for index, row in enumerate(categorized)
        if row["evaluation_kind"] == "KEEP"
    ]
    multi = [
        index
        for index, row in enumerate(categorized)
        if row["evaluation_kind"] == "MULTI"
    ]

    def subset(indices):
        correct = sum(
            predictions[index] == str(rows[index]["expected_text"]) for index in indices
        )
        return {
            "correct": correct,
            "total": len(indices),
            "accuracy": correct / len(indices),
        }

    summary.update(
        positive=subset(positive),
        keep=subset(keep),
        multi=subset(multi),
        per_kind=per_kind,
    )
    return {
        "benchmark_role": "curated_regression_development_benchmark",
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_metadata": checkpoint_metadata(state, exposure),
        "golden_dataset": str(GOLDEN.relative_to(ROOT)),
        "golden_sha256": sha256(GOLDEN),
        "records": len(rows),
        "training_use": False,
        "decoder_inputs": ["text", "all_rust_candidates", "model_scores"],
        "batch_size": BATCH_SIZE,
        "device": str(next(model.parameters()).device),
        "candidate_count": candidate_count,
        "elapsed_seconds": elapsed,
        "examples_per_second": len(rows) / elapsed,
        "trained": summary,
    }


def checkpoint_metadata(state: CheckpointState, exposure: int) -> dict[str, object]:
    return {
        "epoch": state.epoch,
        "step": state.step,
        "batch_offset": state.batch_offset,
        "total_exposure": exposure,
        "training_distribution": dict(state.training_distribution),
    }


def numb3rs_metrics(model, tokenizer, rows, checkpoint, state, exposure):
    predictions, candidate_count, elapsed = evaluate_model(model, tokenizer, rows)
    baseline_predictions = [normalize_sentence(str(row["text"])) for row in rows]
    return {
        "source": "nvidia/Numb3rs",
        "source_revision": "192908075e1bd293914cc7b508f4a183ba6ef2b8",
        "license": "CC BY-NC-SA 4.0",
        "split": "test",
        "evaluation_only": True,
        "dataset": str(NUMB3RS.relative_to(ROOT)),
        "dataset_sha256": sha256(NUMB3RS),
        "input_field": "text",
        "reference_field": "original_text",
        "records": len(rows),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_metadata": checkpoint_metadata(state, exposure),
        "decoder_inputs": ["text", "all_rust_candidates", "model_scores"],
        "device": str(next(model.parameters()).device),
        "batch_size": BATCH_SIZE,
        "candidate_count": candidate_count,
        "elapsed_seconds": elapsed,
        "examples_per_second": len(rows) / elapsed,
        "oracle": {
            "reachable": sum(
                target_is_reachable(str(row["text"]), str(row["original_text"]))
                for row in rows
            ),
            "total": len(rows),
        },
        "trained": evaluate_predictions(
            rows,
            predictions,
            reference_field="original_text",
            category_field="category",
        ),
        "deterministic_baseline": evaluate_predictions(
            rows,
            baseline_predictions,
            reference_field="original_text",
            category_field="category",
        ),
    }


def main() -> None:
    checkpoints = sorted(MILESTONES.glob("checkpoint_*.pt"))
    if not checkpoints:
        raise RuntimeError("no milestone checkpoints found")
    golden_rows = json.loads(GOLDEN.read_text())
    numb3rs_rows = [json.loads(line) for line in NUMB3RS.open()]
    tokenizer = load_model_tokenizer()
    model = load_candidate_scorer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    for checkpoint in checkpoints:
        exposure = exposure_from_path(checkpoint)
        golden_path = OUTPUT / f"golden_{exposure:06d}/metrics.json"
        numb3rs_path = OUTPUT / f"numb3rs_{exposure:06d}/metrics.json"
        if golden_path.exists() and numb3rs_path.exists():
            print(json.dumps({"exposure": exposure, "status": "skipped"}), flush=True)
            continue
        state = load_checkpoint(checkpoint, model, map_location="cpu")
        validate_state(state, exposure)
        model.eval()
        print(json.dumps({"exposure": exposure, "status": "started"}), flush=True)
        if not golden_path.exists():
            write_json(
                golden_path,
                golden_metrics(
                    model, tokenizer, golden_rows, checkpoint, state, exposure
                ),
            )
        if not numb3rs_path.exists():
            write_json(
                numb3rs_path,
                numb3rs_metrics(
                    model, tokenizer, numb3rs_rows, checkpoint, state, exposure
                ),
            )
        print(json.dumps({"exposure": exposure, "status": "completed"}), flush=True)
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
