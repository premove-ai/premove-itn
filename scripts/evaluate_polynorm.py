"""Evaluate one contextual checkpoint on Apple PolyNorm-Bench en-US."""

from __future__ import annotations

import argparse
import json
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
from premove_itn.candidate_scorer import load_candidate_scorer
from premove_itn.model_inputs import MODEL_NAME, MODEL_REVISION, load_model_tokenizer
from premove_itn.training import load_checkpoint

if __package__:
    from scripts.evaluate_full_google_milestones import (
        evaluate_model,
        sha256,
        write_json,
    )
else:
    from evaluate_full_google_milestones import evaluate_model, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "f3c67e047bea6b7c40bc2466c0fdaad51d8ce67d"
DATASET = (
    ROOT / "data/external/apple-polynorm/polynorm_bench/en-US/en-US_groundtruth.jsonl"
)
DEFAULT_RUNS = (
    (
        ROOT / "data/models/google_selected_378k/checkpoint.pt",
        ROOT / "data/generated/polynorm/google_378000/metrics.json",
    ),
    (
        ROOT / "data/models/conversational_selected_20k/checkpoint.pt",
        ROOT / "data/generated/polynorm/checkpoint_020000/metrics.json",
    ),
)
ERROR_SAMPLE_LIMIT = 20
SEMANTIC_KINDS = {
    "Cardinal": SpanKind.CARDINAL,
    "Currency": SpanKind.MONEY,
    "Decimal": SpanKind.DECIMAL,
    "Ordinal": SpanKind.ORDINAL,
    "Phone Number": SpanKind.PHONE,
    "Time": SpanKind.TIME,
    "Unit": SpanKind.MEASUREMENT,
}


def load_rows(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    required = {"index", "category", "original_text", "normalized_text"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("PolyNorm rows do not have the expected schema")
    if len({str(row["index"]) for row in rows}) != len(rows):
        raise ValueError("PolyNorm row indices must be unique")
    return [
        {
            "id": str(row["index"]),
            "category": str(row["category"]),
            "text": str(row["normalized_text"]),
            "original_text": str(row["original_text"]),
        }
        for row in rows
    ]


def summarize(
    rows: Sequence[dict[str, object]],
    predictions: Sequence[str],
    reachable: Sequence[bool],
    semantic_kinds: dict[str, SpanKind] | None = None,
) -> dict[str, object]:
    if not (len(rows) == len(predictions) == len(reachable)):
        raise ValueError("rows, predictions, and reachability must have equal length")

    semantic_kinds = semantic_kinds or {}

    def is_semantic(index: int) -> bool:
        expected = str(rows[index]["original_text"])
        prediction = predictions[index]
        if prediction == expected:
            return True
        kind = semantic_kinds.get(str(rows[index]["category"]))
        return bool(
            kind is not None and representations_equivalent(kind, expected, prediction)
        )

    def metrics(indices: Sequence[int]) -> dict[str, object]:
        correct = sum(
            predictions[index] == str(rows[index]["original_text"]) for index in indices
        )
        total = len(indices)
        return {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else None,
        }

    def semantic_metrics(indices: Sequence[int]) -> dict[str, object]:
        correct = sum(is_semantic(index) for index in indices)
        total = len(indices)
        return {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else None,
        }

    per_category = {}
    oracle_per_category = {}
    for category in sorted({str(row["category"]) for row in rows}):
        indices = [
            index for index, row in enumerate(rows) if row["category"] == category
        ]
        reachable_indices = [index for index in indices if reachable[index]]
        per_category[category] = {
            "all": metrics(indices),
            "reachable_only": metrics(reachable_indices),
            "semantic": semantic_metrics(indices),
        }
        oracle_per_category[category] = {
            "reachable": len(reachable_indices),
            "total": len(indices),
            "rate": len(reachable_indices) / len(indices),
        }

    errors: Counter[str] = Counter()
    samples: dict[str, list[dict[str, object]]] = {}
    for row, prediction, is_reachable in zip(rows, predictions, reachable, strict=True):
        expected = str(row["original_text"])
        if prediction == expected:
            continue
        error = (
            "unreachable_target"
            if not is_reachable
            else "missed_edit"
            if prediction == str(row["text"])
            else "wrong_rewrite"
        )
        errors[error] += 1
        selected = samples.setdefault(error, [])
        if len(selected) < ERROR_SAMPLE_LIMIT:
            selected.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "text": row["text"],
                    "expected": expected,
                    "predicted": prediction,
                }
            )

    reachable_indices = [index for index, value in enumerate(reachable) if value]
    return {
        "all": metrics(list(range(len(rows)))),
        "reachable_only": metrics(reachable_indices),
        "semantic": semantic_metrics(list(range(len(rows)))),
        "per_category": per_category,
        "error_categories": dict(errors),
        "error_samples": samples,
        "oracle": {
            "reachable": len(reachable_indices),
            "total": len(rows),
            "rate": len(reachable_indices) / len(rows),
            "per_category": oracle_per_category,
        },
    }


def evaluate(checkpoint: Path, output: Path) -> dict[str, object]:
    rows = load_rows(DATASET)
    reachable = [
        target_is_reachable(str(row["text"]), str(row["original_text"])) for row in rows
    ]
    tokenizer = load_model_tokenizer()
    model = load_candidate_scorer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    state = load_checkpoint(checkpoint, model, map_location="cpu")
    model.eval()
    predictions, candidate_count, elapsed = evaluate_model(model, tokenizer, rows)
    baseline = [normalize_sentence(str(row["text"])) for row in rows]
    payload = {
        "benchmark_role": "external_reversed_text_normalization_diagnostic",
        "source": "apple/ml-speech-polynorm-bench",
        "source_revision": SOURCE_REVISION,
        "license": "CC BY-NC-ND 4.0",
        "split": "en-US_groundtruth",
        "evaluation_only": True,
        "training_use": False,
        "direction": "normalized_text_to_original_text",
        "scoring_note": (
            "Strict exact scoring includes PolyNorm written-form policy; reversed "
            "text normalization is not always uniquely recoverable."
        ),
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": sha256(DATASET),
        "records": len(rows),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_metadata": {
            "epoch": state.epoch,
            "step": state.step,
            "batch_offset": state.batch_offset,
            "training_distribution": dict(state.training_distribution),
        },
        "decoder_inputs": ["text", "all_rust_candidates", "model_scores"],
        "device": str(device),
        "candidate_count": candidate_count,
        "elapsed_seconds": elapsed,
        "examples_per_second": len(rows) / elapsed,
        "trained": summarize(rows, predictions, reachable, SEMANTIC_KINDS),
        "deterministic_baseline": summarize(rows, baseline, reachable, SEMANTIC_KINDS),
    }
    write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not DATASET.is_file():
        raise RuntimeError("PolyNorm is missing; run scripts/download_polynorm.py")
    if (arguments.checkpoint is None) != (arguments.output is None):
        parser.error("--checkpoint and --output must be supplied together")
    runs = (
        ((arguments.checkpoint, arguments.output),)
        if arguments.checkpoint is not None
        else DEFAULT_RUNS
    )
    for checkpoint, output in runs:
        if not checkpoint.is_file():
            raise RuntimeError(f"checkpoint is missing: {checkpoint}")
        if output.is_file():
            print(json.dumps({"output": str(output), "status": "skipped"}), flush=True)
            continue
        payload = evaluate(checkpoint.resolve(), output.resolve())
        print(
            json.dumps(
                {
                    "checkpoint": payload["checkpoint"],
                    "all": payload["trained"]["all"],
                    "reachable_only": payload["trained"]["reachable_only"],
                    "oracle": payload["trained"]["oracle"]["reachable"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
