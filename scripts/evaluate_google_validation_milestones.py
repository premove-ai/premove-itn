"""Evaluate selected full-Google checkpoints on Google validation only."""

from __future__ import annotations

import gc
import json
import time
from collections import Counter
from pathlib import Path

import torch

from premove_itn import normalize_sentence
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
from premove_itn.training import load_checkpoint

if __package__:
    from scripts.evaluate_full_google_milestones import (
        MILESTONES,
        OUTPUT,
        checkpoint_metadata,
        exposure_from_path,
        sha256,
        validate_state,
        write_json,
    )
else:
    from evaluate_full_google_milestones import (
        MILESTONES,
        OUTPUT,
        checkpoint_metadata,
        exposure_from_path,
        sha256,
        validate_state,
        write_json,
    )

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/google_tn/dataset_1/dataset.jsonl"
EXPOSURES = {
    346_000,
    378_000,
    450_000,
    711_135,
}
EXTRA_CHECKPOINTS = {
    378_000: ROOT / "data/models/google_selected_378k/checkpoint.pt",
}
BATCH_SIZE = 8
WINDOW_SIZE = 128
ERROR_SAMPLE_LIMIT = 20
CONTINUITY_FIRST_ID = 711_552
CONTINUITY_LAST_ID = 721_551


def validation_rows() -> list[dict[str, object]]:
    rows = []
    with DATASET.open() as handle:
        for record_id, line in enumerate(handle):
            row = json.loads(line)
            if row["partition"] == "validation":
                rows.append(dict(row, record_id=record_id))
    return rows


def evaluate_model(model, tokenizer, rows, progress_path):
    predictions: list[str | None] = [None] * len(rows)
    candidate_count = 0
    completed = 0
    started = time.monotonic()
    with torch.inference_mode():
        for window_start in range(0, len(rows), WINDOW_SIZE):
            window_indices = list(
                range(window_start, min(window_start + WINDOW_SIZE, len(rows)))
            )
            window_indices.sort(
                key=lambda index: (len(str(rows[index]["text"])), index)
            )
            for batch_start in range(0, len(window_indices), BATCH_SIZE):
                indices = window_indices[batch_start : batch_start + BATCH_SIZE]
                candidate_sets = []
                encoded = []
                for index in indices:
                    text = str(rows[index]["text"])
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
                for local, (index, candidates) in enumerate(
                    zip(indices, candidate_sets, strict=True)
                ):
                    low, high = batch.candidate_offsets[local : local + 2]
                    predictions[index] = decode_candidates(
                        str(rows[index]["text"]), candidates, scores[low:high]
                    ).text
                completed += len(indices)
            elapsed = time.monotonic() - started
            rate = completed / elapsed
            write_json(
                progress_path,
                {
                    "status": "running",
                    "completed": completed,
                    "total": len(rows),
                    "examples_per_second": rate,
                    "eta_seconds": (len(rows) - completed) / rate,
                },
            )
            if completed % 1_024 == 0:
                print(
                    json.dumps({"evaluated": completed, "total": len(rows)}),
                    flush=True,
                )
    if any(prediction is None for prediction in predictions):
        raise RuntimeError("validation predictions are incomplete")
    elapsed = time.monotonic() - started
    write_json(
        progress_path,
        {
            "status": "completed",
            "completed": len(rows),
            "total": len(rows),
            "examples_per_second": len(rows) / elapsed,
            "eta_seconds": 0.0,
        },
    )
    return [str(prediction) for prediction in predictions], candidate_count, elapsed


def summarize(rows, predictions):
    per_kind = {}
    for kind in sorted({str(row["kind"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if str(row["kind"]) == kind]
        correct = sum(
            predictions[index] == str(rows[index]["expected_text"]) for index in indices
        )
        per_kind[kind] = {
            "correct": correct,
            "total": len(indices),
            "accuracy": correct / len(indices),
        }
    errors: Counter[str] = Counter()
    samples: dict[str, list[dict[str, object]]] = {}
    for row, prediction in zip(rows, predictions, strict=True):
        if prediction == str(row["expected_text"]):
            continue
        kind = str(row["kind"])
        category = (
            "KEEP_over_normalized"
            if kind == "KEEP"
            else (
                "positive_missed_edit"
                if prediction == str(row["text"])
                else "positive_wrong_rewrite"
            )
        )
        errors[category] += 1
        category_samples = samples.setdefault(category, [])
        if len(category_samples) < ERROR_SAMPLE_LIMIT:
            category_samples.append(
                {
                    "id": row["record_id"],
                    "kind": kind,
                    "text": row["text"],
                    "expected": row["expected_text"],
                    "predicted": prediction,
                }
            )

    def subset(kind):
        indices = [
            index
            for index, row in enumerate(rows)
            if (str(row["kind"]) == "KEEP") == (kind == "KEEP")
        ]
        correct = sum(
            predictions[index] == str(rows[index]["expected_text"]) for index in indices
        )
        return {
            "correct": correct,
            "total": len(indices),
            "accuracy": correct / len(indices),
        }

    exact = sum(
        prediction == str(row["expected_text"])
        for row, prediction in zip(rows, predictions, strict=True)
    )
    return {
        "exact": exact,
        "total": len(rows),
        "positive": subset("POSITIVE"),
        "keep": subset("KEEP"),
        "multi": per_kind["MULTI"],
        "per_kind": per_kind,
        "error_categories": dict(errors),
        "error_samples": samples,
    }


def main() -> None:
    checkpoints = {
        exposure_from_path(path): path
        for path in sorted(MILESTONES.glob("checkpoint_*.pt"))
        if exposure_from_path(path) in EXPOSURES
    }
    checkpoints.update(
        {
            exposure: path
            for exposure, path in EXTRA_CHECKPOINTS.items()
            if exposure in EXPOSURES
        }
    )
    missing = EXPOSURES - checkpoints.keys()
    if missing:
        raise RuntimeError(f"missing checkpoints for exposures: {sorted(missing)}")
    rows = validation_rows()
    if len(rows) != 39_543:
        raise RuntimeError(f"expected 39543 validation records; found {len(rows)}")
    tokenizer = load_model_tokenizer()
    model = load_candidate_scorer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    for exposure, checkpoint in sorted(checkpoints.items()):
        output = OUTPUT / f"google_validation_{exposure:06d}"
        metrics_path = output / "metrics.json"
        if metrics_path.exists():
            print(json.dumps({"exposure": exposure, "status": "skipped"}), flush=True)
            continue
        state = load_checkpoint(checkpoint, model, map_location="cpu")
        validate_state(state, exposure)
        model.eval()
        print(json.dumps({"exposure": exposure, "status": "started"}), flush=True)
        predictions, candidate_count, elapsed = evaluate_model(
            model, tokenizer, rows, output / "progress.json"
        )
        continuity_indices = [
            index
            for index, row in enumerate(rows)
            if CONTINUITY_FIRST_ID <= int(row["record_id"]) <= CONTINUITY_LAST_ID
        ]
        continuity_rows = [rows[index] for index in continuity_indices]
        continuity_predictions = [predictions[index] for index in continuity_indices]
        write_json(
            metrics_path,
            {
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_metadata": checkpoint_metadata(state, exposure),
                "dataset": str(DATASET.relative_to(ROOT)),
                "dataset_sha256": sha256(DATASET),
                "partition": "validation",
                "record_ids": [rows[0]["record_id"], rows[-1]["record_id"]],
                "records": len(rows),
                "test_partition_evaluated": False,
                "training_use": False,
                "decoder_inputs": ["text", "all_rust_candidates", "model_scores"],
                "batch_size": BATCH_SIZE,
                "window_size": WINDOW_SIZE,
                "device": str(device),
                "candidate_count": candidate_count,
                "elapsed_seconds": elapsed,
                "examples_per_second": len(rows) / elapsed,
                "trained": summarize(rows, predictions),
                "continuity_holdout": {
                    "record_ids": [CONTINUITY_FIRST_ID, CONTINUITY_LAST_ID],
                    **summarize(continuity_rows, continuity_predictions),
                },
                "deterministic_baseline": summarize(
                    rows, [normalize_sentence(str(row["text"])) for row in rows]
                ),
                "always_keep": summarize(rows, [str(row["text"]) for row in rows]),
            },
        )
        print(json.dumps({"exposure": exposure, "status": "completed"}), flush=True)
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
