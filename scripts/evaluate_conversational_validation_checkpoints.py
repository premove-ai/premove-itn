"""Evaluate selected checkpoints on conversational validation only."""

from __future__ import annotations

import gc
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from premove_itn import normalize_sentence
from premove_itn.candidate_scorer import load_candidate_scorer
from premove_itn.candidates import build_candidate_graph
from premove_itn.model_inputs import MODEL_NAME, MODEL_REVISION, load_model_tokenizer
from premove_itn.training import CheckpointState, load_checkpoint

if __package__:
    from scripts.evaluate_full_google_milestones import (
        MILESTONES,
        checkpoint_metadata,
        exposure_from_path,
        sha256,
        validate_state,
        write_json,
    )
    from scripts.evaluate_google_validation_milestones import evaluate_model
else:
    from evaluate_full_google_milestones import (
        MILESTONES,
        checkpoint_metadata,
        exposure_from_path,
        sha256,
        validate_state,
        write_json,
    )
    from evaluate_google_validation_milestones import evaluate_model

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data/generated/conversational_pool"
OUTPUT = ROOT / "data/generated/full_google_train/evaluations"
CANDIDATE_EXPOSURES = {
    346_000,
    378_000,
    450_000,
    711_135,
}
CHECKPOINTS = {
    exposure_from_path(path): path
    for path in sorted(MILESTONES.glob("checkpoint_*.pt"))
    if exposure_from_path(path) in CANDIDATE_EXPOSURES
}
CHECKPOINTS.update(
    {
        378_000: ROOT / "data/models/google_selected_378k/checkpoint.pt",
    }
)


def validation_rows() -> list[dict[str, object]]:
    manifest = json.loads((POOL / "manifest.json").read_text())
    dataset_path = POOL / "dataset.jsonl"
    provenance_path = POOL / "provenance.jsonl"
    if manifest["artifacts"]["dataset.jsonl"]["sha256"] != sha256(dataset_path):
        raise RuntimeError("conversational dataset hash does not match its manifest")
    if manifest["artifacts"]["provenance.jsonl"]["sha256"] != sha256(provenance_path):
        raise RuntimeError("conversational provenance hash does not match its manifest")
    rows = [json.loads(line) for line in dataset_path.open()]
    provenance = [json.loads(line) for line in provenance_path.open()]
    if len(rows) != len(provenance):
        raise RuntimeError("conversational dataset and provenance are misaligned")
    selected = []
    for record_id, (row, source) in enumerate(zip(rows, provenance, strict=True)):
        if source.get("record_index") != record_id:
            raise RuntimeError("conversational provenance record index is misaligned")
        if row["partition"] != "validation":
            continue
        selected.append(
            {
                **row,
                "record_id": record_id,
                "evaluation_sources": sorted(
                    {str(item["source"]) for item in source["sources"]}
                ),
                "candidate_count": len(build_candidate_graph(str(row["text"]))),
            }
        )
    expected = manifest["partition_distribution"]["validation"]
    if len(selected) != expected:
        raise RuntimeError(
            f"expected {expected} conversational validation records; "
            f"found {len(selected)}"
        )
    return selected


def _subset(
    rows: Sequence[dict[str, object]],
    predictions: Sequence[str],
    indices: Sequence[int],
) -> dict[str, object]:
    correct = sum(
        predictions[index] == str(rows[index]["expected_text"]) for index in indices
    )
    total = len(indices)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
    }


def summarize(
    rows: Sequence[dict[str, object]], predictions: Sequence[str]
) -> dict[str, object]:
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have equal length")
    indices = list(range(len(rows)))
    positive = [index for index in indices if rows[index]["kind"] != "KEEP"]
    keep = [index for index in indices if rows[index]["kind"] == "KEEP"]
    active = [index for index in indices if int(rows[index]["candidate_count"]) > 0]
    active_keep = [index for index in keep if int(rows[index]["candidate_count"]) > 0]
    per_kind = {
        kind: _subset(
            rows,
            predictions,
            [index for index in indices if rows[index]["kind"] == kind],
        )
        for kind in sorted({str(row["kind"]) for row in rows})
    }
    sources = sorted({source for row in rows for source in row["evaluation_sources"]})
    per_source = {}
    for source in sources:
        source_indices = [
            index for index in indices if source in rows[index]["evaluation_sources"]
        ]
        per_source[source] = {
            **_subset(rows, predictions, source_indices),
            "positive": _subset(
                rows,
                predictions,
                [index for index in source_indices if rows[index]["kind"] != "KEEP"],
            ),
            "keep": _subset(
                rows,
                predictions,
                [index for index in source_indices if rows[index]["kind"] == "KEEP"],
            ),
        }
    return {
        "overall": _subset(rows, predictions, indices),
        "positive": _subset(rows, predictions, positive),
        "keep": _subset(rows, predictions, keep),
        "candidate_bearing": _subset(rows, predictions, active),
        "candidate_bearing_keep": _subset(rows, predictions, active_keep),
        "multi": per_kind.get("MULTI", {"correct": 0, "total": 0, "accuracy": None}),
        "per_kind": per_kind,
        "per_source": per_source,
    }


def _validate_checkpoint(state: CheckpointState, exposure: int) -> dict[str, object]:
    if exposure > 10_000:
        validate_state(state, exposure)
        return checkpoint_metadata(state, exposure)
    if state.epoch != 1 or state.batch_offset != 0:
        raise ValueError(f"{exposure} checkpoint does not contain a completed epoch")
    if not state.metrics or state.metrics[-1].examples != exposure:
        raise ValueError(f"{exposure} checkpoint has incorrect example metadata")
    return {
        "epoch": state.epoch,
        "step": state.step,
        "batch_offset": state.batch_offset,
        "total_exposure": exposure,
        "training_distribution": dict(state.training_distribution),
    }


def main() -> None:
    missing = [str(path) for path in CHECKPOINTS.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing checkpoints: {missing}")
    rows = validation_rows()
    baseline_predictions = [normalize_sentence(str(row["text"])) for row in rows]
    keep_predictions = [str(row["text"]) for row in rows]
    tokenizer = load_model_tokenizer()
    model = load_candidate_scorer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    for exposure, checkpoint in CHECKPOINTS.items():
        output = OUTPUT / f"conversational_validation_{exposure:06d}"
        metrics_path = output / "metrics.json"
        if metrics_path.exists():
            print(json.dumps({"exposure": exposure, "status": "skipped"}), flush=True)
            continue
        state = load_checkpoint(checkpoint, model, map_location="cpu")
        metadata = _validate_checkpoint(state, exposure)
        model.eval()
        print(json.dumps({"exposure": exposure, "status": "started"}), flush=True)
        predictions, candidate_count, elapsed = evaluate_model(
            model, tokenizer, rows, output / "progress.json"
        )
        write_json(
            metrics_path,
            {
                "benchmark_role": "conversational_validation",
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_metadata": metadata,
                "dataset": str((POOL / "validation.jsonl").relative_to(ROOT)),
                "dataset_sha256": sha256(POOL / "validation.jsonl"),
                "records": len(rows),
                "partition": "validation",
                "test_partition_evaluated": False,
                "training_use": False,
                "decoder_inputs": ["text", "all_rust_candidates", "model_scores"],
                "device": str(device),
                "candidate_count": candidate_count,
                "elapsed_seconds": elapsed,
                "examples_per_second": len(rows) / elapsed,
                "trained": summarize(rows, predictions),
                "deterministic_baseline": summarize(rows, baseline_predictions),
                "always_keep": summarize(rows, keep_predictions),
            },
        )
        print(json.dumps({"exposure": exposure, "status": "completed"}), flush=True)
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
