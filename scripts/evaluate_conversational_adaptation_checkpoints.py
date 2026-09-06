"""Evaluate conversational-adaptation milestones without changing training state."""

from __future__ import annotations

import fcntl
import gc
import hashlib
import json
from pathlib import Path

import torch

from premove_itn import normalize_sentence
from premove_itn.candidate_scorer import load_candidate_scorer
from premove_itn.model_inputs import MODEL_NAME, MODEL_REVISION, load_model_tokenizer
from premove_itn.training import load_checkpoint

if __package__:
    from scripts import build_evaluation_history as evaluation_history
    from scripts import evaluate_conversational_validation_checkpoints as conversational
    from scripts import evaluate_full_google_milestones as full_google
    from scripts import evaluate_google_validation_milestones as google
else:
    import build_evaluation_history as evaluation_history
    import evaluate_conversational_validation_checkpoints as conversational
    import evaluate_full_google_milestones as full_google
    import evaluate_google_validation_milestones as google


ROOT = Path(__file__).resolve().parents[1]
ADAPTATION = ROOT / "data/generated/conversational_adaptation_44k_v2"
MILESTONES = ADAPTATION / "milestones"
OUTPUT = ADAPTATION / "evaluations"
LOCK = ADAPTATION / ".evaluation.lock"
EXPOSURES = (10_000, 20_000, 30_000, 44_058)
CHECKPOINTS = {
    exposure: MILESTONES / f"checkpoint_{exposure:06d}.pt" for exposure in EXPOSURES
}


def _model_digest(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_checkpoint_set(checkpoints: dict[int, Path]) -> dict[int, str]:
    """Reject mislabeled or duplicate adaptation milestone payloads."""
    fingerprints: dict[int, str] = {}
    run_fingerprint = None
    for exposure, checkpoint in sorted(checkpoints.items()):
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        metrics = payload.get("metrics")
        if not metrics:
            raise RuntimeError(f"{checkpoint} has no training metrics")
        actual = int(metrics[-1]["examples"])
        if actual != exposure:
            raise RuntimeError(
                f"{checkpoint} contains {actual} examples; expected {exposure}"
            )
        current_run = payload.get("run_fingerprint")
        if run_fingerprint is None:
            run_fingerprint = current_run
        elif current_run != run_fingerprint:
            raise RuntimeError("adaptation checkpoints have different run fingerprints")
        model_digest = _model_digest(payload["model_state_dict"])
        for previous_exposure, previous_digest in fingerprints.items():
            if model_digest == previous_digest:
                raise RuntimeError(
                    f"{exposure} and {previous_exposure} checkpoints contain "
                    "identical model tensors"
                )
        fingerprints[exposure] = model_digest
    return fingerprints


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if path.name == "metrics.json" and path.parent.parent == OUTPUT:
        evaluation_history.write_history(
            evaluation_history.HISTORY_PATH,
            evaluation_history.build_history(ROOT),
        )


def metadata(state, checkpoint: Path, exposure: int) -> dict[str, object]:
    return {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_exposure": exposure,
        "base_checkpoint": "data/models/google_selected_378k/checkpoint.pt",
        "base_exposure": 378_000,
        "combined_exposure": 378_000 + exposure,
        "epoch": state.epoch,
        "step": state.step,
        "batch_offset": state.batch_offset,
        "run_fingerprint": state.run_fingerprint,
        "training_distribution": dict(state.training_distribution),
    }


def adaptation_metrics(
    payload: dict[str, object], state, checkpoint: Path, exposure: int
):
    payload["model_name"] = MODEL_NAME
    payload["model_revision"] = MODEL_REVISION
    payload["checkpoint_metadata"] = metadata(state, checkpoint, exposure)
    payload["adaptation_stage"] = True
    payload["evaluation_only"] = True
    payload["training_use"] = False
    return payload


def evaluate_google(
    model, tokenizer, rows, state, checkpoint: Path, exposure: int
) -> None:
    output = OUTPUT / f"google_validation_{exposure:06d}"
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        print(
            json.dumps(
                {
                    "dataset": "google_validation",
                    "exposure": exposure,
                    "status": "skipped",
                }
            ),
            flush=True,
        )
        return
    print(
        json.dumps(
            {"dataset": "google_validation", "exposure": exposure, "status": "started"}
        ),
        flush=True,
    )
    predictions, candidate_count, elapsed = google.evaluate_model(
        model, tokenizer, rows, output / "progress.json"
    )
    continuity_indices = [
        index
        for index, row in enumerate(rows)
        if google.CONTINUITY_FIRST_ID
        <= int(row["record_id"])
        <= google.CONTINUITY_LAST_ID
    ]
    continuity_rows = [rows[index] for index in continuity_indices]
    continuity_predictions = [predictions[index] for index in continuity_indices]
    payload = {
        "benchmark_role": "google_validation_after_conversational_adaptation",
        "dataset": str(google.DATASET.relative_to(ROOT)),
        "dataset_sha256": full_google.sha256(google.DATASET),
        "partition": "validation",
        "records": len(rows),
        "record_ids": [rows[0]["record_id"], rows[-1]["record_id"]],
        "test_partition_evaluated": False,
        "decoder_inputs": ["text", "all_rust_candidates", "model_scores"],
        "device": str(next(model.parameters()).device),
        "batch_size": google.BATCH_SIZE,
        "window_size": google.WINDOW_SIZE,
        "candidate_count": candidate_count,
        "elapsed_seconds": elapsed,
        "examples_per_second": len(rows) / elapsed,
        "trained": google.summarize(rows, predictions),
        "continuity_holdout": {
            "record_ids": [google.CONTINUITY_FIRST_ID, google.CONTINUITY_LAST_ID],
            **google.summarize(continuity_rows, continuity_predictions),
        },
        "deterministic_baseline": google.summarize(
            rows, [normalize_sentence(str(row["text"])) for row in rows]
        ),
        "always_keep": google.summarize(rows, [str(row["text"]) for row in rows]),
    }
    write_json(metrics_path, adaptation_metrics(payload, state, checkpoint, exposure))
    print(
        json.dumps(
            {
                "dataset": "google_validation",
                "exposure": exposure,
                "status": "completed",
            }
        ),
        flush=True,
    )


def evaluate_conversational(
    model, tokenizer, rows, state, checkpoint: Path, exposure: int
) -> None:
    output = OUTPUT / f"conversational_validation_{exposure:06d}"
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        print(
            json.dumps(
                {
                    "dataset": "conversational_validation",
                    "exposure": exposure,
                    "status": "skipped",
                }
            ),
            flush=True,
        )
        return
    print(
        json.dumps(
            {
                "dataset": "conversational_validation",
                "exposure": exposure,
                "status": "started",
            }
        ),
        flush=True,
    )
    predictions, candidate_count, elapsed = google.evaluate_model(
        model, tokenizer, rows, output / "progress.json"
    )
    payload = {
        "benchmark_role": "conversational_validation_after_conversational_adaptation",
        "dataset": str((conversational.POOL / "validation.jsonl").relative_to(ROOT)),
        "dataset_sha256": full_google.sha256(conversational.POOL / "validation.jsonl"),
        "partition": "validation",
        "records": len(rows),
        "test_partition_evaluated": False,
        "decoder_inputs": ["text", "all_rust_candidates", "model_scores"],
        "device": str(next(model.parameters()).device),
        "candidate_count": candidate_count,
        "elapsed_seconds": elapsed,
        "examples_per_second": len(rows) / elapsed,
        "trained": conversational.summarize(rows, predictions),
        "deterministic_baseline": conversational.summarize(
            rows, [normalize_sentence(str(row["text"])) for row in rows]
        ),
        "always_keep": conversational.summarize(
            rows, [str(row["text"]) for row in rows]
        ),
    }
    write_json(metrics_path, adaptation_metrics(payload, state, checkpoint, exposure))
    print(
        json.dumps(
            {
                "dataset": "conversational_validation",
                "exposure": exposure,
                "status": "completed",
            }
        ),
        flush=True,
    )


def _run() -> None:
    missing = [str(path) for path in CHECKPOINTS.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing adaptation checkpoints: {missing}")
    validate_checkpoint_set(CHECKPOINTS)
    golden_rows = json.loads(full_google.GOLDEN.read_text())
    numb3rs_rows = [json.loads(line) for line in full_google.NUMB3RS.open()]
    google_rows = google.validation_rows()
    conversational_rows = conversational.validation_rows()
    tokenizer = load_model_tokenizer()
    model = load_candidate_scorer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    for exposure in EXPOSURES:
        checkpoint = CHECKPOINTS[exposure]
        state = load_checkpoint(checkpoint, model, map_location="cpu")
        evaluate_google(model, tokenizer, google_rows, state, checkpoint, exposure)
        evaluate_conversational(
            model, tokenizer, conversational_rows, state, checkpoint, exposure
        )
        golden_path = OUTPUT / f"golden_{exposure:06d}/metrics.json"
        if not golden_path.exists():
            print(
                json.dumps(
                    {"dataset": "golden", "exposure": exposure, "status": "started"}
                ),
                flush=True,
            )
            payload = full_google.golden_metrics(
                model, tokenizer, golden_rows, checkpoint, state, exposure
            )
            write_json(
                golden_path, adaptation_metrics(payload, state, checkpoint, exposure)
            )
            print(
                json.dumps(
                    {"dataset": "golden", "exposure": exposure, "status": "completed"}
                ),
                flush=True,
            )
        else:
            print(
                json.dumps(
                    {"dataset": "golden", "exposure": exposure, "status": "skipped"}
                ),
                flush=True,
            )
        numb3rs_path = OUTPUT / f"numb3rs_{exposure:06d}/metrics.json"
        if not numb3rs_path.exists():
            print(
                json.dumps(
                    {"dataset": "numb3rs", "exposure": exposure, "status": "started"}
                ),
                flush=True,
            )
            payload = full_google.numb3rs_metrics(
                model, tokenizer, numb3rs_rows, checkpoint, state, exposure
            )
            write_json(
                numb3rs_path, adaptation_metrics(payload, state, checkpoint, exposure)
            )
            print(
                json.dumps(
                    {"dataset": "numb3rs", "exposure": exposure, "status": "completed"}
                ),
                flush=True,
            )
        else:
            print(
                json.dumps(
                    {"dataset": "numb3rs", "exposure": exposure, "status": "skipped"}
                ),
                flush=True,
            )
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()


def main() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "already_running"}), flush=True)
            return
        _run()


if __name__ == "__main__":
    main()
