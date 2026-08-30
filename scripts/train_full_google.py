"""Continue the 10k checkpoint over every remaining eligible Google train row."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import torch

from premove_itn.candidate_scorer import load_candidate_scorer
from premove_itn.model_inputs import MODEL_NAME, MODEL_REVISION, load_model_tokenizer
from premove_itn.training import (
    TrainingConfig,
    create_optimizer,
    load_checkpoint,
    train,
)
from premove_itn.training_batch import TrainingBatch, prepare_training_batch

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/google_tn/dataset_1/dataset.jsonl"
SOURCE_CHECKPOINT = ROOT / "data/generated/first_run_10000/checkpoint_epoch_1.pt"
OUTPUT = ROOT / "data/generated/full_google_train"
CHECKPOINT = OUTPUT / "checkpoint.pt"
MANIFEST = OUTPUT / "manifest.json"
METRICS = OUTPUT / "metrics.json"
FIRST_UNSEEN_RECORD_ID = 10_000
MAX_SOURCE_CHARS = 300
BATCH_SIZE = 8
BUCKET_WINDOW = 256
CHECKPOINT_EVERY_STEPS = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_rows() -> Iterator[tuple[int, dict[str, object]]]:
    with DATASET.open() as handle:
        for record_id, line in enumerate(handle):
            row = json.loads(line)
            if (
                record_id >= FIRST_UNSEEN_RECORD_ID
                and row["partition"] == "train"
                and len(str(row["text"])) <= MAX_SOURCE_CHARS
            ):
                yield record_id, row


def scan_dataset() -> tuple[int, Counter[str], Counter[str], int, int]:
    remaining_distribution: Counter[str] = Counter()
    cumulative_distribution: Counter[str] = Counter()
    first_id = -1
    last_id = -1
    count = 0
    with DATASET.open() as handle:
        for record_id, line in enumerate(handle):
            row = json.loads(line)
            if row["partition"] != "train" or len(str(row["text"])) > MAX_SOURCE_CHARS:
                continue
            cumulative_distribution[str(row["kind"])] += 1
            if record_id < FIRST_UNSEEN_RECORD_ID:
                continue
            if first_id < 0:
                first_id = record_id
            last_id = record_id
            count += 1
            remaining_distribution[str(row["kind"])] += 1
    return (
        count,
        remaining_distribution,
        cumulative_distribution,
        first_id,
        last_id,
    )


def batch_source(
    tokenizer: object,
    pad_token_id: int,
) -> Iterator[TrainingBatch]:
    """Yield deterministic batches without retaining prepared epoch data."""
    window: list[tuple[int, dict[str, object]]] = []
    for item in eligible_rows():
        window.append(item)
        if len(window) == BUCKET_WINDOW:
            for records in window_batches(window):
                yield prepare_records(records, tokenizer, pad_token_id)
            window.clear()
    if window:
        for records in window_batches(window):
            yield prepare_records(records, tokenizer, pad_token_id)


def window_batches(
    window: list[tuple[int, dict[str, object]]],
) -> Iterator[list[tuple[int, dict[str, object]]]]:
    window.sort(key=lambda item: (len(str(item[1]["text"])), item[0]))
    for start in range(0, len(window), BATCH_SIZE):
        yield window[start : start + BATCH_SIZE]


def prepare_records(
    rows: list[tuple[int, dict[str, object]]],
    tokenizer: object,
    pad_token_id: int,
) -> TrainingBatch:
    return prepare_training_batch(
        tuple((str(row["text"]), str(row["expected_text"])) for _, row in rows),
        tokenizer,
        pad_token_id=pad_token_id,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (
        remaining_count,
        remaining_distribution,
        full_distribution,
        first_id,
        last_id,
    ) = scan_dataset()
    manifest = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": sha256(DATASET),
        "partition": "train",
        "validation_included": False,
        "test_included": False,
        "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
        "already_trained_examples": FIRST_UNSEEN_RECORD_ID,
        "remaining_examples": remaining_count,
        "remaining_record_ids": [first_id, last_id],
        "remaining_distribution": dict(sorted(remaining_distribution.items())),
        "cumulative_training_distribution": dict(sorted(full_distribution.items())),
        "max_source_chars": MAX_SOURCE_CHARS,
        "batch_size": BATCH_SIZE,
        "bucket_window": BUCKET_WINDOW,
        "checkpoint_every_optimizer_steps": CHECKPOINT_EVERY_STEPS,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    tokenizer = load_model_tokenizer()
    if tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer must define a pad token")
    model = load_candidate_scorer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    optimizer = create_optimizer(model, TrainingConfig())
    resume_path = CHECKPOINT if CHECKPOINT.exists() else SOURCE_CHECKPOINT
    resume_state = load_checkpoint(resume_path, model, optimizer, map_location="cpu")
    if (
        resume_path == CHECKPOINT
        and resume_state.epoch >= 2
        and resume_state.batch_offset == 0
    ):
        message = {"status": "already_complete", "checkpoint": str(CHECKPOINT)}
        print(json.dumps(message), flush=True)
        return
    history = train(
        model,
        lambda: batch_source(tokenizer, tokenizer.pad_token_id),
        optimizer,
        epochs=1,
        device=device,
        checkpoint_path=CHECKPOINT,
        checkpoint_every_steps=CHECKPOINT_EVERY_STEPS,
        training_distribution=full_distribution,
        resume_state=resume_state,
    )
    result = {
        **manifest,
        "device": str(device),
        "epoch_metrics": [asdict(item) for item in history],
    }
    METRICS.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    message = {"checkpoint": str(CHECKPOINT), "metrics": str(METRICS)}
    print(json.dumps(message), flush=True)


if __name__ == "__main__":
    main()
