"""Continue the 10k checkpoint over every remaining eligible Google train row."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path

import torch

from premove_itn.candidate_scorer import load_candidate_scorer
from premove_itn.model_inputs import (
    MODEL_NAME,
    MODEL_REVISION,
    OffsetTokenizer,
    load_model_tokenizer,
)
from premove_itn.training import (
    CheckpointState,
    TrainingConfig,
    TrainingProgress,
    create_optimizer,
    load_checkpoint,
    previous_checkpoint_path,
    train,
)
from premove_itn.training_batch import (
    TrainingBatch,
    ordered_process_prefetch,
    prepare_training_batch,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/google_tn/dataset_1/dataset.jsonl"
SOURCE_CHECKPOINT = ROOT / "data/generated/first_run_10000/checkpoint_epoch_1.pt"
OUTPUT = ROOT / "data/generated/full_google_train"
CHECKPOINT = OUTPUT / "checkpoint.pt"
MANIFEST = OUTPUT / "manifest.json"
METRICS = OUTPUT / "metrics.json"
PROGRESS = OUTPUT / "progress.json"
FIRST_UNSEEN_RECORD_ID = 10_000
MAX_SOURCE_CHARS = 300
BATCH_SIZE = 8
BUCKET_WINDOW = 256
CHECKPOINT_EVERY_BATCHES = 1_000
PREFETCH_WORKERS = 2
PREFETCH_BATCHES = 8
PROGRESS_EVERY_BATCHES = 25
_WORKER_TOKENIZER: OffsetTokenizer | None = None


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Replace one JSON artifact only after its complete contents are durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


@dataclass(slots=True)
class ProgressReporter:
    """Persist measured full-run progress without materializing model tensors."""

    path: Path
    total_batches: int
    total_examples: int
    initial_batch_offset: int
    initial_examples: int
    checkpoint_every_batches: int
    report_every_batches: int = PROGRESS_EVERY_BATCHES
    _started: float = field(init=False)

    def __post_init__(self) -> None:
        if self.total_batches <= 0 or self.total_examples <= 0:
            raise ValueError("progress totals must be positive")
        if self.report_every_batches <= 0 or self.checkpoint_every_batches <= 0:
            raise ValueError("progress intervals must be positive")
        self._started = time.monotonic()

    def __call__(self, progress: TrainingProgress) -> None:
        if (
            progress.batch_offset % self.report_every_batches
            and progress.batch_offset != self.total_batches
        ):
            return
        elapsed = time.monotonic() - self._started
        session_batches = progress.batch_offset - self.initial_batch_offset
        session_examples = progress.examples - self.initial_examples
        batch_rate = session_batches / elapsed if elapsed else 0.0
        example_rate = session_examples / elapsed if elapsed else 0.0
        remaining_batches = self.total_batches - progress.batch_offset
        eta = remaining_batches / batch_rate if batch_rate else None
        durable_batch_offset = max(
            self.initial_batch_offset,
            progress.batch_offset
            // self.checkpoint_every_batches
            * self.checkpoint_every_batches,
        )
        payload: dict[str, object] = {
            "status": "running",
            "epoch": progress.epoch,
            "completed_batches": progress.batch_offset,
            "total_batches": self.total_batches,
            "durable_batch_offset": durable_batch_offset,
            "optimizer_steps": progress.optimizer_steps,
            "completed_examples": progress.examples,
            "total_examples": self.total_examples,
            "elapsed_active_seconds": elapsed,
            "batches_per_second": batch_rate,
            "examples_per_second": example_rate,
            "eta_seconds": eta,
        }
        _write_json_atomic(self.path, payload)
        if (
            progress.batch_offset % self.checkpoint_every_batches == 0
            or progress.batch_offset == self.total_batches
        ):
            print(json.dumps(payload, sort_keys=True), flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_fingerprint(dataset_sha256: str) -> str:
    """Identify every input that determines the full-run batch sequence."""
    identity = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "dataset_sha256": dataset_sha256,
        "partition": "train",
        "first_unseen_record_id": FIRST_UNSEEN_RECORD_ID,
        "max_source_chars": MAX_SOURCE_CHARS,
        "batch_size": BATCH_SIZE,
        "bucket_window": BUCKET_WINDOW,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


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


def record_batches() -> Iterator[tuple[tuple[str, str], ...]]:
    """Yield stable source-target batches before expensive preparation."""
    window: list[tuple[int, dict[str, object]]] = []
    for item in eligible_rows():
        window.append(item)
        if len(window) == BUCKET_WINDOW:
            for records in window_batches(window):
                yield record_pairs(records)
            window.clear()
    if window:
        for records in window_batches(window):
            yield record_pairs(records)


def batch_source(*, start_batch_offset: int = 0) -> Iterator[TrainingBatch]:
    """Prepare bounded batches concurrently while preserving exact order."""
    if start_batch_offset < 0:
        raise ValueError("start batch offset must be non-negative")
    remaining = islice(record_batches(), start_batch_offset, None)
    yield from ordered_process_prefetch(
        _prepare_worker_batch,
        remaining,
        workers=PREFETCH_WORKERS,
        max_pending=PREFETCH_BATCHES,
        initializer=_initialize_worker,
    )


def window_batches(
    window: list[tuple[int, dict[str, object]]],
) -> Iterator[list[tuple[int, dict[str, object]]]]:
    window.sort(key=lambda item: (len(str(item[1]["text"])), item[0]))
    for start in range(0, len(window), BATCH_SIZE):
        yield window[start : start + BATCH_SIZE]


def record_pairs(
    rows: list[tuple[int, dict[str, object]]],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(row["text"]), str(row["expected_text"])) for _, row in rows
    )


def _initialize_worker() -> None:
    global _WORKER_TOKENIZER
    torch.set_num_threads(1)
    _WORKER_TOKENIZER = load_model_tokenizer()


def load_resume_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_run_fingerprint: str | None = None,
) -> tuple[Path, CheckpointState]:
    """Load the newest valid full-run generation, then the source checkpoint."""
    candidates = (
        CHECKPOINT,
        previous_checkpoint_path(CHECKPOINT),
        SOURCE_CHECKPOINT,
    )
    load_errors = (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError)
    failures: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            state = load_checkpoint(path, model, optimizer, map_location="cpu")
        except load_errors as error:
            failures.append(f"{path}: {error}")
            print(
                json.dumps(
                    {
                        "status": "checkpoint_rejected",
                        "path": str(path),
                        "error": str(error),
                    }
                ),
                flush=True,
            )
            continue
        if (
            path != SOURCE_CHECKPOINT
            and state.run_fingerprint != expected_run_fingerprint
        ):
            failures.append(f"{path}: run fingerprint mismatch")
            print(
                json.dumps(
                    {
                        "status": "checkpoint_rejected",
                        "path": str(path),
                        "error": "run fingerprint mismatch",
                    }
                ),
                flush=True,
            )
            continue
        return path, state
    detail = "; ".join(failures) if failures else "no checkpoint files exist"
    raise RuntimeError(f"no valid training checkpoint: {detail}")


def _prepare_worker_batch(records: tuple[tuple[str, str], ...]) -> TrainingBatch:
    tokenizer = _WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("training worker tokenizer is not initialized")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise RuntimeError("tokenizer must define a pad token")
    return prepare_training_batch(
        records,
        tokenizer,
        pad_token_id=pad_token_id,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _write_json_atomic(PROGRESS, {"status": "starting"})
    (
        remaining_count,
        remaining_distribution,
        full_distribution,
        first_id,
        last_id,
    ) = scan_dataset()
    dataset_sha256 = sha256(DATASET)
    fingerprint = run_fingerprint(dataset_sha256)
    manifest = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": dataset_sha256,
        "run_fingerprint": fingerprint,
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
        "checkpoint_every_batches": CHECKPOINT_EVERY_BATCHES,
        "progress_every_batches": PROGRESS_EVERY_BATCHES,
        "prefetch_workers": PREFETCH_WORKERS,
        "prefetch_batches": PREFETCH_BATCHES,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    model = load_candidate_scorer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    optimizer = create_optimizer(model, TrainingConfig())
    resume_path, resume_state = load_resume_checkpoint(
        model,
        optimizer,
        expected_run_fingerprint=fingerprint,
    )
    if (
        resume_path in (CHECKPOINT, previous_checkpoint_path(CHECKPOINT))
        and resume_state.epoch >= 2
        and resume_state.batch_offset == 0
    ):
        message = {"status": "already_complete", "checkpoint": str(CHECKPOINT)}
        _write_json_atomic(PROGRESS, message)
        print(json.dumps(message), flush=True)
        return
    total_batches = (remaining_count + BATCH_SIZE - 1) // BATCH_SIZE
    initial_examples = (
        resume_state.metrics[-1].examples if resume_state.batch_offset else 0
    )
    progress_reporter = ProgressReporter(
        path=PROGRESS,
        total_batches=total_batches,
        total_examples=remaining_count,
        initial_batch_offset=resume_state.batch_offset,
        initial_examples=initial_examples,
        checkpoint_every_batches=CHECKPOINT_EVERY_BATCHES,
    )
    history = train(
        model,
        batch_source,
        optimizer,
        epochs=1,
        device=device,
        checkpoint_path=CHECKPOINT,
        checkpoint_every_batches=CHECKPOINT_EVERY_BATCHES,
        training_distribution=full_distribution,
        resume_state=resume_state,
        resumed_batch_source=lambda offset: batch_source(
            start_batch_offset=offset
        ),
        progress_callback=progress_reporter,
        run_fingerprint=fingerprint,
    )
    result = {
        **manifest,
        "device": str(device),
        "epoch_metrics": [asdict(item) for item in history],
    }
    METRICS.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    final_progress = json.loads(PROGRESS.read_text())
    final_progress.update(status="completed", eta_seconds=0.0)
    _write_json_atomic(PROGRESS, final_progress)
    message = {"checkpoint": str(CHECKPOINT), "metrics": str(METRICS)}
    print(json.dumps(message), flush=True)


if __name__ == "__main__":
    main()
