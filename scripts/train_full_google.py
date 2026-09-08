"""Continue the 10k checkpoint over every remaining eligible Google train row."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import tempfile
import time
from array import array
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
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
SHUFFLE_START_BATCH_OFFSET = 26_000
SHUFFLE_START_EXPOSURE = (
    FIRST_UNSEEN_RECORD_ID + SHUFFLE_START_BATCH_OFFSET * BATCH_SIZE
)
SHUFFLE_SEED = "premove-itn/full-google/remaining-after-218000/v1"
TRAINING_RECORD_IDS_SHA256 = (
    "80cec20da44e0adb024e6ae55b59317daafdac5daf7ad915248371b95510d26f"
)
SHUFFLE_SOURCE_CHECKPOINT = (
    OUTPUT / f"checkpoint_shuffle_source_{SHUFFLE_START_EXPOSURE:06d}.pt"
)
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


def _fingerprint(identity: dict[str, object]) -> str:
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def ordered_run_fingerprint(dataset_sha256: str) -> str:
    """Identify the original source-ordered full-run sequence."""
    return _fingerprint(
        {
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "dataset_sha256": dataset_sha256,
            "partition": "train",
            "first_unseen_record_id": FIRST_UNSEEN_RECORD_ID,
            "max_source_chars": MAX_SOURCE_CHARS,
            "batch_size": BATCH_SIZE,
            "bucket_window": BUCKET_WINDOW,
            "optimizer": "AdamW",
            "optimizer_fused": True,
        }
    )


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
        "optimizer": "AdamW",
        "optimizer_fused": True,
        "shuffle_start_batch_offset": SHUFFLE_START_BATCH_OFFSET,
        "shuffle_seed": SHUFFLE_SEED,
        "training_record_ids_sha256": TRAINING_RECORD_IDS_SHA256,
    }
    return _fingerprint(identity)


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


def original_record_batches() -> Iterator[list[tuple[int, dict[str, object]]]]:
    """Yield the original stable, locally bucketed record batches."""
    window: list[tuple[int, dict[str, object]]] = []
    for item in eligible_rows():
        window.append(item)
        if len(window) == BUCKET_WINDOW:
            yield from window_batches(window)
            window.clear()
    if window:
        yield from window_batches(window)


def _line_offsets() -> array[int]:
    offsets = array("Q")
    with DATASET.open("rb") as handle:
        while True:
            offset = handle.tell()
            if not handle.readline():
                break
            offsets.append(offset)
    return offsets


def _rows_by_record_id(
    record_ids: list[int],
) -> Iterator[tuple[int, dict[str, object]]]:
    offsets = _line_offsets()
    with DATASET.open("rb") as handle:
        for record_id in record_ids:
            handle.seek(offsets[record_id])
            yield record_id, json.loads(handle.readline())


def training_record_batches() -> Iterator[list[tuple[int, dict[str, object]]]]:
    """Keep the trained prefix fixed and shuffle only its exact unseen suffix."""
    unseen_record_ids: list[int] = []
    original_batches = original_record_batches()
    for batch_offset, records in enumerate(original_batches):
        if batch_offset < SHUFFLE_START_BATCH_OFFSET:
            yield records
        else:
            unseen_record_ids.extend(record_id for record_id, _ in records)

    random.Random(SHUFFLE_SEED).shuffle(unseen_record_ids)
    if not unseen_record_ids:
        return
    rows = _rows_by_record_id(unseen_record_ids)
    while window := list(islice(rows, BUCKET_WINDOW)):
        for records in window_batches(window):
            yield records


def record_batches() -> Iterator[tuple[tuple[str, str], ...]]:
    """Yield source-target batches in the identified training order."""
    for records in training_record_batches():
        yield record_pairs(records)


def audit_training_order(
    *,
    expected_count: int,
    expected_distribution: Counter[str],
    last_record_id: int,
) -> tuple[Counter[str], Counter[str]]:
    """Prove that the ordered prefix and shuffled suffix cover each row once."""
    seen = bytearray(last_record_id + 1)
    digest = hashlib.sha256()
    prefix_distribution: Counter[str] = Counter()
    suffix_distribution: Counter[str] = Counter()
    transition_examples = SHUFFLE_START_BATCH_OFFSET * BATCH_SIZE
    count = 0
    for batch in training_record_batches():
        for record_id, row in batch:
            if record_id > last_record_id or seen[record_id]:
                raise RuntimeError(f"duplicate or invalid training record {record_id}")
            seen[record_id] = 1
            digest.update(record_id.to_bytes(4, "big"))
            distribution = (
                prefix_distribution
                if count < transition_examples
                else suffix_distribution
            )
            distribution[str(row["kind"])] += 1
            count += 1
    if count != expected_count:
        raise RuntimeError(
            f"training order contains {count} records; expected {expected_count}"
        )
    if prefix_distribution + suffix_distribution != expected_distribution:
        raise RuntimeError("training order distribution does not match eligible data")
    actual_digest = digest.hexdigest()
    if actual_digest != TRAINING_RECORD_IDS_SHA256:
        raise RuntimeError(
            f"training record order hash {actual_digest} does not match "
            f"{TRAINING_RECORD_IDS_SHA256}"
        )
    return prefix_distribution, suffix_distribution


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
    return tuple((str(row["text"]), str(row["expected_text"])) for _, row in rows)


def _initialize_worker() -> None:
    global _WORKER_TOKENIZER
    torch.set_num_threads(1)
    _WORKER_TOKENIZER = load_model_tokenizer()


def load_resume_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_run_fingerprint: str | None = None,
    transition_run_fingerprint: str | None = None,
    transition_batch_offset: int | None = None,
) -> tuple[Path, CheckpointState]:
    """Load the newest valid full-run generation, then the source checkpoint."""
    full_run_candidates = (
        CHECKPOINT,
        previous_checkpoint_path(CHECKPOINT),
        SHUFFLE_SOURCE_CHECKPOINT,
    )
    candidates = full_run_candidates
    if not any(path.exists() for path in full_run_candidates):
        candidates += (SOURCE_CHECKPOINT,)
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
        expected = state.run_fingerprint == expected_run_fingerprint
        valid_ordered_prefix = (
            state.run_fingerprint == transition_run_fingerprint
            and transition_batch_offset is not None
            and state.batch_offset <= transition_batch_offset
        )
        if path != SOURCE_CHECKPOINT and not (expected or valid_ordered_prefix):
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
        if valid_ordered_prefix and not expected:
            state = replace(state, run_fingerprint=expected_run_fingerprint)
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
    ordered_fingerprint = ordered_run_fingerprint(dataset_sha256)
    prefix_distribution, shuffled_distribution = audit_training_order(
        expected_count=remaining_count,
        expected_distribution=remaining_distribution,
        last_record_id=last_id,
    )
    shuffled_remaining_examples = (
        remaining_count - SHUFFLE_START_BATCH_OFFSET * BATCH_SIZE
    )
    if shuffled_remaining_examples <= 0:
        raise RuntimeError("shuffle transition must leave unseen training examples")
    manifest = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": dataset_sha256,
        "run_fingerprint": fingerprint,
        "parent_run_fingerprint": ordered_fingerprint,
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
        "batch_order": "ordered_prefix_then_deterministic_remaining_shuffle",
        "shuffle_start_batch_offset": SHUFFLE_START_BATCH_OFFSET,
        "shuffle_start_exposure": SHUFFLE_START_EXPOSURE,
        "shuffle_seed": SHUFFLE_SEED,
        "training_record_ids_sha256": TRAINING_RECORD_IDS_SHA256,
        "shuffled_remaining_examples": shuffled_remaining_examples,
        "ordered_prefix_distribution": dict(sorted(prefix_distribution.items())),
        "shuffled_remaining_distribution": dict(sorted(shuffled_distribution.items())),
        "shuffle_source_checkpoint": str(SHUFFLE_SOURCE_CHECKPOINT.relative_to(ROOT)),
        "checkpoint_every_batches": CHECKPOINT_EVERY_BATCHES,
        "progress_every_batches": PROGRESS_EVERY_BATCHES,
        "prefetch_workers": PREFETCH_WORKERS,
        "prefetch_batches": PREFETCH_BATCHES,
        "optimizer": "AdamW",
        "optimizer_fused": True,
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
        transition_run_fingerprint=ordered_fingerprint,
        transition_batch_offset=SHUFFLE_START_BATCH_OFFSET,
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
        resumed_batch_source=lambda offset: batch_source(start_batch_offset=offset),
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
