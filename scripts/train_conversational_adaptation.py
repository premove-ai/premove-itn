"""Adapt the selected 378k model on the deterministic 44k schedule."""

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
SOURCE = ROOT / "data/models/google_selected_378k/checkpoint.pt"
SOURCE_MANIFEST = ROOT / "data/models/google_selected_378k/manifest.json"
SCHEDULE_ROOT = ROOT / "data/generated/conversational_adaptation_44k"
OUTPUT = ROOT / "data/generated/conversational_adaptation_44k_v2"
SCHEDULE = SCHEDULE_ROOT / "training_schedule.jsonl"
SCHEDULE_MANIFEST = SCHEDULE_ROOT / "schedule_manifest.json"
TRAINING_MANIFEST = OUTPUT / "training_manifest.json"
CHECKPOINT = OUTPUT / "checkpoint.pt"
METRICS = OUTPUT / "metrics.json"
PROGRESS = OUTPUT / "progress.json"
MILESTONES = OUTPUT / "milestones"
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8
CHECKPOINT_EVERY_BATCHES = 250
PROGRESS_EVERY_BATCHES = 25
PREFETCH_WORKERS = 1
PREFETCH_BATCHES = 2
MILESTONE_EXPOSURES = (10_000, 20_000, 30_000, 44_058)
GOOGLE_EXACT_FLOOR = 0.9474
CANDIDATE_BEARING_KEEP_FLOOR = 0.7871
_WORKER_TOKENIZER: OffsetTokenizer | None = None


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(identity: dict[str, object]) -> str:
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def run_fingerprint(schedule_sha256: str, source_sha256: str) -> str:
    return _fingerprint(
        {
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "schedule_sha256": schedule_sha256,
            "source_checkpoint_sha256": source_sha256,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "loss_weighting": "none",
        }
    )


def _read_schedule() -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest = json.loads(SCHEDULE_MANIFEST.read_text())
    artifact = manifest["artifacts"]["training_schedule.jsonl"]
    if artifact["bytes"] != SCHEDULE.stat().st_size:
        raise RuntimeError("training schedule size does not match its manifest")
    if artifact["sha256"] != sha256(SCHEDULE):
        raise RuntimeError("training schedule hash does not match its manifest")
    entries = [json.loads(line) for line in SCHEDULE.open()]
    if manifest["records"] != len(entries):
        raise RuntimeError("training schedule record count does not match its manifest")
    expected_buckets = Counter(manifest["bucket_distribution"])
    actual_buckets = Counter(str(entry["bucket"]) for entry in entries)
    if actual_buckets != expected_buckets:
        raise RuntimeError("training schedule bucket distribution is invalid")
    identities = [
        (str(entry["source_dataset"]), int(entry["record_id"])) for entry in entries
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("training schedule contains duplicate record references")
    if any(entry.get("partition") != "train" for entry in entries):
        raise RuntimeError("training schedule contains a non-train reference")
    return entries, manifest


def _selected_rows(path: Path, record_ids: set[int]) -> dict[int, dict[str, object]]:
    selected = {}
    with path.open() as handle:
        for record_id, line in enumerate(handle):
            if record_id in record_ids:
                selected[record_id] = json.loads(line)
    missing = record_ids - selected.keys()
    if missing:
        raise RuntimeError(
            f"{path} is missing scheduled record IDs: {sorted(missing)[:10]}"
        )
    return selected


def resolve_schedule() -> tuple[list[dict[str, object]], dict[str, object]]:
    entries, manifest = _read_schedule()
    input_hashes = manifest["inputs"]
    for relative_path, expected_hash in input_hashes.items():
        if sha256(ROOT / relative_path) != expected_hash:
            raise RuntimeError(f"schedule input hash changed: {relative_path}")
    by_dataset: dict[str, set[int]] = {}
    for entry in entries:
        dataset = str(entry["source_dataset"])
        by_dataset.setdefault(dataset, set()).add(int(entry["record_id"]))
    rows_by_dataset = {}
    for relative_path, record_ids in by_dataset.items():
        path = ROOT / relative_path
        rows_by_dataset[relative_path] = _selected_rows(path, record_ids)

    resolved = []
    for entry in entries:
        dataset = str(entry["source_dataset"])
        record_id = int(entry["record_id"])
        row = rows_by_dataset[dataset][record_id]
        if row.get("partition") != "train":
            raise RuntimeError("scheduled source row is no longer in train")
        if row.get("kind") != entry.get("kind"):
            raise RuntimeError("scheduled source row kind changed")
        resolved.append(row)
    return resolved, manifest


def record_batches(
    rows: list[dict[str, object]],
) -> Iterator[tuple[tuple[str, str], ...]]:
    iterator = iter(rows)
    while batch := list(islice(iterator, BATCH_SIZE)):
        yield tuple((str(row["text"]), str(row["expected_text"])) for row in batch)


def _initialize_worker() -> None:
    global _WORKER_TOKENIZER
    torch.set_num_threads(1)
    _WORKER_TOKENIZER = load_model_tokenizer()


def _prepare_worker_batch(records: tuple[tuple[str, str], ...]) -> TrainingBatch:
    tokenizer = _WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("training worker tokenizer is not initialized")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise RuntimeError("tokenizer must define a pad token")
    return prepare_training_batch(records, tokenizer, pad_token_id=pad_token_id)


def batch_source(
    rows: list[dict[str, object]], *, start_batch_offset: int = 0
) -> Iterator[TrainingBatch]:
    if start_batch_offset < 0:
        raise ValueError("start batch offset must be non-negative")
    remaining = islice(record_batches(rows), start_batch_offset, None)
    yield from ordered_process_prefetch(
        _prepare_worker_batch,
        remaining,
        workers=PREFETCH_WORKERS,
        max_pending=PREFETCH_BATCHES,
        initializer=_initialize_worker,
    )


def _preserve(source: Path, destination: Path, *, replace: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (source.samefile(destination) or not replace):
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(slots=True)
class ProgressReporter:
    total_batches: int
    total_examples: int
    initial_batch_offset: int
    initial_examples: int
    started: float = field(default_factory=time.monotonic)

    def __call__(self, progress: TrainingProgress) -> None:
        elapsed = time.monotonic() - self.started
        session_batches = progress.batch_offset - self.initial_batch_offset
        session_examples = progress.examples - self.initial_examples
        batch_rate = session_batches / elapsed if elapsed else 0.0
        example_rate = session_examples / elapsed if elapsed else 0.0
        remaining_batches = self.total_batches - progress.batch_offset
        eta = remaining_batches / batch_rate if batch_rate else None
        if (
            progress.batch_offset % PROGRESS_EVERY_BATCHES == 0
            or progress.batch_offset == self.total_batches
        ):
            _write_json_atomic(
                PROGRESS,
                {
                    "status": "running",
                    "completed_batches": progress.batch_offset,
                    "total_batches": self.total_batches,
                    "durable_batch_offset": (
                        progress.batch_offset
                        // CHECKPOINT_EVERY_BATCHES
                        * CHECKPOINT_EVERY_BATCHES
                    ),
                    "optimizer_steps": progress.optimizer_steps,
                    "completed_examples": progress.examples,
                    "total_examples": self.total_examples,
                    "elapsed_active_seconds": elapsed,
                    "batches_per_second": batch_rate,
                    "examples_per_second": example_rate,
                    "eta_seconds": eta,
                },
            )
        if (
            progress.examples in MILESTONE_EXPOSURES
            and progress.examples < self.total_examples
            and CHECKPOINT.is_file()
        ):
            _preserve(
                CHECKPOINT,
                MILESTONES / f"checkpoint_{progress.examples:06d}.pt",
            )


def _load_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    fingerprint: str,
) -> tuple[Path, CheckpointState] | None:
    failures = []
    for path in (CHECKPOINT, previous_checkpoint_path(CHECKPOINT)):
        if not path.exists():
            continue
        try:
            state = load_checkpoint(path, model, optimizer, map_location="cpu")
        except (
            OSError,
            EOFError,
            RuntimeError,
            ValueError,
            pickle.UnpicklingError,
        ) as error:
            failures.append(f"{path}: {error}")
            continue
        if state.run_fingerprint != fingerprint:
            failures.append(f"{path}: run fingerprint mismatch")
            continue
        return path, state
    if failures:
        raise RuntimeError("no valid adaptation checkpoint: " + "; ".join(failures))
    return None


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    rows, schedule_manifest = resolve_schedule()
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    source_sha256 = sha256(SOURCE)
    if source_manifest["checkpoint_sha256"] != source_sha256:
        raise RuntimeError("selected source checkpoint hash mismatch")
    schedule_sha256 = schedule_manifest["artifacts"]["training_schedule.jsonl"][
        "sha256"
    ]
    fingerprint = run_fingerprint(schedule_sha256, source_sha256)
    distribution = Counter(str(row["kind"]) for row in rows)
    total_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    training_manifest = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_checkpoint_sha256": source_sha256,
        "schedule": str(SCHEDULE.relative_to(ROOT)),
        "schedule_sha256": schedule_sha256,
        "run_fingerprint": fingerprint,
        "records": len(rows),
        "kind_distribution": dict(sorted(distribution.items())),
        "bucket_distribution": schedule_manifest["bucket_distribution"],
        "batch_size": BATCH_SIZE,
        "checkpoint_every_batches": CHECKPOINT_EVERY_BATCHES,
        "milestone_exposures": list(MILESTONE_EXPOSURES),
        "optimizer": "AdamW",
        "optimizer_state": "fresh_at_start",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss_weighting": "none",
        "validation_included": False,
        "test_included": False,
        "selection_rule": {
            "google_exact_floor": GOOGLE_EXACT_FLOOR,
            "candidate_bearing_keep_floor": CANDIDATE_BEARING_KEEP_FLOOR,
            "eligible_checkpoint_objective": "conversational_balanced_accuracy",
            "regression_guards": ["Golden", "Numb3rs"],
        },
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
    }
    _write_json_atomic(TRAINING_MANIFEST, training_manifest)
    _write_json_atomic(PROGRESS, {"status": "starting"})

    model = load_candidate_scorer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    optimizer = create_optimizer(
        model,
        TrainingConfig(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
    )
    resume = _load_resume(model, optimizer, fingerprint)
    if resume is None:
        source_state = load_checkpoint(SOURCE, model, map_location="cpu")
        if source_state.batch_offset <= 0:
            raise RuntimeError("selected 378k checkpoint metadata is not mid-run")
        resume_state = None
        initial_batch_offset = 0
        initial_examples = 0
    else:
        resume_path, resume_state = resume
        if resume_state.epoch == 1 and resume_state.batch_offset == 0:
            message = {"status": "already_complete", "checkpoint": str(resume_path)}
            _write_json_atomic(PROGRESS, message)
            print(json.dumps(message), flush=True)
            return
        initial_batch_offset = resume_state.batch_offset
        initial_examples = resume_state.metrics[-1].examples
        if initial_examples in MILESTONE_EXPOSURES:
            _preserve(
                resume_path,
                MILESTONES / f"checkpoint_{initial_examples:06d}.pt",
            )

    reporter = ProgressReporter(
        total_batches=total_batches,
        total_examples=len(rows),
        initial_batch_offset=initial_batch_offset,
        initial_examples=initial_examples,
    )
    history = train(
        model,
        lambda: batch_source(rows),
        optimizer,
        epochs=1,
        device=device,
        checkpoint_path=CHECKPOINT,
        checkpoint_every_batches=CHECKPOINT_EVERY_BATCHES,
        training_distribution=distribution,
        resume_state=resume_state,
        resumed_batch_source=lambda offset: batch_source(
            rows, start_batch_offset=offset
        ),
        progress_callback=reporter,
        run_fingerprint=fingerprint,
    )
    _preserve(
        CHECKPOINT,
        MILESTONES / "checkpoint_044058.pt",
        replace=True,
    )
    result = {
        **training_manifest,
        "device": str(device),
        "epoch_metrics": [asdict(item) for item in history],
    }
    _write_json_atomic(METRICS, result)
    final_progress = json.loads(PROGRESS.read_text())
    final_progress.update(status="completed", eta_seconds=0.0)
    _write_json_atomic(PROGRESS, final_progress)
    print(
        json.dumps({"checkpoint": str(CHECKPOINT), "metrics": str(METRICS)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
