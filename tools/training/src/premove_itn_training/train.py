from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from premove_itn_training.alignment import EncodedDataset, LabelPaddingCollator
from premove_itn_training.dataset import FrozenSplit, load_frozen_split
from premove_itn_training.labels import BIO_LABELS, ID_TO_LABEL, LABEL_TO_ID
from premove_itn_training.metrics import score_aligned_predictions

_DEFAULT_MODEL = "microsoft/deberta-v3-large"
_DEFAULT_REVISION = "64a8c8eab3e352a784c658aef62be1662607476f"


@dataclass(frozen=True, slots=True)
class TrainingRunConfig:
    dataset_manifest: Path
    output_directory: Path
    model_name: str = _DEFAULT_MODEL
    model_revision: str = _DEFAULT_REVISION
    tokenizer_name: str = _DEFAULT_MODEL
    tokenizer_revision: str = _DEFAULT_REVISION
    max_length: int = 72
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 4
    seed: int = 1337


def _resolved_revision(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value:
        return value
    return fallback


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _load_tokenizer(config: TrainingRunConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        revision=config.tokenizer_revision,
        use_fast=True,
        trust_remote_code=False,
    )
    if not tokenizer.is_fast:
        raise ValueError("Model V1 requires a fast tokenizer with word_ids()")
    return tokenizer


def _encode_split(
    split: FrozenSplit, tokenizer: Any, *, max_length: int
) -> tuple[EncodedDataset, EncodedDataset]:
    train = EncodedDataset(split.train, tokenizer, max_length=max_length)
    validation = EncodedDataset(
        split.validation, tokenizer, max_length=max_length
    )
    return train, validation


def audit_model_v1(config: TrainingRunConfig) -> dict[str, object]:
    split = load_frozen_split(config.dataset_manifest)
    tokenizer = _load_tokenizer(config)
    train, validation = _encode_split(split, tokenizer, max_length=config.max_length)
    return {
        "dataset": {
            "manifest": str(split.manifest_path),
            "train_sha256": split.train_sha256,
            "validation_sha256": split.validation_sha256,
        },
        "tokenizer": {
            "name": config.tokenizer_name,
            "requested_revision": config.tokenizer_revision,
            "resolved_revision": _resolved_revision(
                tokenizer.init_kwargs.get("_commit_hash"),
                config.tokenizer_revision,
            ),
        },
        "max_length": config.max_length,
        "train_lengths": asdict(train.audit),
        "validation_lengths": asdict(validation.audit),
    }


def _best_epoch(log_history: list[dict[str, Any]]) -> float | None:
    evaluations = [
        item
        for item in log_history
        if isinstance(item.get("eval_span_f1"), (int, float))
        and isinstance(item.get("epoch"), (int, float))
    ]
    if not evaluations:
        return None
    return float(max(evaluations, key=lambda item: item["eval_span_f1"])["epoch"])


def _json_numbers(values: dict[str, Any]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values.items():
        if isinstance(value, bool | int | float | str) or value is None:
            result[key] = value
        elif hasattr(value, "item"):
            result[key] = value.item()
        else:
            result[key] = str(value)
    return result


def train_model_v1(config: TrainingRunConfig) -> dict[str, object]:
    """Train and save one reproducible contextual BIO baseline."""
    import torch
    import transformers
    from transformers import (
        AutoModelForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.train_batch_size < 1 or config.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.output_directory.exists() and any(config.output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty model directory: "
            f"{config.output_directory}"
        )

    set_seed(config.seed)
    split = load_frozen_split(config.dataset_manifest)
    tokenizer = _load_tokenizer(config)
    train_dataset, validation_dataset = _encode_split(
        split, tokenizer, max_length=config.max_length
    )
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        num_labels=len(BIO_LABELS),
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    arguments = TrainingArguments(
        output_dir=str(config.output_directory),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        seed=config.seed,
        data_seed=config.seed,
        load_best_model_at_end=True,
        metric_for_best_model="span_f1",
        greater_is_better=True,
        save_total_limit=2,
        save_safetensors=True,
        report_to="none",
        remove_unused_columns=True,
    )

    def argmax_logits(logits: Any, _labels: Any) -> Any:
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(prediction: Any) -> dict[str, float]:
        return score_aligned_predictions(prediction.predictions, prediction.label_ids)

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=LabelPaddingCollator(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=argmax_logits,
    )
    train_result = trainer.train()
    final_metrics = trainer.evaluate()

    best_directory = config.output_directory / "best"
    trainer.save_model(best_directory)
    tokenizer.save_pretrained(best_directory)
    metadata: dict[str, object] = {
        "kind": "contextual_bio_model_v1_training_run",
        "schema_version": 1,
        "git": _git_state(),
        "dataset": {
            "manifest": str(split.manifest_path),
            "train_records": len(split.train),
            "train_sha256": split.train_sha256,
            "validation_records": len(split.validation),
            "validation_sha256": split.validation_sha256,
        },
        "labels": {
            "bio_labels": list(BIO_LABELS),
            "label_to_id": LABEL_TO_ID,
        },
        "model": {
            "name": config.model_name,
            "requested_revision": config.model_revision,
            "resolved_revision": _resolved_revision(
                getattr(model.config, "_commit_hash", None), config.model_revision
            ),
        },
        "tokenizer": {
            "name": config.tokenizer_name,
            "requested_revision": config.tokenizer_revision,
            "resolved_revision": _resolved_revision(
                tokenizer.init_kwargs.get("_commit_hash"),
                config.tokenizer_revision,
            ),
        },
        "hyperparameters": {
            "max_length": config.max_length,
            "train_batch_size": config.train_batch_size,
            "eval_batch_size": config.eval_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_train_batch_size_per_process": (
                config.train_batch_size * config.gradient_accumulation_steps
            ),
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "epochs": config.epochs,
            "optimizer": "adamw_torch",
            "scheduler": "linear",
            "seed": config.seed,
        },
        "sequence_lengths": {
            "train": asdict(train_dataset.audit),
            "validation": asdict(validation_dataset.audit),
        },
        "selection": {
            "metric": "strict span micro-F1",
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_epoch": _best_epoch(trainer.state.log_history),
            "best_validation_span_f1": trainer.state.best_metric,
        },
        "metrics": {
            "train": _json_numbers(train_result.metrics),
            "validation": _json_numbers(final_metrics),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "best_model_directory": str(best_directory),
        "golden_benchmark_used_for_selection": False,
    }
    config.output_directory.mkdir(parents=True, exist_ok=True)
    (config.output_directory / "run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Model V1 contextual BIO token classifier."
    )
    parser.add_argument("dataset_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--model-name", default=_DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=_DEFAULT_REVISION)
    parser.add_argument("--tokenizer-name", default=_DEFAULT_MODEL)
    parser.add_argument("--tokenizer-revision", default=_DEFAULT_REVISION)
    parser.add_argument("--max-length", type=int, default=72)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Verify data and tokenizer lengths without loading or training a model.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = TrainingRunConfig(
        dataset_manifest=args.dataset_manifest,
        output_directory=args.output_directory,
        model_name=args.model_name,
        model_revision=args.model_revision,
        tokenizer_name=args.tokenizer_name,
        tokenizer_revision=args.tokenizer_revision,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        epochs=args.epochs,
        seed=args.seed,
    )
    result = audit_model_v1(config) if args.audit_only else train_model_v1(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
