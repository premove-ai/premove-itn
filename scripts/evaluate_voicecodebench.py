"""Evaluate both retained checkpoints on VoiceCodeBench annotated entities."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from premove_itn import SpanKind, _rust, normalize_sentence, target_is_reachable
from premove_itn.candidate_scorer import load_candidate_scorer
from premove_itn.model_inputs import MODEL_NAME, MODEL_REVISION, load_model_tokenizer
from premove_itn.training import load_checkpoint

if __package__:
    from scripts.evaluate_full_google_milestones import (
        evaluate_model,
        sha256,
        write_json,
    )
    from scripts.evaluate_polynorm import summarize
else:
    from evaluate_full_google_milestones import evaluate_model, sha256, write_json
    from evaluate_polynorm import summarize

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "3ccea73877a159eb2a8b17304148c325c5fe5061"
DATASET = ROOT / "data/external/voice-code-bench/data/metadata.jsonl"
REACHABILITY_CACHE = ROOT / "data/generated/voicecodebench/reachability.json"
REACHABILITY_IMPLEMENTATION_FILES = (
    ("rust_extension", Path(_rust.__file__)),
    ("candidates", ROOT / "src/premove_itn/candidates.py"),
    ("labels", ROOT / "src/premove_itn/labels.py"),
)
DEFAULT_RUNS = (
    (
        ROOT / "data/models/google_selected_378k/checkpoint.pt",
        ROOT / "data/generated/voicecodebench/google_378000/metrics.json",
    ),
    (
        ROOT / "data/models/conversational_selected_20k/checkpoint.pt",
        ROOT / "data/generated/voicecodebench/checkpoint_020000/metrics.json",
    ),
)
SEMANTIC_KINDS = {
    "currency_amount": SpanKind.MONEY,
    "date": SpanKind.DATE,
    "ip_address": SpanKind.PHONE,
    "measurement": SpanKind.MEASUREMENT,
    "percentage": SpanKind.MEASUREMENT,
    "phone_extension": SpanKind.PHONE,
    "phone_number": SpanKind.PHONE,
    "plain_number": SpanKind.CARDINAL,
    "port_number": SpanKind.CARDINAL,
    "time": SpanKind.TIME,
}


def load_rows(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source = [json.loads(line) for line in path.read_text().splitlines() if line]
    sentence_rows = []
    entity_rows = []
    for row in source:
        audio_id = str(row["audio_id"])
        transcripts = row["transcripts"]
        sentence_rows.append(
            {
                "id": audio_id,
                "category": str(row["domain"]),
                "text": str(transcripts["acoustic"]),
                "original_text": str(transcripts["canonical"]),
            }
        )
        for entity in row["entities"]:
            entity_rows.append(
                {
                    "id": str(entity["id"]),
                    "category": str(entity["type"]),
                    "text": str(entity["acoustic"]),
                    "original_text": str(entity["canonical"]),
                }
            )
    sentence_ids = {row["id"] for row in sentence_rows}
    if not sentence_rows or len(sentence_ids) != len(sentence_rows):
        raise ValueError("VoiceCodeBench sentence IDs must be present and unique")
    if not entity_rows or len({row["id"] for row in entity_rows}) != len(entity_rows):
        raise ValueError("VoiceCodeBench entity IDs must be present and unique")
    return sentence_rows, entity_rows


def reachability_fingerprint(files: Sequence[tuple[str, Path]]) -> str:
    """Hash the loaded implementation that determines graph reachability."""
    digest = hashlib.sha256()
    for name, path in files:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def reachability_cache_matches(
    cached: dict[str, object],
    identities: list[str],
    dataset_sha256: str,
    implementation_fingerprint: str,
) -> bool:
    """Accept only a cache produced by the same data and loaded implementation."""
    return (
        cached.get("source_revision") == SOURCE_REVISION
        and cached.get("dataset_sha256") == dataset_sha256
        and cached.get("identities") == identities
        and cached.get("reachability_fingerprint") == implementation_fingerprint
    )


def evaluate_rows(
    model, tokenizer, rows: Sequence[dict[str, object]]
) -> dict[str, object]:
    identities = [str(row["id"]) for row in rows]
    dataset_sha256 = sha256(DATASET)
    implementation_fingerprint = reachability_fingerprint(
        REACHABILITY_IMPLEMENTATION_FILES
    )
    reachable = None
    if REACHABILITY_CACHE.is_file():
        cached = json.loads(REACHABILITY_CACHE.read_text())
        if reachability_cache_matches(
            cached, identities, dataset_sha256, implementation_fingerprint
        ):
            reachable = [bool(value) for value in cached["reachable"]]
    if reachable is None:
        reachable = [
            target_is_reachable(str(row["text"]), str(row["original_text"]))
            for row in rows
        ]
        write_json(
            REACHABILITY_CACHE,
            {
                "source_revision": SOURCE_REVISION,
                "dataset_sha256": dataset_sha256,
                "identities": identities,
                "reachability_fingerprint": implementation_fingerprint,
                "reachable": reachable,
            },
        )
    predictions, candidate_count, elapsed = evaluate_model(model, tokenizer, rows)
    baseline = [normalize_sentence(str(row["text"])) for row in rows]
    return {
        "candidate_count": candidate_count,
        "elapsed_seconds": elapsed,
        "trained": summarize(rows, predictions, reachable, SEMANTIC_KINDS),
        "deterministic_baseline": summarize(rows, baseline, reachable, SEMANTIC_KINDS),
    }


def evaluate(checkpoint: Path, output: Path) -> dict[str, object]:
    sentence_rows, entity_rows = load_rows(DATASET)
    tokenizer = load_model_tokenizer()
    model = load_candidate_scorer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    state = load_checkpoint(checkpoint, model, map_location="cpu")
    model.eval()
    payload = {
        "benchmark_role": "external_test_only_structured_value_benchmark",
        "source": "besimple-ai/voice-code-bench",
        "source_revision": SOURCE_REVISION,
        "license": "MIT",
        "evaluation_only": True,
        "training_use": False,
        "protocol": "provided_entity_acoustic_text_to_canonical_text",
        "scoring_note": (
            "Entity exact evaluates each supplied entity in isolation. This is a "
            "Premove realizer and selection diagnostic, not the benchmark's official "
            "raw-audio ASR protocol or contextual CTEM."
        ),
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": sha256(DATASET),
        "records": len(sentence_rows),
        "entities": len(entity_rows),
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
        "device": str(device),
        "entity_isolated": evaluate_rows(model, tokenizer, entity_rows),
    }
    write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not DATASET.is_file():
        raise RuntimeError(
            "VoiceCodeBench is missing; run scripts/download_voicecodebench.py"
        )
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
                    "entity_isolated": payload["entity_isolated"]["trained"]["all"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
