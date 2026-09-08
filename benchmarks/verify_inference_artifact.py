"""Verify that an inference artifact is identical to the frozen checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from benchmarks.run_comparison import PremoveITNBackend, read_rows
except ModuleNotFoundError:  # direct execution from the benchmarks directory
    from run_comparison import PremoveITNBackend, read_rows

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/fixtures/normalization_regression.json"
DATASET = ROOT / "eval/voice_agent_itn/voice_agent_eval.jsonl"
PRODUCTION = ROOT / "data/models/production.json"
DEFAULT_ARTIFACT = ROOT / "artifacts/premove-itn-contextual-v0.1.0"
DEFAULT_OUTPUT = (
    ROOT / "eval/voice_agent_itn/results/inference-artifact-v0.1.0/acceptance.json"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def prediction_digest(predictions: list[str]) -> str:
    digest = hashlib.sha256()
    for prediction in predictions:
        encoded = prediction.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def state_digest(state: Any) -> str:
    """Hash tensor values, names, shapes, and dtypes in stable order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_state_digest(path: Path) -> str:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return state_digest(checkpoint["model_state_dict"])


def artifact_state_digest(path: Path) -> str:
    from safetensors.torch import load_file

    return state_digest(load_file(str(path), device="cpu"))


def run_predictions(
    backend: PremoveITNBackend, rows: list[dict[str, Any]]
) -> list[str]:
    backend.initialize()
    backend.warmup()
    predictions: list[str] = []
    try:
        for row in rows:
            prediction, _details = backend.normalize(str(row["text"]))
            predictions.append(prediction)
    finally:
        backend.close()
    return predictions


def compare_dataset(
    name: str,
    rows: list[dict[str, Any]],
    checkpoint_predictions: list[str],
    artifact_predictions: list[str],
) -> dict[str, Any]:
    mismatches = [
        {
            "id": str(row["id"]),
            "checkpoint": checkpoint,
            "artifact": artifact,
        }
        for row, checkpoint, artifact in zip(
            rows, checkpoint_predictions, artifact_predictions, strict=True
        )
        if checkpoint != artifact
    ]
    expected_matches = None
    if all("expected_text" in row for row in rows):
        expected_matches = sum(
            prediction == str(row["expected_text"])
            for row, prediction in zip(rows, artifact_predictions, strict=True)
        )
    return {
        "name": name,
        "rows": len(rows),
        "identical": len(mismatches) == 0,
        "identical_rows": len(rows) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "artifact_expected_matches": expected_matches,
        "checkpoint_prediction_sha256": prediction_digest(checkpoint_predictions),
        "artifact_prediction_sha256": prediction_digest(artifact_predictions),
    }


def write_report(output: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Inference artifact acceptance",
        "",
        "The acceptance gate compares the frozen production checkpoint with the",
        "inference-only `premove-itn-contextual` artifact. Both use the same",
        "candidate builder and decoder. No training is performed.",
        "",
        f"- Artifact: `{result['artifact']}`",
        f"- Artifact SHA-256: `{result['artifact_sha256']}`",
        f"- Tensor state identical: **{result['state_dict_identical']}**",
        "",
        "| Corpus | Rows | Identical predictions | Artifact expected matches |",
        "|---|---:|---:|---:|",
    ]
    for item in result["corpora"]:
        expected = item["artifact_expected_matches"]
        expected_text = (
            "not scored" if expected is None else f"{expected}/{item['rows']}"
        )
        lines.append(
            f"| `{item['name']}` | {item['rows']} | "
            f"{item['identical_rows']}/{item['rows']} | {expected_text} |"
        )
    lines.extend(
        [
            "",
            "A release is acceptable only when both corpora report all rows",
            "identical and the tensor state is identical.",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    production = read_json(PRODUCTION)
    checkpoint = ROOT / str(production["checkpoint"])
    golden_rows = read_json(GOLDEN)
    benchmark_rows = read_rows(DATASET)
    all_rows = golden_rows + benchmark_rows
    checkpoint_state = checkpoint_state_digest(checkpoint)
    artifact_state = artifact_state_digest(args.artifact / "model.safetensors")
    checkpoint_predictions = run_predictions(PremoveITNBackend(checkpoint), all_rows)
    artifact_predictions = run_predictions(
        PremoveITNBackend(None, args.artifact.resolve()), all_rows
    )
    golden_count = len(golden_rows)
    corpora = [
        compare_dataset(
            "golden",
            golden_rows,
            checkpoint_predictions[:golden_count],
            artifact_predictions[:golden_count],
        ),
        compare_dataset(
            "frozen_voice_agent",
            benchmark_rows,
            checkpoint_predictions[golden_count:],
            artifact_predictions[golden_count:],
        ),
    ]
    provenance = read_json(args.artifact / "provenance.json")
    try:
        artifact_name = str(args.artifact.resolve().relative_to(ROOT))
    except ValueError:
        artifact_name = str(args.artifact.resolve())
    result = {
        "artifact": artifact_name,
        "artifact_sha256": provenance["artifact_sha256"],
        "source_checkpoint_sha256": provenance["source_checkpoint_sha256"],
        "checkpoint_state_sha256": checkpoint_state,
        "artifact_state_sha256": artifact_state,
        "state_dict_identical": checkpoint_state == artifact_state,
        "corpora": corpora,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_report(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["state_dict_identical"] or any(
        not corpus["identical"] for corpus in corpora
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
