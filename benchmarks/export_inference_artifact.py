"""Export the selected production checkpoint as a frozen inference artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

from premove_itn.inference_artifact import ARCHITECTURE_VERSION
from premove_itn.labels import SPAN_KINDS
from premove_itn.model_inputs import MODEL_MAX_TOKENS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCTION = ROOT / "data/models/production.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def package_version() -> str:
    try:
        return importlib.metadata.version("premove-itn")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def write_readme(root: Path, provenance: dict[str, Any]) -> None:
    root.joinpath("README.md").write_text(
        """---
library_name: premove-itn
tags:
  - inverse-text-normalization
  - voice-agents
  - custom-code
license: mit
---

# Premove ITN Contextual v0.1.0

This is the first frozen, inference-only artifact for Premove ITN's
structured-value contextual scorer. It contains the complete trained model
parameters in `model.safetensors`, the exact DeBERTa configuration, and the
tokenizer files used by the selected production checkpoint. Optimizer state,
schedulers, counters, and training datasets are not included.

The artifact is loaded through the `premove-itn` package. It is not a generic
Transformers model; use the package loader so candidate generation and scoring
stay aligned with the release implementation:

```python
from premove_itn.inference_artifact import load_inference_artifact

artifact = load_inference_artifact(".")
model = artifact.model
tokenizer = artifact.tokenizer
```

The model uses `microsoft/deberta-v3-large` at the exact revision recorded in
`provenance.json`. The base model is referenced rather than redistributed.
Microsoft's model card lists the base model and tokenizer under the MIT
license; retain that attribution when redistributing this artifact.

This release is tagged `v0.1.0`. Do not move or overwrite that tag after
publication. The artifact digest and source checkpoint lineage are recorded in
`provenance.json`.

## Scope

The supported span kinds are the values listed in `config.json`. This artifact
selects deterministic Rust candidates with a contextual scorer. It does not
contain training data or the frozen VoiceAgent benchmark dataset.

## Release identity

- Artifact version: `v0.1.0`
- Architecture: `premove-candidate-scorer-v1`
- Package version: `"""
        + str(provenance["package_version"])
        + "`\n"
    )


def export(
    checkpoint: Path,
    production_path: Path,
    output: Path,
    *,
    force: bool = False,
) -> Path:
    production = read_json(production_path)
    expected_checkpoint = Path(str(production["checkpoint"]))
    if not expected_checkpoint.is_absolute():
        expected_checkpoint = ROOT / expected_checkpoint
    if checkpoint.resolve() != expected_checkpoint.resolve():
        raise RuntimeError(
            "refusing to export a non-production checkpoint: "
            f"expected {expected_checkpoint}, got {checkpoint}"
        )
    expected_sha = str(production["checkpoint_sha256"])
    actual_sha = sha256_file(checkpoint)
    if actual_sha != expected_sha:
        raise RuntimeError(
            "production checkpoint SHA-256 mismatch: "
            f"expected {expected_sha}, got {actual_sha}"
        )
    if output.exists():
        if not force:
            raise FileExistsError(
                "output exists; pass --force only for a disposable local export: "
                f"{output}"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_data, dict) or not isinstance(
        checkpoint_data.get("model_state_dict"), dict
    ):
        raise RuntimeError("production checkpoint has no model_state_dict")
    state_dict = checkpoint_data["model_state_dict"]
    # The output is deliberately built from only this mapping. The complete
    # training checkpoint is never copied into the artifact directory.

    model_name = str(production["model_name"])
    model_revision = str(production["model_revision"])
    base_config = AutoConfig.from_pretrained(model_name, revision=model_revision)
    base_config.to_json_file(output / "base_config.json")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        use_fast=True,
        model_max_length=MODEL_MAX_TOKENS,
    )
    tokenizer_dir = output / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    save_file(state_dict, str(output / "model.safetensors"), metadata={"format": "pt"})
    artifact_sha = sha256_file(output / "model.safetensors")
    provenance = {
        "artifact_version": "v0.1.0",
        "hub_repository": "premove-ai/premove-itn-contextual",
        "hub_revision": "v0.1.0",
        "publication_status": "ready_for_upload",
        "artifact_sha256": artifact_sha,
        "artifact_sha256_definition": "SHA-256 of model.safetensors",
        "source_checkpoint": str(checkpoint.relative_to(ROOT)),
        "source_checkpoint_sha256": actual_sha,
        "benchmark_checkpoint_sha256": actual_sha,
        "production_manifest": str(
            (ROOT / str(production["manifest"])).relative_to(ROOT)
        ),
        "base_model": model_name,
        "base_model_revision": model_revision,
        "tokenizer_revision": model_revision,
        "architecture": ARCHITECTURE_VERSION,
        "package_version": package_version(),
        "supported_span_kinds": [kind.value for kind in SPAN_KINDS],
        "training_lineage": {
            "combined_exposure": 418000,
            "run_fingerprint": str(
                read_json(ROOT / str(production["manifest"]))["run_fingerprint"]
            ),
        },
        "redistribution": {
            "project_license": "MIT",
            "base_model_license": "MIT (as listed by microsoft/deberta-v3-large)",
            "base_model_weights_included": False,
            "tokenizer_files_included": True,
        },
    }
    config = {
        "artifact_schema_version": 1,
        "architecture_version": ARCHITECTURE_VERSION,
        "state_dict_file": "model.safetensors",
        "base_config_file": "base_config.json",
        "tokenizer_dir": "tokenizer",
        "base_model": model_name,
        "base_model_revision": model_revision,
        "tokenizer_revision": model_revision,
        "kind_embedding_size": int(state_dict["kind_projection.weight"].shape[0]),
        "scorer_hidden_size": int(state_dict["scoring_head.0.weight"].shape[0]),
        "dropout": 0.1,
        "model_max_tokens": MODEL_MAX_TOKENS,
        "supported_span_kinds": [kind.value for kind in SPAN_KINDS],
    }
    dump_json(output / "config.json", config)
    dump_json(output / "provenance.json", provenance)
    write_readme(output, provenance)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    production = read_json(args.production)
    checkpoint = args.checkpoint or (ROOT / str(production["checkpoint"]))
    export(
        checkpoint.resolve(),
        args.production.resolve(),
        args.output.resolve(),
        force=args.force,
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
