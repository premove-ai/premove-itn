"""Load the frozen inference-only contextual model artifact.

The artifact contains the complete trained ``CandidateScorer`` state dict,
the base encoder configuration, and the exact tokenizer files.  It does not
contain optimizer state or any training metadata.  The loader is deliberately
strict: a missing file, unsupported schema, or changed model digest aborts
before inference starts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA_VERSION = 1
ARCHITECTURE_VERSION = "premove-candidate-scorer-v1"
STATE_DICT_FILE = "model.safetensors"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid inference artifact file: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"inference artifact file must contain an object: {path}")
    return value


@dataclass(frozen=True, slots=True)
class LoadedInferenceArtifact:
    """A fully loaded inference model and its tokenizer."""

    model: Any
    tokenizer: Any
    config: dict[str, Any]
    provenance: dict[str, Any]


def load_inference_artifact(
    artifact_dir: str | Path,
    *,
    device: str = "cpu",
    verify_hash: bool = True,
) -> LoadedInferenceArtifact:
    """Load a frozen artifact and return an eval-mode scorer plus tokenizer.

    ``artifact_dir`` can be a downloaded Hugging Face model snapshot or a
    local export directory.  The artifact is self-contained apart from the
    Python model dependencies.  ``verify_hash`` is intended only for trusted
    development fixtures; callers should keep its default value for releases.
    """
    root = Path(artifact_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"inference artifact directory not found: {root}")
    config = _read_json(root / "config.json")
    provenance = _read_json(root / "provenance.json")
    if config.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("unsupported inference artifact schema")
    if config.get("architecture_version") != ARCHITECTURE_VERSION:
        raise RuntimeError("unsupported inference artifact architecture")
    from premove_itn.labels import SPAN_KINDS

    expected_kinds = [kind.value for kind in SPAN_KINDS]
    if config.get("supported_span_kinds") != expected_kinds:
        raise RuntimeError("inference artifact SpanKind order does not match package")
    if provenance.get("architecture") != ARCHITECTURE_VERSION:
        raise RuntimeError("inference artifact provenance architecture mismatch")
    if (
        config.get("base_model") != provenance.get("base_model")
        or config.get("base_model_revision")
        != provenance.get("base_model_revision")
        or config.get("tokenizer_revision") != provenance.get("tokenizer_revision")
    ):
        raise RuntimeError("inference artifact model provenance does not match config")
    state_path = root / str(config.get("state_dict_file", STATE_DICT_FILE))
    if not state_path.is_file():
        raise FileNotFoundError(f"inference model state is missing: {state_path}")
    if verify_hash:
        expected = provenance.get("artifact_sha256")
        actual = _sha256_file(state_path)
        if expected != actual:
            raise RuntimeError(
                "inference artifact SHA-256 mismatch: "
                f"expected {expected}, got {actual}"
            )

    # Imports stay local so installing the deterministic Rust/Python API does
    # not require torch, Transformers, or safetensors.
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from premove_itn.candidate_scorer import CandidateScorer

    base_config_path = root / str(config.get("base_config_file", "base_config.json"))
    if not base_config_path.is_file():
        raise FileNotFoundError(f"base encoder config is missing: {base_config_path}")
    base_config = AutoConfig.from_pretrained(base_config_path)
    encoder = AutoModel.from_config(base_config)
    encoder = encoder.to(dtype=torch.float32)
    model = CandidateScorer(
        encoder,
        kind_embedding_size=int(config["kind_embedding_size"]),
        scorer_hidden_size=int(config["scorer_hidden_size"]),
        dropout=float(config["dropout"]),
    )
    state_dict = load_file(str(state_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        root / str(config.get("tokenizer_dir", "tokenizer")),
        local_files_only=True,
        use_fast=True,
        model_max_length=int(config["model_max_tokens"]),
    )
    if not tokenizer.is_fast:
        raise RuntimeError("inference artifact tokenizer is not a fast tokenizer")
    return LoadedInferenceArtifact(model, tokenizer, config, provenance)


__all__ = [
    "ARCHITECTURE_VERSION",
    "ARTIFACT_SCHEMA_VERSION",
    "LoadedInferenceArtifact",
    "load_inference_artifact",
]
