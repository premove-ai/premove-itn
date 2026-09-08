"""Public API for the frozen contextual inverse text normalizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model_inputs import MODEL_NAME, MODEL_REVISION

DEFAULT_MODEL_ID = "premove-itn/premove-itn"
DEFAULT_RELEASE = "v0.1.0"
DEFAULT_REVISION = "80bda5e2e1fe9542aa628597090242df57c1a157"
EXPECTED_ARTIFACT_SHA256 = (
    "119c0f19767b61446e04da1f8f01a001edf97a47a66965e7146db2483b4937a1"
)
SUPPORTED_DEVICES = frozenset({"auto", "cpu", "mps", "cuda"})


def _resolve_device(device: str) -> tuple[Any, Any]:
    """Return the selected torch device and the imported torch module."""
    if device not in SUPPORTED_DEVICES:
        options = ", ".join(sorted(SUPPORTED_DEVICES))
        raise ValueError(f"unsupported device {device!r}; choose one of {options}")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "contextual inference requires the model dependencies; "
            "install the package's model dependency group"
        ) from exc

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), torch
        return torch.device("cpu"), torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(device), torch


def _resolve_artifact(model_id: str | Path, revision: str) -> Path:
    """Resolve a local artifact or a cached/downloaded Hub snapshot."""
    local_path = Path(model_id).expanduser()
    if local_path.is_dir():
        return local_path

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "Hugging Face loading requires the model dependencies; "
            "install the package's model dependency group"
        ) from exc

    try:
        snapshot = snapshot_download(repo_id=str(model_id), revision=revision)
    except Exception as exc:  # keep the public error tied to the requested model
        raise RuntimeError(
            f"could not resolve Hugging Face model {model_id!r} at revision "
            f"{revision!r}"
        ) from exc
    snapshot_path = Path(snapshot)
    resolved_revision = snapshot_path.name
    if resolved_revision != DEFAULT_REVISION:
        raise RuntimeError(
            "Hugging Face model resolved to an unexpected commit: "
            f"expected {DEFAULT_REVISION}, got {resolved_revision}"
        )
    return snapshot_path


def _verify_release_metadata(artifact_dir: Path) -> None:
    """Reject a snapshot that is not the requested frozen release."""
    provenance_path = artifact_dir / "provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"invalid inference artifact provenance: {provenance_path}"
        ) from exc
    if not isinstance(provenance, dict):
        raise RuntimeError("inference artifact provenance must be a JSON object")
    if provenance.get("artifact_version") != DEFAULT_RELEASE:
        raise RuntimeError(
            "inference artifact release mismatch: "
            f"expected {DEFAULT_RELEASE!r}, "
            f"got {provenance.get('artifact_version')!r}"
        )
    if provenance.get("hub_revision") != DEFAULT_RELEASE:
        raise RuntimeError(
            "inference artifact Hub revision mismatch: "
            f"expected {DEFAULT_RELEASE!r}, "
            f"got {provenance.get('hub_revision')!r}"
        )
    if provenance.get("hub_repository") != DEFAULT_MODEL_ID:
        raise RuntimeError(
            "inference artifact repository mismatch: "
            f"expected {DEFAULT_MODEL_ID!r}, got {provenance.get('hub_repository')!r}"
        )
    if provenance.get("artifact_sha256") != EXPECTED_ARTIFACT_SHA256:
        raise RuntimeError("inference artifact model-file digest mismatch")
    if provenance.get("base_model") != MODEL_NAME:
        raise RuntimeError(
            "inference artifact base model mismatch: "
            f"expected {MODEL_NAME!r}, got {provenance.get('base_model')!r}"
        )
    if provenance.get("base_model_revision") != MODEL_REVISION:
        raise RuntimeError("inference artifact base model revision mismatch")


class PremoveITN:
    """A loaded contextual inverse text normalizer.

    The model and tokenizer are loaded once by :meth:`from_pretrained` and
    reused for every call to :meth:`normalize`.
    """

    __slots__ = ("_torch", "device", "model", "model_id", "revision", "tokenizer")

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        device: Any,
        model_id: str,
        revision: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._torch = torch_module
        self.device = device
        self.model_id = model_id
        self.revision = revision

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | Path = DEFAULT_MODEL_ID,
        *,
        revision: str = DEFAULT_REVISION,
        device: str = "auto",
    ) -> PremoveITN:
        """Load one frozen Hub release and keep it warm in memory.

        ``model_id`` may also be a local inference-artifact directory for
        offline use. Hub snapshots use the normal Hugging Face cache, so a
        second initialization reuses the cached revision. The default revision
        is the immutable Hub commit for release ``v0.1.0``; callers may use the
        release tag because its resolved commit is verified before loading.
        """
        if not revision:
            raise ValueError("revision must not be empty")
        selected_device, torch_module = _resolve_device(device)
        artifact_dir = _resolve_artifact(model_id, revision)
        _verify_release_metadata(artifact_dir)

        from .inference_artifact import load_inference_artifact

        try:
            loaded = load_inference_artifact(
                artifact_dir,
                device=str(selected_device),
            )
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            raise RuntimeError(
                f"could not load inference artifact for {model_id!r} at "
                f"revision {revision!r}"
            ) from exc
        return cls(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            torch_module=torch_module,
            device=selected_device,
            model_id=str(model_id),
            revision=DEFAULT_REVISION,
        )

    def normalize(self, text: str) -> str:
        """Normalize one transcript and return the resulting string."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text or text.isspace():
            return text

        from .candidate_scorer import collate_candidate_batch
        from .candidates import build_candidate_graph
        from .decoder import decode_candidates
        from .model_inputs import encode_candidates

        candidates = build_candidate_graph(text)
        if not candidates:
            return text
        encoded = encode_candidates(text, candidates, self.tokenizer)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise RuntimeError("inference tokenizer has no pad token")
        batch = collate_candidate_batch(
            [(encoded, candidates)],
            pad_token_id=pad_token_id,
        ).to(self.device)
        with self._torch.inference_mode():
            scores = self.model(batch)
        return decode_candidates(text, candidates, scores).text


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_RELEASE",
    "DEFAULT_REVISION",
    "EXPECTED_ARTIFACT_SHA256",
    "PremoveITN",
]
