import json

import pytest

from premove_itn.inference_artifact import (
    ARCHITECTURE_VERSION,
    ARTIFACT_SCHEMA_VERSION,
    load_inference_artifact,
)
from premove_itn.labels import SPAN_KINDS


def test_inference_artifact_fails_closed_on_digest_mismatch(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"not a model")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "architecture_version": ARCHITECTURE_VERSION,
                "state_dict_file": "model.safetensors",
                "supported_span_kinds": [kind.value for kind in SPAN_KINDS],
                "base_model": "test",
                "base_model_revision": "test",
                "tokenizer_revision": "test",
            }
        )
    )
    (tmp_path / "provenance.json").write_text(
        json.dumps(
            {
                "artifact_sha256": "wrong",
                "architecture": ARCHITECTURE_VERSION,
                "base_model": "test",
                "base_model_revision": "test",
                "tokenizer_revision": "test",
            }
        )
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_inference_artifact(tmp_path)
