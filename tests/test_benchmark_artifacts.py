import hashlib
import json

import pytest

from benchmarks.run_comparison import verify_frozen_artifacts


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    checkpoint = tmp_path / "checkpoint.pt"
    extension = tmp_path / "_rust.so"
    dataset.write_bytes(b'{"text":"hello"}\n')
    checkpoint.write_bytes(b"checkpoint")
    extension.write_bytes(b"native extension")
    production = tmp_path / "production.json"
    production.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "model_name": "model/name",
                "model_revision": "model-revision",
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact": {
                    "path": str(dataset),
                    "sha256": _sha256(dataset),
                }
            }
        )
    )
    return dataset, checkpoint, extension, production, manifest


def test_verify_frozen_artifacts_returns_recordable_digests(tmp_path):
    dataset, checkpoint, extension, production, manifest = _artifacts(tmp_path)

    result = verify_frozen_artifacts(
        dataset,
        production,
        manifest,
        {
            "profile": "release",
            "debug_assertions": False,
        },
        extension,
        "model/name",
        "model-revision",
    )

    assert result["dataset_sha256"] == _sha256(dataset)
    assert result["checkpoint_sha256"] == _sha256(checkpoint)
    assert result["rust_extension_sha256"] == _sha256(extension)


def test_verify_frozen_artifacts_rejects_changed_dataset(tmp_path):
    dataset, checkpoint, extension, production, manifest = _artifacts(tmp_path)
    dataset.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="dataset SHA-256 mismatch"):
        verify_frozen_artifacts(
            dataset,
            production,
            manifest,
            {"profile": "release", "debug_assertions": False},
            extension,
            "model/name",
            "model-revision",
        )


def test_verify_frozen_artifacts_rejects_changed_checkpoint(tmp_path):
    dataset, checkpoint, extension, production, manifest = _artifacts(tmp_path)
    checkpoint.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="checkpoint SHA-256 mismatch"):
        verify_frozen_artifacts(
            dataset,
            production,
            manifest,
            {"profile": "release", "debug_assertions": False},
            extension,
            "model/name",
            "model-revision",
        )


def test_verify_frozen_artifacts_rejects_model_revision_mismatch(tmp_path):
    dataset, checkpoint, extension, production, manifest = _artifacts(tmp_path)

    with pytest.raises(RuntimeError, match="model revision mismatch"):
        verify_frozen_artifacts(
            dataset,
            production,
            manifest,
            {"profile": "release", "debug_assertions": False},
            extension,
            "model/name",
            "changed-revision",
        )


def test_verify_frozen_artifacts_rejects_debug_extension(tmp_path):
    dataset, checkpoint, extension, production, manifest = _artifacts(tmp_path)

    with pytest.raises(RuntimeError, match="non-release Rust extension"):
        verify_frozen_artifacts(
            dataset,
            production,
            manifest,
            {"profile": "debug", "debug_assertions": True},
            extension,
            "model/name",
            "model-revision",
        )
