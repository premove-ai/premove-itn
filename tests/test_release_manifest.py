from pathlib import Path

import pytest

from scripts.generate_release_manifest import build_manifest, expected_names


def test_expected_names_define_seven_public_artifacts() -> None:
    names = expected_names("0.1.0")
    assert len(names) == 7
    assert "premove_itn-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl" in names
    assert "premove_itn-0.1.0.tar.gz" in names


def test_build_manifest_hashes_exact_artifact_set(tmp_path: Path) -> None:
    for name in expected_names("0.1.0"):
        (tmp_path / name).write_bytes(name.encode())
    manifest = build_manifest(
        tmp_path, version="0.1.0", git_commit="abc123", model_sha256="model123"
    )
    assert manifest["git_commit"] == "abc123"
    assert manifest["model_sha256"] == "model123"
    assert len(manifest["artifacts"]) == 7


def test_build_manifest_rejects_incomplete_set(tmp_path: Path) -> None:
    (tmp_path / "premove_itn-0.1.0.tar.gz").write_bytes(b"sdist")
    with pytest.raises(RuntimeError, match="artifact set mismatch"):
        build_manifest(
            tmp_path,
            version="0.1.0",
            git_commit="abc123",
            model_sha256="model123",
        )
