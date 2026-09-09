"""Verify the identities that define the frozen v0.1.0 release boundary."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

from premove_itn.contextual import (
    DEFAULT_MODEL_ID,
    DEFAULT_RELEASE,
    DEFAULT_REVISION,
    EXPECTED_ARTIFACT_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval" / "voice_agent_itn" / "voice_agent_eval.jsonl"
MANIFEST = ROOT / "eval" / "voice_agent_itn" / "manifest.json"
FROZEN_DIGEST = ROOT / "eval" / "voice_agent_itn" / "FROZEN.sha256"
RUN = ROOT / "eval" / "voice_agent_itn" / "results" / "first-evaluation" / "run.json"
ACCEPTANCE = (
    ROOT
    / "eval"
    / "voice_agent_itn"
    / "results"
    / "inference-artifact-v0.1.0"
    / "acceptance.json"
)
ARTIFACT_DOC = ROOT / "docs" / "inference-artifact.md"
PYPROJECT = ROOT / "pyproject.toml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def verify_frozen_boundaries(root: Path = ROOT) -> None:
    """Fail if the frozen dataset, artifact, or release identity diverges."""
    manifest = _read_json(root / MANIFEST.relative_to(ROOT))
    run = _read_json(root / RUN.relative_to(ROOT))
    acceptance = _read_json(root / ACCEPTANCE.relative_to(ROOT))
    run_artifacts = run.get("artifacts")
    if not isinstance(run_artifacts, dict):
        raise RuntimeError("First Evaluation run has no artifact record")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise RuntimeError("frozen benchmark manifest has no artifact record")

    dataset = root / DATASET.relative_to(ROOT)
    expected_dataset_sha = str(artifact.get("sha256"))
    if _sha256(dataset) != expected_dataset_sha:
        raise RuntimeError("frozen benchmark dataset digest does not match manifest")
    frozen_line = (root / FROZEN_DIGEST.relative_to(ROOT)).read_text().strip()
    frozen_sha, frozen_name = frozen_line.split(maxsplit=1)
    if frozen_name != dataset.name or frozen_sha != expected_dataset_sha:
        raise RuntimeError("FROZEN.sha256 does not match the benchmark manifest")
    if run_artifacts.get("dataset_sha256") != expected_dataset_sha:
        raise RuntimeError("First Evaluation dataset digest diverges from manifest")

    checkpoint_sha = manifest.get("frozen_checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", checkpoint_sha
    ):
        raise RuntimeError("frozen checkpoint digest is missing or malformed")
    if run_artifacts.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("First Evaluation checkpoint digest diverges from manifest")
    if acceptance.get("source_checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError(
            "inference acceptance checkpoint digest diverges from manifest"
        )
    if acceptance.get("artifact_sha256") != EXPECTED_ARTIFACT_SHA256:
        raise RuntimeError("inference artifact digest diverges from the package")
    if acceptance.get("state_dict_identical") is not True:
        raise RuntimeError("inference acceptance does not prove identical model state")
    corpus = acceptance.get("corpora")
    if not isinstance(corpus, list) or len(corpus) != 1:
        raise RuntimeError("inference acceptance corpus record is missing")
    corpus_record = corpus[0]
    if not isinstance(corpus_record, dict) or (
        corpus_record.get("rows") != 1500
        or corpus_record.get("identical_rows") != 1500
        or corpus_record.get("mismatch_count") != 0
    ):
        raise RuntimeError("inference acceptance is not a complete 1500-row match")

    project = tomllib.loads((root / PYPROJECT.relative_to(ROOT)).read_text())
    if f"v{project['project']['version']}" != DEFAULT_RELEASE:
        raise RuntimeError("package version diverges from the frozen release")
    if DEFAULT_MODEL_ID != "premove-ai/premove-itn":
        raise RuntimeError("default Hub repository diverges from the frozen release")
    if not re.fullmatch(r"[0-9a-f]{40}", DEFAULT_REVISION):
        raise RuntimeError("default Hub revision is not an immutable commit")
    artifact_doc = (root / ARTIFACT_DOC.relative_to(ROOT)).read_text()
    if DEFAULT_RELEASE not in artifact_doc or DEFAULT_REVISION not in artifact_doc:
        raise RuntimeError("inference artifact documentation is stale")


def main() -> None:
    verify_frozen_boundaries()
    print(f"verified frozen release boundaries for {DEFAULT_RELEASE}")


if __name__ == "__main__":
    main()
