"""Export the dev-selected Run-1 checkpoint as the v0.2.0 release candidate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from export_inference_artifact import dump_json, export, sha256_file  # noqa: E402

MANIFEST = ROOT / "data/models/time_identifier_targeted_run_1/manifest.json"
OUTPUT = ROOT / "artifacts/premove-itn-v0.2.0-rc1"
MODEL_CARD = ROOT / "docs/model-card-v0.2.0.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    checkpoint = ROOT / manifest["checkpoint"]
    actual_sha = sha256_file(checkpoint)
    if actual_sha != manifest["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint SHA-256 mismatch")

    compatibility_manifest = args.output.parent / ".v0.2.0-production.json"
    compatibility_manifest.write_text(
        json.dumps(
            {
                "checkpoint": manifest["checkpoint"],
                "checkpoint_sha256": actual_sha,
                "model_name": manifest["base_model"],
                "model_revision": manifest["base_model_revision"],
                "manifest": str(MANIFEST.relative_to(ROOT)),
            }
        )
    )
    try:
        export(checkpoint, compatibility_manifest, args.output, force=args.force)
    finally:
        compatibility_manifest.unlink(missing_ok=True)

    provenance_path = args.output / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance.update(
        {
            "artifact_version": "v0.2.0",
            "hub_repository": "premove-ai/premove-itn",
            "hub_revision": "v0.2.0",
            "package_version": "0.2.0",
            "release_status": "candidate",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "release_commit": None,
            "source_worktree_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
            ),
            "production_manifest": str(MANIFEST.relative_to(ROOT)),
            "training_lineage": {
                "starting_artifact_sha256": manifest["starting_artifact_sha256"],
                "training_data_sha256": manifest["training_data_sha256"],
                "training_rows": manifest["training"]["rows"],
                "selection_basis": manifest["selection_basis"],
                "selected_epoch": manifest["selected_epoch"],
                "frozen_tests_used_for_selection": False,
            },
            "training_config": manifest["training"],
            "benchmark_results": {
                "context_record_accuracy": 0.862,
                "context_pair_accuracy": 0.724,
                "identifier_accuracy": 0.98,
                "time_accuracy": 0.744,
                "dual_span_accuracy": 0.96,
                "dual_sentence_exact_accuracy": 0.92,
                "broad_strict_exact_accuracy": 0.43733333333333335,
            },
        }
    )
    dump_json(provenance_path, provenance)
    card = (
        MODEL_CARD.read_text()
        .replace("{{MODEL_SHA256}}", provenance["artifact_sha256"])
        .replace("{{CHECKPOINT_SHA256}}", actual_sha)
    )
    (args.output / "README.md").write_text(card)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
