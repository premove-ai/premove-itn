import json
from pathlib import Path

import pytest

import scripts.build_voice_agent_evaluation as builder


def test_main_dataset_repairs_audited_data_and_contract_defects() -> None:
    rows = builder.build_rows()
    report = builder.validate(rows)

    assert len(rows) == 1500
    assert report["semantic_signatures"] >= 1200
    assert report["collision_pairs"] == 175
    assert report["invalid_premove_span_kinds"] == 0
    assert report["category_to_premove_mapping_conflicts"] == 0
    assert report["percent_premove_mapping"] is None
    assert report["url_premove_mapping"] == "ELECTRONIC"
    assert report["implicit_currency_collision_rows"] == 0
    assert report["room_codes_labeled_order_id"] == 0
    assert report["international_phone_semantics_missing"] == 0
    assert report["identifier_semantic_comparison"] == "case_insensitive"
    assert report["date_collision_unique_payloads"] == 25


def test_main_dataset_freezes_evaluator_safety_and_reporting_scope(
    tmp_path: Path,
) -> None:
    output = tmp_path / "voice-agent-itn"
    manifest = builder.build(output)

    assert manifest["rows"] == 1500
    assert manifest["content_status"] == "frozen"
    assert manifest["artifact"]["sha256"] == builder.FROZEN_ARTIFACT_SHA256
    assert manifest["independent_audit"]["verdict"] == "no_dataset_blockers"
    assert manifest["gold_review_status"] == "pending_blind_human_adjudication"
    assert manifest["contamination_audit_status"] == "pending"
    assert manifest["permitted_inference_input_fields"] == ["text"]
    assert manifest["metric_hierarchy"]["voice_agent"]["row_filter"] == {
        "group": "voice_agent"
    }
    assert manifest["metric_hierarchy"]["voice_agent"]["excluded_groups"] == ["multi"]
    assert manifest["metric_hierarchy"]["keep"]["all_keep_preservation_accuracy"] == {
        "no_span": True,
        "rows": 100,
    }
    assert len(manifest["required_pre_run_gates"]) == 3
    assert json.loads((output / "manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.build(output)


def test_main_frozen_hash_detects_content_drift(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(builder, "FROZEN_ARTIFACT_SHA256", "wrong-hash")

    with pytest.raises(ValueError, match="frozen artifact drift"):
        builder.build(tmp_path / "drifted-main")
