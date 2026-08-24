import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from premove_itn_data import assemble as assemble_module
from premove_itn_data.assemble import assemble_dataset_v1
from premove_itn_data.records import TrainingSpan, compile_training_record

from premove_itn.labels import SpanKind


def _record(
    text: str,
    kind: SpanKind | None = None,
    replacement: str | None = None,
) -> dict[str, object]:
    spans: tuple[TrainingSpan, ...] = ()
    if kind is not None:
        assert replacement is not None
        spans = (TrainingSpan(kind, 0, len(text), text, replacement),)
    return asdict(compile_training_record(text, spans))


def _google_payload(
    record: dict[str, object],
    sentence_number: int,
) -> dict[str, object]:
    return {
        "record": record,
        "provenance": {
            "source": "google_tn",
            "source_file": "google/source.tsv",
            "sentence_number": sentence_number,
        },
    }


def _enrichment_payload(
    record: dict[str, object],
    context: str,
    *,
    category: str = "purpose_phone",
) -> dict[str, object]:
    spans = record["spans"]
    return {
        "record": record,
        "provenance": {
            "source": "enrichment",
            "category": category,
            "context": context,
            "spans": [
                {"context": context, "donor": f"donor/{context}"} for _ in spans
            ],
        },
    }


def _write_jsonl(path: Path, payloads: list[dict[str, object]]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
        for payload in payloads
    ).encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest(), len(content)


def _write_google(root: Path, payloads: list[dict[str, object]]) -> Path:
    artifact_hash, _ = _write_jsonl(root / "candidates.jsonl", payloads)
    (root / "candidates.sha256").write_text(f"{artifact_hash}\n")
    manifest = {
        "kind": "google_tn_selection_candidates",
        "selector": "deterministic_reservoir_v1",
        "schema_version": 1,
        "output_records": len(payloads),
        "cleanup": {
            "structural_validation": "PASS",
            "output_candidates_sha256": artifact_hash,
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def _write_enrichment(root: Path, payloads: list[dict[str, object]]) -> Path:
    artifact_hash, artifact_bytes = _write_jsonl(root / "candidates.jsonl", payloads)
    (root / "candidates.sha256").write_text(
        f"{artifact_hash}  candidates.jsonl\n"
    )
    manifest = {
        "kind": "enrichment_selection_candidates",
        "selector": "balanced_enrichment_v1",
        "schema_version": 1,
        "output_records": len(payloads),
        "output_artifact": "candidates.jsonl",
        "output_artifact_sha256": artifact_hash,
        "output_artifact_bytes": artifact_bytes,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def _inputs(root: Path, *, reverse: bool = False) -> tuple[Path, Path]:
    shared = _record("four thirty", SpanKind.TIME, "4:30")
    organic_context = _record("thanks")
    phone = _record("nine one one", SpanKind.PHONE, "911")
    google_payloads = [
        _google_payload(shared, 1),
        _google_payload(organic_context, 2),
    ]
    enrichment_payloads = [
        _enrichment_payload(
            shared,
            "purpose_built/collision/group/time/000000",
            category="collision_time_digit_sequence",
        ),
        _enrichment_payload(phone, "purpose_built/phone/one/000001"),
        _enrichment_payload(phone, "purpose_built/phone/two/000002"),
    ]
    if reverse:
        google_payloads.reverse()
        enrichment_payloads.reverse()
    return (
        _write_google(root / "google", google_payloads),
        _write_enrichment(root / "enrichment", enrichment_payloads),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_assembly_prefers_google_and_preserves_duplicate_origins(
    tmp_path: Path,
) -> None:
    google, enrichment = _inputs(tmp_path / "input")
    output = tmp_path / "dataset_v1"

    manifest = assemble_dataset_v1(google, enrichment, output)

    payloads = _read_jsonl(output / "candidates.jsonl")
    assert manifest.pre_dedup_records == 5
    assert manifest.output_records == len(payloads) == 3
    assert manifest.cross_source_duplicates == 1
    assert manifest.within_google_duplicates == 0
    assert manifest.within_enrichment_duplicates == 1
    assert manifest.google_retained_from_cross_source_duplicates == 1
    assert manifest.enrichment_discarded_from_cross_source_duplicates == 1
    assert manifest.records_with_duplicate_origins == 2
    assert manifest.source_counts == {"enrichment": 1, "google_tn": 2}
    assert manifest.spans_by_kind == {"PHONE": 1, "TIME": 1}
    assert manifest.context_only_records == 1
    assert manifest.multi_span_records == 0

    shared = next(
        payload
        for payload in payloads
        if payload["record"]["text"] == "four thirty"  # type: ignore[index]
    )
    provenance = shared["provenance"]
    assert provenance["primary"]["source"] == "google_tn"  # type: ignore[index]
    assert provenance["duplicates"][0]["source"] == "enrichment"  # type: ignore[index]
    assert provenance["duplicates"][0]["context"].startswith(  # type: ignore[index]
        "purpose_built/collision/"
    )

    artifact = (output / "candidates.jsonl").read_bytes()
    assert manifest.output_artifact_sha256 == hashlib.sha256(artifact).hexdigest()
    assert (output / "candidates.sha256").read_text() == (
        f"{manifest.output_artifact_sha256}  candidates.jsonl\n"
    )
    assert json.loads((output / "manifest.json").read_text()) == asdict(manifest)


def test_assembly_output_is_independent_of_input_order(tmp_path: Path) -> None:
    first_google, first_enrichment = _inputs(tmp_path / "first-input")
    second_google, second_enrichment = _inputs(
        tmp_path / "second-input", reverse=True
    )

    first = assemble_dataset_v1(
        first_google, first_enrichment, tmp_path / "first-output"
    )
    second = assemble_dataset_v1(
        second_google, second_enrichment, tmp_path / "second-output"
    )

    assert (tmp_path / "first-output/candidates.jsonl").read_bytes() == (
        tmp_path / "second-output/candidates.jsonl"
    ).read_bytes()
    assert first.output_artifact_sha256 == second.output_artifact_sha256
    assert first.source_counts == second.source_counts


def test_assembly_revalidates_compiled_records(tmp_path: Path) -> None:
    google, enrichment = _inputs(tmp_path / "input")
    artifact = enrichment.parent / "candidates.jsonl"
    payloads = _read_jsonl(artifact)
    payloads[0]["record"]["bio_labels"] = ["O", "O"]  # type: ignore[index]
    artifact_hash, artifact_bytes = _write_jsonl(artifact, payloads)
    manifest = json.loads(enrichment.read_text())
    manifest["output_artifact_sha256"] = artifact_hash
    manifest["output_artifact_bytes"] = artifact_bytes
    enrichment.write_text(json.dumps(manifest))
    (enrichment.parent / "candidates.sha256").write_text(
        f"{artifact_hash}  candidates.jsonl\n"
    )

    with pytest.raises(ValueError, match="invalid enrichment candidate"):
        assemble_dataset_v1(google, enrichment, tmp_path / "output")

    assert not (tmp_path / "output").exists()


def test_assembly_rejects_checksum_mismatch(tmp_path: Path) -> None:
    google, enrichment = _inputs(tmp_path / "input")
    (google.parent / "candidates.sha256").write_text("0" * 64 + "\n")

    with pytest.raises(ValueError, match="checksum does not match manifest"):
        assemble_dataset_v1(google, enrichment, tmp_path / "output")

    assert not (tmp_path / "output").exists()


def test_assembly_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    candidates = output / "candidates.jsonl"
    candidates.write_text("existing\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        assemble_dataset_v1(
            tmp_path / "missing/google.json",
            tmp_path / "missing/enrichment.json",
            output,
        )

    assert candidates.read_text() == "existing\n"


def test_assembly_write_failure_leaves_no_partial_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    google, enrichment = _inputs(tmp_path / "input")
    output = tmp_path / "output"

    def fail_payload(_candidate):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(assemble_module, "_output_payload", fail_payload)

    with pytest.raises(RuntimeError, match="serialization failed"):
        assemble_dataset_v1(google, enrichment, output)

    assert tuple(output.iterdir()) == ()
