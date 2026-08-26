import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from premove_itn_data.records import TrainingSpan, compile_training_record
from premove_itn_data.split import split_dataset_v1

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


def _google(text: str, source_file: str, sentence_number: int) -> dict[str, object]:
    return {
        "provenance": {
            "primary": {
                "source": "google_tn",
                "source_file": source_file,
                "sentence_number": sentence_number,
            },
            "duplicates": [],
        },
        "record": _record(text),
    }


def _enrichment(
    text: str,
    kind: SpanKind,
    replacement: str,
    *,
    category: str,
    context: str,
    donor: str,
) -> dict[str, object]:
    record = _record(text, kind, replacement)
    return {
        "provenance": {
            "primary": {
                "source": "enrichment",
                "category": category,
                "context": context,
                "spans": [{"context": context, "donor": donor}],
            },
            "duplicates": [],
        },
        "record": record,
    }


def _write_input(root: Path, payloads: list[dict[str, object]]) -> Path:
    root.mkdir(parents=True)
    content = "".join(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
        for payload in payloads
    ).encode()
    artifact_hash = hashlib.sha256(content).hexdigest()
    (root / "candidates.jsonl").write_bytes(content)
    (root / "candidates.sha256").write_text(f"{artifact_hash}  candidates.jsonl\n")
    manifest = {
        "kind": "dataset_v1_candidates",
        "assembler": "dataset_v1_assembly_v1",
        "schema_version": 1,
        "output_records": len(payloads),
        "output_artifact": "candidates.jsonl",
        "output_artifact_sha256": artifact_hash,
        "output_artifact_bytes": len(content),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def _payloads() -> list[dict[str, object]]:
    return [
        _google("hello", "source.tsv", 1),
        _google("thanks", "source.tsv", 1),
        _google("goodbye", "source.tsv", 2),
        _enrichment(
            "call nine one one",
            SpanKind.PHONE,
            "911",
            category="collision_phone_digit_sequence",
            context="purpose_built/collision/collision-a/phone/000001",
            donor="rust_collision/generated_phone/emergency/000000/phone",
        ),
        _enrichment(
            "code nine one one",
            SpanKind.DIGIT_SEQUENCE,
            "911",
            category="collision_phone_digit_sequence",
            context=("purpose_built/collision/collision-a/digit_sequence/000001"),
            donor=("rust_collision/generated_phone/emergency/000000/digit_sequence"),
        ),
        _enrichment(
            "one two three",
            SpanKind.CARDINAL,
            "123",
            category="sgd_positive",
            context="sgd/train/dialogues_001.json/dialogue-a/2",
            donor="google_tn/source.tsv/8/0",
        ),
        _enrichment(
            "four five six",
            SpanKind.CARDINAL,
            "456",
            category="sgd_positive",
            context="sgd/train/dialogues_001.json/dialogue-a/4",
            donor="google_tn/source.tsv/9/0",
        ),
        _enrichment(
            "six five zero",
            SpanKind.PHONE,
            "650",
            category="purpose_phone",
            context="purpose_built/phone/phone_call/000010",
            donor="generated_phone/local/000010",
        ),
        _enrichment(
            "six five zero please",
            SpanKind.PHONE,
            "650 please",
            category="purpose_phone",
            context="purpose_built/phone/phone_reach/000011",
            donor="generated_phone/local/000010",
        ),
    ]


def _read(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _texts(path: Path) -> set[str]:
    return {
        str(payload["record"]["text"])  # type: ignore[index]
        for payload in _read(path)
    }


def test_split_keeps_all_related_records_together(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path / "input", _payloads())
    output = tmp_path / "output"

    manifest = split_dataset_v1(
        manifest_path,
        output,
        seed="dataset-v1",
        validation_fraction=0.5,
    )

    train_texts = _texts(output / "train.jsonl")
    validation_texts = _texts(output / "validation.jsonl")
    for related in (
        {"hello", "thanks"},
        {"call nine one one", "code nine one one"},
        {"one two three", "four five six"},
        {"six five zero", "six five zero please"},
    ):
        assert related <= train_texts or related <= validation_texts

    assert train_texts.isdisjoint(validation_texts)
    assert train_texts | validation_texts == {
        str(payload["record"]["text"])  # type: ignore[index]
        for payload in _payloads()
    }
    assert manifest.group_leakage == 0
    assert manifest.exact_record_leakage == 0
    assert manifest.collision_groups_split == 0
    assert manifest.donor_value_groups_split == 0
    assert manifest.train.records + manifest.validation.records == 9
    assert manifest.train.source_counts or manifest.validation.source_counts
    assert json.loads((output / "manifest.json").read_text()) == asdict(manifest)


def test_split_is_independent_of_input_order(tmp_path: Path) -> None:
    payloads = _payloads()
    first = _write_input(tmp_path / "first-input", payloads)
    second = _write_input(tmp_path / "second-input", list(reversed(payloads)))

    first_manifest = split_dataset_v1(
        first, tmp_path / "first-output", seed="stable", validation_fraction=0.4
    )
    second_manifest = split_dataset_v1(
        second,
        tmp_path / "second-output",
        seed="stable",
        validation_fraction=0.4,
    )

    assert (tmp_path / "first-output/train.jsonl").read_bytes() == (
        tmp_path / "second-output/train.jsonl"
    ).read_bytes()
    assert (tmp_path / "first-output/validation.jsonl").read_bytes() == (
        tmp_path / "second-output/validation.jsonl"
    ).read_bytes()
    assert first_manifest.train_artifact_sha256 == (
        second_manifest.train_artifact_sha256
    )
    assert first_manifest.validation_artifact_sha256 == (
        second_manifest.validation_artifact_sha256
    )


def test_split_revalidates_input_and_refuses_overwrite(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path / "input", _payloads())
    artifact = manifest_path.parent / "candidates.jsonl"
    lines = artifact.read_bytes().splitlines(keepends=True)
    artifact.write_bytes(b"".join(reversed(lines)))

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        split_dataset_v1(manifest_path, tmp_path / "bad-output", seed="stable")

    clean_manifest = _write_input(tmp_path / "clean-input", _payloads())
    output = tmp_path / "existing-output"
    output.mkdir()
    (output / "train.jsonl").write_text("existing\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        split_dataset_v1(clean_manifest, output, seed="stable")
    assert (output / "train.jsonl").read_text() == "existing\n"


def test_split_validates_configuration(tmp_path: Path) -> None:
    manifest_path = _write_input(tmp_path / "input", _payloads())
    with pytest.raises(ValueError, match="seed must be non-empty"):
        split_dataset_v1(manifest_path, tmp_path / "empty-seed", seed="")
    with pytest.raises(ValueError, match="between zero and one"):
        split_dataset_v1(
            manifest_path,
            tmp_path / "invalid-fraction",
            seed="stable",
            validation_fraction=1.0,
        )
