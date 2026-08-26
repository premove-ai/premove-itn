import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from premove_itn_training.dataset import load_frozen_split


def _positive() -> dict[str, object]:
    return {
        "provenance": {
            "primary": {
                "source": "google_tn",
                "source_file": "source.tsv",
                "sentence_number": 1,
            },
            "duplicates": [],
        },
        "record": {
            "text": "call four five",
            "expected_text": "call 45",
            "spans": [
                {
                    "kind": "PHONE",
                    "start": 5,
                    "end": 14,
                    "source": "four five",
                    "replacement": "45",
                }
            ],
            "tokens": [
                {"text": "call", "start": 0, "end": 4},
                {"text": "four", "start": 5, "end": 9},
                {"text": "five", "start": 10, "end": 14},
            ],
            "bio_labels": ["O", "B-PHONE", "I-PHONE"],
        },
    }


def _negative() -> dict[str, object]:
    return {
        "provenance": {
            "primary": {
                "source": "enrichment",
                "category": "context_only",
                "context": "purpose_built/context/1",
                "spans": [],
            },
            "duplicates": [],
        },
        "record": {
            "text": "hello world",
            "expected_text": "hello world",
            "spans": [],
            "tokens": [
                {"text": "hello", "start": 0, "end": 5},
                {"text": "world", "start": 6, "end": 11},
            ],
            "bio_labels": ["O", "O"],
        },
    }


def _statistics(records: list[dict[str, object]]) -> dict[str, object]:
    source_counts: Counter[str] = Counter()
    spans_by_kind: Counter[str] = Counter()
    for envelope in records:
        provenance = envelope["provenance"]
        record = envelope["record"]
        assert isinstance(provenance, dict)
        assert isinstance(record, dict)
        primary = provenance["primary"]
        spans = record["spans"]
        assert isinstance(primary, dict)
        assert isinstance(spans, list)
        source_counts[str(primary["source"])] += 1
        spans_by_kind.update(str(span["kind"]) for span in spans)
    return {
        "records": len(records),
        "groups": len(records),
        "source_counts": dict(sorted(source_counts.items())),
        "spans_by_kind": dict(sorted(spans_by_kind.items())),
        "context_only_records": sum(
            not record["record"]["spans"]  # type: ignore[index]
            for record in records
        ),
        "multi_span_records": sum(
            len(record["record"]["spans"]) > 1 for record in records  # type: ignore[index]
        ),
    }


def _write_partition(
    directory: Path, name: str, records: list[dict[str, object]]
) -> tuple[str, int]:
    content = "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        for record in records
    ).encode()
    digest = hashlib.sha256(content).hexdigest()
    (directory / f"{name}.jsonl").write_bytes(content)
    (directory / f"{name}.sha256").write_text(
        f"{digest}  {name}.jsonl\n", encoding="ascii"
    )
    return digest, len(content)


def _write_split(
    directory: Path,
    *,
    train: list[dict[str, object]] | None = None,
    validation: list[dict[str, object]] | None = None,
) -> Path:
    directory.mkdir()
    train = train or [_positive()]
    validation = validation or [_negative()]
    train_hash, train_bytes = _write_partition(directory, "train", train)
    validation_hash, validation_bytes = _write_partition(
        directory, "validation", validation
    )
    manifest = {
        "kind": "dataset_v1_split",
        "splitter": "grouped_dataset_v1_split_v1",
        "schema_version": 1,
        "group_leakage": 0,
        "exact_record_leakage": 0,
        "train": _statistics(train),
        "validation": _statistics(validation),
        "train_artifact": "train.jsonl",
        "train_artifact_sha256": train_hash,
        "train_artifact_bytes": train_bytes,
        "validation_artifact": "validation.jsonl",
        "validation_artifact_sha256": validation_hash,
        "validation_artifact_bytes": validation_bytes,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_loader_verifies_and_loads_the_frozen_split(tmp_path: Path) -> None:
    manifest_path = _write_split(tmp_path / "split")

    split = load_frozen_split(manifest_path)

    assert len(split.train) == len(split.validation) == 1
    assert split.train[0].tokens == ("call", "four", "five")
    assert not split.train[0].context_only
    assert split.validation[0].context_only
    assert split.train_sha256 == (
        manifest_path.parent / "train.sha256"
    ).read_text().split()[0]


def test_loader_rejects_artifact_hash_drift(tmp_path: Path) -> None:
    manifest_path = _write_split(tmp_path / "split")
    artifact = manifest_path.parent / "train.jsonl"
    artifact.write_bytes(artifact.read_bytes().replace(b"\n", b" \n", 1))

    with pytest.raises(ValueError, match="SHA-256 does not match manifest"):
        load_frozen_split(manifest_path)


def test_loader_rejects_labels_outside_model_v1(tmp_path: Path) -> None:
    record = _positive()
    payload = record["record"]
    assert isinstance(payload, dict)
    payload["bio_labels"] = ["O", "B-WORD", "I-WORD"]
    manifest_path = _write_split(tmp_path / "split", train=[record])

    with pytest.raises(ValueError, match="invalid training record") as error:
        load_frozen_split(manifest_path)
    assert isinstance(error.value.__cause__, ValueError)
    assert "not a Model V1 label" in str(error.value.__cause__)


def test_loader_rejects_reported_split_leakage(tmp_path: Path) -> None:
    manifest_path = _write_split(tmp_path / "split")
    manifest = json.loads(manifest_path.read_text())
    manifest["group_leakage"] = 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="reports group leakage"):
        load_frozen_split(manifest_path)
