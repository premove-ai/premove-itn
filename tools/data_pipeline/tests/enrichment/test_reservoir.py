import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from premove_itn_data.enrichment import reservoir as reservoir_module
from premove_itn_data.enrichment.reservoir import write_enrichment_reservoir

from premove_itn import _rust
from premove_itn.labels import SpanKind


def _google_candidate(
    kind: SpanKind,
    spoken: str,
    replacement: str,
    sentence_number: int,
) -> dict[str, object]:
    return {
        "record": {
            "text": spoken,
            "spans": [
                {
                    "kind": kind.value,
                    "start": 0,
                    "end": len(spoken),
                    "source": spoken,
                    "replacement": replacement,
                }
            ],
        },
        "provenance": {
            "source": "google_tn",
            "source_file": "google/source.tsv",
            "sentence_number": sentence_number,
        },
    }


def _write_google_candidates(
    path: Path,
    *,
    reverse: bool = False,
) -> None:
    records = [
        _google_candidate(SpanKind.TIME, "eight o eight", "08:08", 1),
        _google_candidate(SpanKind.DIGIT_SEQUENCE, "eight o eight", "808", 2),
        _google_candidate(SpanKind.DIGIT_SEQUENCE, "nine one one", "911", 3),
    ]
    if reverse:
        records.reverse()
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def _write_sgd_train(root: Path) -> Path:
    train = root / "train"
    train.mkdir(parents=True)
    (train / "schema.json").write_text(
        json.dumps(
            [
                {
                    "service_name": "Restaurants_1",
                    "slots": [
                        {
                            "name": "time",
                            "description": (
                                "Time for the reservation or to find availability"
                            ),
                            "is_categorical": False,
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    (train / "dialogues_001.json").write_text(
        json.dumps(
            [
                {
                    "dialogue_id": "1_00000",
                    "turns": [
                        {
                            "speaker": "USER",
                            "utterance": "Book it at 7:30 pm.",
                            "frames": [
                                {
                                    "service": "Restaurants_1",
                                    "slots": [
                                        {
                                            "slot": "time",
                                            "start": 11,
                                            "exclusive_end": 18,
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "speaker": "USER",
                            "utterance": "Thanks for your help.",
                            "frames": [
                                {"service": "Restaurants_1", "slots": []}
                            ],
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return train


def _write_inputs(root: Path, *, reverse_google: bool = False) -> tuple[Path, Path]:
    train = _write_sgd_train(root / "sgd")
    google = root / "google/candidates.jsonl"
    _write_google_candidates(google, reverse=reverse_google)
    return train, google


def _write_reservoir(train: Path, google: Path, output: Path):
    return write_enrichment_reservoir(
        train,
        google,
        output,
        seed="dataset-v1",
        realize=_rust.realize,
        phone_donor_count=1,
        electronic_donor_count=1,
    )


def test_writer_preserves_lineage_counts_and_checksum(tmp_path: Path) -> None:
    train, google = _write_inputs(tmp_path / "input")
    output = tmp_path / "output"

    manifest = _write_reservoir(train, google, output)

    artifact = (output / "reservoir.jsonl").read_bytes()
    payloads = [json.loads(line) for line in artifact.splitlines()]
    assert len(payloads) == manifest.output_records
    assert manifest.google_donor_count == 3
    assert manifest.records_by_category["sgd_positive"] == 1
    assert manifest.records_by_category["sgd_context_only"] == 1
    assert manifest.records_by_category["purpose_phone"] == 1
    assert manifest.records_by_category["purpose_electronic"] == 1
    assert manifest.records_by_category["purpose_digit_sequence"] == 2
    assert manifest.records_by_category["collision_time_digit_sequence"] > 0
    assert manifest.records_by_category["collision_phone_digit_sequence"] > 0
    assert manifest.context_only_records == 1
    assert sum(manifest.records_by_category.values()) == manifest.output_records
    assert manifest.artifact_bytes == len(artifact)
    assert manifest.artifact_sha256 == hashlib.sha256(artifact).hexdigest()
    assert (output / "reservoir.sha256").read_text() == (
        f"{manifest.artifact_sha256}  reservoir.jsonl\n"
    )
    assert json.loads((output / "manifest.json").read_text()) == asdict(manifest)

    for payload in payloads:
        provenance = payload["provenance"]
        assert provenance["source"] == "enrichment"
        assert len(provenance["spans"]) == len(payload["record"]["spans"])
    sgd_positive = next(
        payload
        for payload in payloads
        if payload["provenance"]["category"] == "sgd_positive"
    )
    assert sgd_positive["provenance"]["context"].startswith("sgd/train/")
    assert sgd_positive["provenance"]["spans"][0]["context"].startswith(
        "sgd_slot/Restaurants_1/time/"
    )
    assert sgd_positive["provenance"]["spans"][0]["donor"].startswith(
        "google_tn/"
    )


def test_writer_is_independent_of_google_input_order(tmp_path: Path) -> None:
    train = _write_sgd_train(tmp_path / "sgd")
    first_google = tmp_path / "first/candidates.jsonl"
    second_google = tmp_path / "second/candidates.jsonl"
    _write_google_candidates(first_google)
    _write_google_candidates(second_google, reverse=True)

    _write_reservoir(train, first_google, tmp_path / "first-output")
    _write_reservoir(train, second_google, tmp_path / "second-output")

    for name in ("reservoir.jsonl", "reservoir.sha256", "manifest.json"):
        assert (tmp_path / "first-output" / name).read_bytes() == (
            tmp_path / "second-output" / name
        ).read_bytes()


def test_writer_refuses_to_overwrite_existing_artifact(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    reservoir = output / "reservoir.jsonl"
    reservoir.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_enrichment_reservoir(
            tmp_path / "missing/train",
            tmp_path / "missing/candidates.jsonl",
            output,
            seed="dataset-v1",
            realize=_rust.realize,
        )

    assert reservoir.read_text() == "existing\n"
    assert not (output / "manifest.json").exists()
    assert not (output / "reservoir.sha256").exists()


def test_writer_failure_leaves_no_completed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, google = _write_inputs(tmp_path / "input")
    output = tmp_path / "output"

    def fail_compile(_candidate):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(
        reservoir_module,
        "compile_enrichment_candidate",
        fail_compile,
    )

    with pytest.raises(RuntimeError, match="compile failed"):
        _write_reservoir(train, google, output)

    assert not (output / "reservoir.jsonl").exists()
    assert not (output / "manifest.json").exists()
    assert not (output / "reservoir.sha256").exists()
    assert tuple(output.iterdir()) == ()


@pytest.mark.parametrize(
    ("seed", "phone_count", "electronic_count", "match"),
    [
        ("", 1, 1, "seed must be non-empty"),
        ("seed", -1, 1, "phone_donor_count must be non-negative"),
        ("seed", 1, -1, "electronic_donor_count must be non-negative"),
    ],
)
def test_writer_rejects_invalid_configuration(
    tmp_path: Path,
    seed: str,
    phone_count: int,
    electronic_count: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        write_enrichment_reservoir(
            tmp_path / "train",
            tmp_path / "candidates.jsonl",
            tmp_path / "output",
            seed=seed,
            realize=_rust.realize,
            phone_donor_count=phone_count,
            electronic_donor_count=electronic_count,
        )

    assert not (tmp_path / "output").exists()
