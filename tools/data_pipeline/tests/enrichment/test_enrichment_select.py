import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from premove_itn_data.enrichment.reservoir import write_enrichment_reservoir
from premove_itn_data.enrichment.select import (
    ENRICHMENT_V1_SELECTION_QUOTAS,
    EnrichmentSelectionQuotas,
    write_enrichment_candidates,
)

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


def _write_google_candidates(path: Path) -> None:
    records = (
        _google_candidate(SpanKind.TIME, "eight o eight", "08:08", 1),
        _google_candidate(SpanKind.DIGIT_SEQUENCE, "eight o eight", "808", 2),
        _google_candidate(SpanKind.DIGIT_SEQUENCE, "nine one one", "911", 3),
    )
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
                        },
                        {
                            "name": "city",
                            "description": "City where the restaurant is located",
                            "is_categorical": False,
                        },
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
                        {
                            "speaker": "USER",
                            "utterance": "Find food in Paris.",
                            "frames": [
                                {
                                    "service": "Restaurants_1",
                                    "slots": [
                                        {
                                            "slot": "city",
                                            "start": 13,
                                            "exclusive_end": 18,
                                        }
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return train


def _write_reservoir(root: Path) -> tuple[Path, Path]:
    train = _write_sgd_train(root / "sgd")
    google = root / "google/candidates.jsonl"
    _write_google_candidates(google)
    output = root / "reservoir"
    write_enrichment_reservoir(
        train,
        google,
        output,
        seed="dataset-v1",
        realize=_rust.realize,
        phone_donor_count=1,
        electronic_donor_count=1,
    )
    return train, output / "manifest.json"


def _quotas() -> EnrichmentSelectionQuotas:
    return EnrichmentSelectionQuotas(
        sgd_span_targets={SpanKind.TIME: 1},
        sgd_context_only_records=1,
        purpose_phone_records=1,
        purpose_electronic_records=1,
        purpose_digit_sequence_records=1,
        time_digit_sequence_pairs=1,
        phone_digit_sequence_pairs=1,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_selects_balanced_records_and_preserves_collision_pairs(tmp_path: Path) -> None:
    train, reservoir_manifest = _write_reservoir(tmp_path / "input")
    output = tmp_path / "selected"

    manifest = write_enrichment_candidates(
        reservoir_manifest,
        train,
        output,
        seed="selection-v1",
        quotas=_quotas(),
    )

    payloads = _read_jsonl(output / "candidates.jsonl")
    reservoir_payloads = {
        payload["provenance"]["context"]: payload  # type: ignore[index]
        for payload in _read_jsonl(reservoir_manifest.parent / "reservoir.jsonl")
    }
    assert len(payloads) == manifest.output_records == 9
    assert manifest.available_span_free_sgd_context_records == 1
    assert manifest.excluded_ignored_span_sgd_context_records == 1
    assert manifest.selected_context_only_records == 1
    assert manifest.sgd_span_shortfall_by_kind == {}
    assert manifest.purpose_record_shortfalls == {}
    assert manifest.collision_pair_shortfalls == {}
    assert "Find food in Paris." not in {
        payload["record"]["text"] for payload in payloads  # type: ignore[index]
    }

    collision_payloads = [
        payload
        for payload in payloads
        if str(payload["provenance"]["category"]).startswith("collision_")  # type: ignore[index]
    ]
    assert len(collision_payloads) == 4
    collision_groups: dict[str, list[dict[str, object]]] = {}
    for payload in collision_payloads:
        context = payload["provenance"]["context"]  # type: ignore[index]
        group = str(context).split("/")[2]
        collision_groups.setdefault(group, []).append(payload)
    assert sorted(map(len, collision_groups.values())) == [2, 2]
    assert all("group" not in payload["provenance"] for payload in payloads)
    assert all(
        payload == reservoir_payloads[payload["provenance"]["context"]]  # type: ignore[index]
        for payload in payloads
    )
    canonical_records = {
        json.dumps(payload["record"], sort_keys=True) for payload in payloads
    }
    assert len(canonical_records) == len(payloads)

    artifact = (output / "candidates.jsonl").read_bytes()
    assert manifest.output_artifact_sha256 == hashlib.sha256(artifact).hexdigest()
    assert (output / "candidates.sha256").read_text() == (
        f"{manifest.output_artifact_sha256}  candidates.jsonl\n"
    )
    assert json.loads((output / "manifest.json").read_text()) == asdict(manifest)


def test_selection_is_deterministic_for_one_seed(tmp_path: Path) -> None:
    train, reservoir_manifest = _write_reservoir(tmp_path / "input")

    write_enrichment_candidates(
        reservoir_manifest,
        train,
        tmp_path / "first",
        seed="selection-v1",
        quotas=_quotas(),
    )
    write_enrichment_candidates(
        reservoir_manifest,
        train,
        tmp_path / "second",
        seed="selection-v1",
        quotas=_quotas(),
    )

    for name in ("candidates.jsonl", "candidates.sha256", "manifest.json"):
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "second" / name
        ).read_bytes()


def test_selection_rejects_changed_reservoir(tmp_path: Path) -> None:
    train, reservoir_manifest = _write_reservoir(tmp_path / "input")
    artifact = reservoir_manifest.parent / "reservoir.jsonl"
    lines = artifact.read_text().splitlines()
    lines[0] = json.dumps(json.loads(lines[0]))
    artifact.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="SHA-256"):
        write_enrichment_candidates(
            reservoir_manifest,
            train,
            tmp_path / "selected",
            seed="selection-v1",
            quotas=_quotas(),
        )

    assert not (tmp_path / "selected").exists()


def test_selection_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "selected"
    output.mkdir()
    candidates = output / "candidates.jsonl"
    candidates.write_text("existing\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_enrichment_candidates(
            tmp_path / "missing/manifest.json",
            tmp_path / "missing/train",
            output,
            seed="selection-v1",
            quotas=_quotas(),
        )

    assert candidates.read_text() == "existing\n"


def test_selection_quotas_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="purpose_phone_records"):
        EnrichmentSelectionQuotas(
            sgd_span_targets={},
            sgd_context_only_records=0,
            purpose_phone_records=-1,
            purpose_electronic_records=0,
            purpose_digit_sequence_records=0,
            time_digit_sequence_pairs=0,
            phone_digit_sequence_pairs=0,
        )


def test_v1_quotas_freeze_balanced_policy() -> None:
    quotas = ENRICHMENT_V1_SELECTION_QUOTAS

    assert quotas.sgd_span_targets == {
        SpanKind.DATE: 1_000,
        SpanKind.TIME: 1_500,
        SpanKind.CARDINAL: 500,
        SpanKind.MONEY: 300,
        SpanKind.DECIMAL: 250,
    }
    assert quotas.sgd_context_only_records == 2_500
    assert quotas.purpose_phone_records == 3_500
    assert quotas.purpose_electronic_records == 3_500
    assert quotas.purpose_digit_sequence_records == 1_000
    assert quotas.time_digit_sequence_pairs == 61
    assert quotas.phone_digit_sequence_pairs == 1_000
