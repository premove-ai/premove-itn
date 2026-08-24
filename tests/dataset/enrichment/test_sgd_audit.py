import json
from dataclasses import asdict
from pathlib import Path

import pytest

from premove_itn.dataset.enrichment.sgd_audit import (
    SgdContextAudit,
    audit_sgd_train,
    write_sgd_context_audit,
)


def _write_train(root: Path) -> Path:
    train = root / "train"
    train.mkdir()
    (train / "schema.json").write_text(
        json.dumps(
            [
                {
                    "service_name": "Restaurants_1",
                    "slots": [
                        {
                            "name": "time",
                            "description": "Time of the restaurant reservation",
                            "is_categorical": False,
                        },
                        {
                            "name": "party_size",
                            "description": "Number of people in the party",
                            "is_categorical": False,
                        },
                    ],
                }
            ]
        )
    )
    dialogues = [
        {
            "dialogue_id": "1_00000",
            "turns": [
                {
                    "speaker": "USER",
                    "utterance": "Book at 7:30 for two.",
                    "frames": [
                        {
                            "service": "Restaurants_1",
                            "slots": [
                                {"slot": "time", "start": 8, "exclusive_end": 12},
                                {
                                    "slot": "party_size",
                                    "start": 17,
                                    "exclusive_end": 20,
                                },
                            ],
                        }
                    ],
                },
                {
                    "speaker": "USER",
                    "utterance": "Yes please.",
                    "frames": [{"service": "Restaurants_1", "slots": []}],
                },
            ],
        },
        {
            "dialogue_id": "1_00001",
            "turns": [
                {
                    "speaker": "USER",
                    "utterance": "Use 8:15 inside 8:15 pm.",
                    "frames": [
                        {
                            "service": "Restaurants_1",
                            "slots": [
                                {"slot": "time", "start": 4, "exclusive_end": 8},
                                {"slot": "time", "start": 16, "exclusive_end": 23},
                                {"slot": "time", "start": 16, "exclusive_end": 20},
                            ],
                        }
                    ],
                }
            ],
        },
    ]
    (train / "dialogues_001.json").write_text(json.dumps(dialogues))
    return train


def test_audits_counts_order_and_bounded_examples(tmp_path: Path) -> None:
    train = _write_train(tmp_path)

    report = audit_sgd_train(train, examples_per_slot=2)

    assert report.kind == "sgd_context_audit"
    assert report.schema_version == 1
    assert report.source_split == "train"
    assert report.dialogue_files == 1
    assert report.dialogues == 2
    assert report.user_turns == 3
    assert report.context_only_user_turns == 1
    assert report.non_categorical_slot_spans == 5
    assert report.overlapping_span_turns == 1
    assert [
        (
            slot.service,
            slot.slot,
            slot.description,
            slot.count,
            len(slot.examples),
        )
        for slot in report.service_slot_pairs
    ] == [
        (
            "Restaurants_1",
            "time",
            "Time of the restaurant reservation",
            4,
            2,
        ),
        (
            "Restaurants_1",
            "party_size",
            "Number of people in the party",
            1,
            1,
        ),
    ]
    assert all(
        example.source_file == "train/dialogues_001.json"
        for slot in report.service_slot_pairs
        for example in slot.examples
    )


def test_sampling_is_deterministic(tmp_path: Path) -> None:
    train = _write_train(tmp_path)

    first = audit_sgd_train(train, examples_per_slot=1)
    second = audit_sgd_train(train, examples_per_slot=1)

    assert first == second


def test_can_omit_examples(tmp_path: Path) -> None:
    report = audit_sgd_train(_write_train(tmp_path), examples_per_slot=0)

    assert all(not slot.examples for slot in report.service_slot_pairs)


def test_rejects_negative_sample_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        audit_sgd_train(_write_train(tmp_path), examples_per_slot=-1)


def test_writes_report_without_overwrite(tmp_path: Path) -> None:
    train = _write_train(tmp_path)
    output = tmp_path / "generated" / "audit.json"

    report = write_sgd_context_audit(train, output, examples_per_slot=2)

    assert json.loads(output.read_text()) == json.loads(json.dumps(asdict(report)))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_sgd_context_audit(train, output, examples_per_slot=2)


def test_report_contract_is_immutable(tmp_path: Path) -> None:
    report = audit_sgd_train(_write_train(tmp_path))

    assert isinstance(report, SgdContextAudit)
    with pytest.raises(AttributeError):
        report.user_turns = 0  # type: ignore[misc]
