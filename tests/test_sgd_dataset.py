import json
from pathlib import Path

from premove_itn import build_gold_graph
from scripts.audit_sgd_dataset import audit
from scripts.build_sgd_dataset import build_dataset


def _turn(
    utterance: str,
    *,
    value: str | None = None,
    canonical: str | None = None,
    slot: str = "value",
) -> dict[str, object]:
    slots = []
    actions = []
    if value is not None and canonical is not None:
        start = utterance.index(value)
        slots = [{"slot": slot, "start": start, "exclusive_end": start + len(value)}]
        actions = [
            {
                "act": "INFORM",
                "slot": slot,
                "values": [value],
                "canonical_values": [canonical],
            }
        ]
    return {
        "speaker": "USER",
        "utterance": utterance,
        "frames": [{"service": "Test_1", "slots": slots, "actions": actions}],
    }


def _write_source(root: Path, split: str, turns: list[dict[str, object]]) -> None:
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dialogues_001.json").write_text(
        json.dumps(
            [{"dialogue_id": f"{split}-1", "services": ["Test_1"], "turns": turns}]
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_build_sgd_dataset_canonicalizes_only_rust_verified_slot_spans(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sgd"
    _write_source(
        source,
        "train",
        [
            _turn(
                "Meet at four thirty.",
                value="four thirty",
                canonical="04:30",
                slot="time",
            ),
            _turn("Leave this sentence alone."),
            _turn(
                "Check in March 5th.",
                value="March 5th",
                canonical="2019-03-05",
                slot="date",
            ),
        ],
    )
    _write_source(source, "dev", [_turn("Validation sentence.")])
    _write_source(source, "test", [_turn("Test sentence.")])

    output = tmp_path / "output"
    manifest = build_dataset(source, output, max_per_kind=10)
    records = _read_jsonl(output / "dataset.jsonl")

    assert {record["partition"] for record in records} == {
        "train",
        "validation",
        "test",
    }
    time_record = next(record for record in records if record["kind"] == "TIME")
    assert time_record == {
        "text": "Meet at four thirty.",
        "expected_text": "Meet at 04:30.",
        "partition": "train",
        "kind": "TIME",
        "kinds": ["TIME"],
    }
    assert all(
        build_gold_graph(record["text"], record["expected_text"]) is not None
        for record in records
    )
    assert manifest["distribution"] == {"KEEP": 3, "TIME": 1}
    assert manifest["quarantined"] == 1
    assert manifest["quarantine_reasons"] == {"unsupported_canonicalization": 1}
    assert audit(output, sample_per_kind=10) == 0


def test_build_sgd_dataset_rejects_bad_span_alignment(tmp_path: Path) -> None:
    source = tmp_path / "sgd"
    bad = _turn("Meet at four thirty.", value="four thirty", canonical="04:30")
    bad["frames"][0]["slots"][0]["start"] = 0
    for split in ("train", "dev", "test"):
        _write_source(
            source, split, [bad] if split == "train" else [_turn(f"{split} keep")]
        )

    output = tmp_path / "output"
    manifest = build_dataset(source, output)

    assert manifest["quarantine_reasons"] == {"slot_action_mismatch": 1}


def test_build_sgd_dataset_deduplicates_across_official_splits(tmp_path: Path) -> None:
    source = tmp_path / "sgd"
    for split in ("train", "dev", "test"):
        _write_source(source, split, [_turn("Same utterance.")])

    output = tmp_path / "output"
    manifest = build_dataset(source, output)

    assert _read_jsonl(output / "dataset.jsonl") == [
        {
            "text": "Same utterance.",
            "expected_text": "Same utterance.",
            "partition": "train",
            "kind": "KEEP",
            "kinds": [],
        }
    ]
    assert manifest["duplicates"] == 2
