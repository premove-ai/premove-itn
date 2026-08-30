import json
from pathlib import Path

from premove_itn import build_gold_graph
from scripts.audit_spokenwoz_dataset import audit
from scripts.build_spokenwoz_dataset import build_dataset


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _dialogue(*turns: dict[str, object]) -> dict[str, object]:
    return {"goal": {}, "log": list(turns)}


def _turn(
    text: str, spans: list[list[object]], *, tag: str = "user"
) -> dict[str, object]:
    return {
        "dialog_act": {},
        "metadata": {},
        "span_info": spans,
        "tag": tag,
        "text": text,
        "words": [],
    }


def test_build_spokenwoz_uses_official_splits_and_constructive_targets(
    tmp_path: Path,
) -> None:
    train_dev = tmp_path / "train_dev.json"
    validation_ids = tmp_path / "validation.json"
    test = tmp_path / "test.json"
    _write(
        train_dev,
        {
            "SNG0001": _dialogue(
                _turn("call 6 3 5 8 .", [["profile-inform", "phone", "6358", 1, 4]]),
                _turn("ignore 1 2 .", [], tag="system"),
            ),
            "SNG0002": _dialogue(_turn("keep this .", [])),
        },
    )
    _write(validation_ids, ["SNG0002"])
    _write(test, {"SNG9000": _dialogue(_turn("test context .", []))})

    output = tmp_path / "output"
    manifest = build_dataset(train_dev, validation_ids, test, output)
    rows = [json.loads(line) for line in (output / "dataset.jsonl").open()]

    assert {(row["text"], row["expected_text"], row["partition"]) for row in rows} == {
        ("call 6 3 5 8.", "call 6358.", "train"),
        ("keep this.", "keep this.", "validation"),
        ("test context.", "test context.", "test"),
    }
    assert all(build_gold_graph(row["text"], row["expected_text"]) for row in rows)
    assert manifest["distribution"] == {"KEEP": 2, "MULTI": 1}
    assert manifest["quarantine_reasons"] == {"non_user_turn": 1}
    assert audit(output) == 0


def test_build_spokenwoz_quarantines_bad_annotations_and_is_deterministic(
    tmp_path: Path,
) -> None:
    train_dev = tmp_path / "train_dev.json"
    validation_ids = tmp_path / "validation.json"
    test = tmp_path / "test.json"
    _write(
        train_dev,
        {
            "SNG0001": _dialogue(
                _turn("bad annotation .", [["profile-inform", "phone", "12", 50, 51]]),
                _turn("call 1 2 .", [["profile-inform", "phone", "12", 1, 2]]),
            )
        },
    )
    _write(validation_ids, [])
    _write(test, {})
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = build_dataset(train_dev, validation_ids, test, first)
    second_manifest = build_dataset(train_dev, validation_ids, test, second)

    assert first_manifest["quarantine_reasons"] == {"span_out_of_bounds": 1}
    assert (first / "dataset.jsonl").read_bytes() == (
        second / "dataset.jsonl"
    ).read_bytes()
    assert first_manifest["artifact_sha256"] == second_manifest["artifact_sha256"]
