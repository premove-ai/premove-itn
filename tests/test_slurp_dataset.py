import json
from pathlib import Path

from premove_itn import build_gold_graph
from scripts.audit_slurp_dataset import audit
from scripts.build_slurp_dataset import build_dataset


def _record(
    slurp_id: int,
    sentence: str,
    surfaces: list[str],
    entities: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "slurp_id": slurp_id,
        "sentence": sentence,
        "tokens": [
            {"id": index, "surface": surface}
            for index, surface in enumerate(surfaces)
        ],
        "entities": entities,
        "scenario": "alarm",
        "intent": "alarm_set",
    }


def _write_source(root: Path, records: dict[str, list[dict[str, object]]]) -> None:
    root.mkdir()
    for split in ("train", "devel", "test"):
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records.get(split, [])),
            encoding="utf-8",
        )


def test_build_slurp_uses_only_aligned_semantic_entities(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(
        source,
        {
            "train": [
                _record(
                    1,
                    "wake me at eight pm",
                    ["wake", "me", "at", "eight", "pm"],
                    [{"span": [3, 4], "type": "time"}],
                ),
                _record(
                    2,
                    "play movie three",
                    ["play", "movie", "three"],
                    [{"span": [1, 2], "type": "movie_name"}],
                ),
                _record(
                    4,
                    "timer for ten minutes",
                    ["timer", "for", "ten", "minutes"],
                    [{"span": [2, 3], "type": "time"}],
                ),
                _record(
                    5,
                    "between eight and nine",
                    ["between", "eight", "and", "nine"],
                    [{"span": [1, 2, 3], "type": "time"}],
                ),
            ],
            "devel": [
                _record(
                    3,
                    "on april two",
                    ["on", "April", "two"],
                    [{"span": [1, 2], "type": "date"}],
                )
            ],
        },
    )
    output = tmp_path / "output"
    manifest = build_dataset(source, output)
    rows = [json.loads(line) for line in (output / "dataset.jsonl").open()]

    assert [(row["text"], row["expected_text"]) for row in rows] == [
        ("wake me at eight pm", "wake me at 08:00 p.m."),
        ("play movie three", "play movie three"),
        ("timer for ten minutes", "timer for 10 min"),
        ("between eight and nine", "between eight and nine"),
        ("on april two", "on april 2"),
    ]
    assert [row["partition"] for row in rows] == [
        "train",
        "train",
        "train",
        "train",
        "validation",
    ]
    assert all(
        build_gold_graph(row["text"], row["expected_text"]) is not None
        for row in rows
    )
    assert manifest["records"] == 5
    assert manifest["synthetic_excluded"] is True
    assert audit(output) == 0


def test_build_slurp_quarantines_bad_alignment_and_duplicates(tmp_path: Path) -> None:
    duplicate = _record(1, "keep this", ["keep", "this"], [])
    source = tmp_path / "source"
    _write_source(
        source,
        {
            "train": [duplicate],
            "devel": [duplicate],
            "test": [_record(2, "spoken text", ["different"], [])],
        },
    )
    output = tmp_path / "output"
    manifest = build_dataset(source, output)

    assert manifest["records"] == 1
    assert manifest["quarantined"] == 2
    assert manifest["quarantine_reasons"] == {
        "duplicate": 1,
        "token_alignment": 1,
    }
    [provenance] = [
        json.loads(line) for line in (output / "provenance.jsonl").open()
    ]
    assert provenance["slurp_id"] == 1
