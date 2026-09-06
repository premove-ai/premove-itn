import json
from pathlib import Path

import pytest

import scripts.build_conversational_adaptation_schedule as schedule_builder


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row(text: str, expected: str, partition: str, kind: str) -> dict[str, object]:
    return {
        "text": text,
        "expected_text": expected,
        "partition": partition,
        "kind": kind,
        "kinds": [] if kind == "KEEP" else [kind],
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.open()]


def test_schedule_references_only_train_rows_and_interleaves_exact_ratio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(schedule_builder, "ROOT", tmp_path)
    conversational = tmp_path / "conversational.jsonl"
    provenance = tmp_path / "provenance.jsonl"
    google = tmp_path / "google.jsonl"
    output = tmp_path / "schedule"
    rows = [
        *(
            _row(
                f"active keep {index}",
                f"active keep {index}",
                "train",
                "KEEP",
            )
            for index in range(8)
        ),
        *(
            _row(f"positive {index}", str(index), "train", "CARDINAL")
            for index in range(4)
        ),
        _row("inactive keep", "inactive keep", "train", "KEEP"),
        _row("held out", "1", "validation", "CARDINAL"),
    ]
    _write_jsonl(conversational, rows)
    _write_jsonl(
        provenance,
        [
            {
                "record_index": index,
                "sources": [{"source": f"source-{index % 2}"}],
            }
            for index in range(len(rows))
        ],
    )
    _write_jsonl(
        google,
        [
            _row("google cardinal 1", "1", "train", "CARDINAL"),
            _row("google time 1", "01:00", "train", "TIME"),
            _row("google cardinal 2", "2", "train", "CARDINAL"),
            _row("google time 2", "02:00", "train", "TIME"),
            _row("google keep", "google keep", "train", "KEEP"),
            _row("google held out", "3", "validation", "CARDINAL"),
        ],
    )

    def profile(item: tuple[int, str]) -> tuple[int, int, tuple[str, ...]]:
        record_id, text = item
        count = int(text.startswith("active"))
        return record_id, count, ("CARDINAL",) if count else ()

    manifest = schedule_builder.build_schedule(
        conversational,
        provenance,
        google,
        output,
        candidate_keep_count=8,
        conversational_positive_count=4,
        google_replay_count=2,
        workers=1,
        profiler=profile,
    )
    schedule = _read_jsonl(output / "training_schedule.jsonl")

    assert [row["bucket"] for row in schedule] == list(
        schedule_builder.BUCKET_PATTERN * 2
    )
    assert len(schedule) == 14
    assert all(row["partition"] == "train" for row in schedule)
    assert all("text" not in row and "expected_text" not in row for row in schedule)
    conversational_positive_ids = {
        row["record_id"]
        for row in schedule
        if row["bucket"] == "conversational_positive"
    }
    assert conversational_positive_ids == {8, 9, 10, 11}
    assert {
        row["kind"] for row in schedule if row["bucket"] == "google_positive_replay"
    } == {"CARDINAL", "TIME"}
    assert manifest["bucket_distribution"] == {
        "candidate_bearing_keep": 8,
        "conversational_positive": 4,
        "google_positive_replay": 2,
    }
    assert manifest["selection"]["validation_included"] is False
    assert manifest["selection"]["test_included"] is False


def test_schedule_refuses_overwrite(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(schedule_builder, "ROOT", tmp_path)
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        schedule_builder.build_schedule(
            tmp_path / "missing-conversational.jsonl",
            tmp_path / "missing-provenance.jsonl",
            tmp_path / "missing-google.jsonl",
            output,
            candidate_keep_count=4,
            conversational_positive_count=2,
            google_replay_count=1,
            workers=1,
        )
