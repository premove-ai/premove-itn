import json
from pathlib import Path

import scripts.train_full_google as full_training


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_scan_excludes_seen_validation_and_long_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_rows(
        dataset,
        [
            {"partition": "train", "text": "seen", "kind": "KEEP"},
            {"partition": "validation", "text": "held out", "kind": "DATE"},
            {"partition": "train", "text": "new", "kind": "DATE"},
            {"partition": "train", "text": "too long", "kind": "TIME"},
        ],
    )
    monkeypatch.setattr(full_training, "DATASET", dataset)
    monkeypatch.setattr(full_training, "FIRST_UNSEEN_RECORD_ID", 2)
    monkeypatch.setattr(full_training, "MAX_SOURCE_CHARS", 7)

    count, remaining, cumulative, first_id, last_id = full_training.scan_dataset()

    assert count == 1
    assert remaining == {"DATE": 1}
    assert cumulative == {"DATE": 1, "KEEP": 1}
    assert (first_id, last_id) == (2, 2)
    assert list(full_training.eligible_rows()) == [
        (2, {"partition": "train", "text": "new", "kind": "DATE"})
    ]


def test_window_batches_are_stable_and_length_bucketed(monkeypatch) -> None:
    monkeypatch.setattr(full_training, "BATCH_SIZE", 2)
    rows = [
        (3, {"text": "bbb"}),
        (2, {"text": "a"}),
        (1, {"text": "c"}),
    ]

    batches = list(full_training.window_batches(rows))

    assert [[record_id for record_id, _ in batch] for batch in batches] == [
        [1, 2],
        [3],
    ]
