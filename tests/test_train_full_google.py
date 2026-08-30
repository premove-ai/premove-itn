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


def test_batch_source_prefetches_remaining_batches_in_order(monkeypatch) -> None:
    rows = (
        (0, {"text": "bbb", "expected_text": "3"}),
        (1, {"text": "a", "expected_text": "1"}),
        (2, {"text": "cc", "expected_text": "2"}),
    )
    monkeypatch.setattr(full_training, "eligible_rows", lambda: iter(rows))
    monkeypatch.setattr(full_training, "BATCH_SIZE", 1)
    monkeypatch.setattr(full_training, "BUCKET_WINDOW", 3)
    observed: dict[str, int] = {}

    def ordered(function, items, *, workers, max_pending, initializer):
        observed.update(workers=workers, max_pending=max_pending)
        return iter(items)

    monkeypatch.setattr(full_training, "ordered_process_prefetch", ordered)

    batches = list(full_training.batch_source(start_batch_offset=1))

    assert batches == [(('cc', '2'),), (('bbb', '3'),)]
    assert observed == {
        "workers": full_training.PREFETCH_WORKERS,
        "max_pending": full_training.PREFETCH_BATCHES,
    }
