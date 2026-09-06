import json

import pytest

from premove_itn import SpanKind
from scripts.evaluate_polynorm import DEFAULT_RUNS, load_rows, summarize


def test_default_runs_always_include_baseline_and_selected_checkpoint() -> None:
    assert [checkpoint.name for checkpoint, _ in DEFAULT_RUNS] == [
        "checkpoint.pt",
        "checkpoint.pt",
    ]
    assert [checkpoint.parent.name for checkpoint, _ in DEFAULT_RUNS] == [
        "google_selected_378k",
        "conversational_selected_20k",
    ]


def test_load_rows_reverses_polynorm_direction(tmp_path) -> None:
    source = tmp_path / "en-US_groundtruth.jsonl"
    source.write_text(
        json.dumps(
            {
                "index": "1",
                "category": "Time",
                "original_text": "Meet at 5:30.",
                "normalized_text": "Meet at five thirty.",
            }
        )
        + "\n"
    )

    assert load_rows(source) == [
        {
            "id": "1",
            "category": "Time",
            "text": "Meet at five thirty.",
            "original_text": "Meet at 5:30.",
        }
    ]


def test_summary_keeps_full_and_reachable_denominators_separate() -> None:
    rows = [
        {
            "id": "1",
            "category": "Time",
            "text": "five thirty",
            "original_text": "5:30",
        },
        {
            "id": "2",
            "category": "Time",
            "text": "six thirty",
            "original_text": "6.30",
        },
        {
            "id": "3",
            "category": "URL or Email",
            "text": "a at example dot com",
            "original_text": "a@example.com",
        },
    ]

    metrics = summarize(
        rows,
        ["5:30", "six thirty", "a at example dot com"],
        [True, False, True],
    )

    assert metrics["all"] == {"correct": 1, "total": 3, "accuracy": 1 / 3}
    assert metrics["reachable_only"] == {
        "correct": 1,
        "total": 2,
        "accuracy": 0.5,
    }
    assert metrics["oracle"]["reachable"] == 2
    assert metrics["error_categories"] == {
        "unreachable_target": 1,
        "missed_edit": 1,
    }


def test_summary_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        summarize([], ["extra"], [])


def test_summary_reports_kind_aware_semantic_accuracy() -> None:
    rows = [
        {
            "id": "1",
            "category": "Time",
            "text": "nine a m",
            "original_text": "9:00 AM",
        }
    ]
    metrics = summarize(
        rows,
        ["09:00 a.m."],
        [False],
        {"Time": SpanKind.TIME},
    )
    assert metrics["all"]["correct"] == 0
    assert metrics["semantic"] == {"correct": 1, "total": 1, "accuracy": 1.0}
