import json
from pathlib import Path

from scripts.build_evaluation_history import build_history


def _write_metrics(path: Path, *, exposure: int, accuracy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "benchmark_role": "evaluation",
                "checkpoint_metadata": {
                    "checkpoint": f"milestones/checkpoint_{exposure:06d}.pt",
                    "checkpoint_exposure": exposure,
                },
                "records": 10,
                "trained": {
                    "accuracy": accuracy,
                    "semantic": {
                        "accuracy": 0.9,
                        "correct": 9,
                        "total": 10,
                    },
                    "per_kind": {"KEEP": {"accuracy": accuracy}},
                    "error_samples": ["omitted"],
                },
                "strict_exact": {
                    "accuracy": 0.5,
                    "exact": 5,
                    "total": 10,
                },
            }
        )
    )


def test_build_history_contains_compact_entries_and_excludes_error_samples(
    tmp_path: Path,
) -> None:
    _write_metrics(
        tmp_path / "data/generated/run/evaluations/golden_020000/metrics.json",
        exposure=20_000,
        accuracy=0.8,
    )
    _write_metrics(
        tmp_path
        / "data/generated/run/evaluations/google_validation_010000/metrics.json",
        exposure=10_000,
        accuracy=0.9,
    )

    history = build_history(tmp_path)

    assert [entry["source_file"] for entry in history["entries"]] == [
        "data/generated/run/evaluations/golden_020000/metrics.json",
        "data/generated/run/evaluations/google_validation_010000/metrics.json",
    ]
    assert history["entries"][0]["trained"]["accuracy"] == 0.8
    assert history["entries"][0]["trained"]["semantic"] == {
        "accuracy": 0.9,
        "correct": 9,
        "total": 10,
    }
    assert history["entries"][0]["strict_exact"] == {
        "accuracy": 0.5,
        "exact": 5,
        "total": 10,
    }
    assert history["entries"][0]["trained"]["per_kind"] == {"KEEP": {"accuracy": 0.8}}
    assert "error_samples" not in history["entries"][0]["trained"]
