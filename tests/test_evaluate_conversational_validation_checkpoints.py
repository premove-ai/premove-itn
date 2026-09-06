from scripts.evaluate_conversational_validation_checkpoints import summarize


def _row(kind, expected, sources, candidate_count=1):
    return {
        "text": "source",
        "expected_text": expected,
        "kind": kind,
        "evaluation_sources": sources,
        "candidate_count": candidate_count,
    }


def test_summarize_separates_active_keep_positive_kind_and_source() -> None:
    rows = [
        _row("KEEP", "source", ["sgd"], candidate_count=0),
        _row("KEEP", "source", ["sgd"], candidate_count=2),
        _row("TIME", "target", ["spokenwoz"]),
        _row("MULTI", "target", ["sgd", "spokenwoz"]),
    ]
    predictions = ["source", "wrong", "target", "target"]

    result = summarize(rows, predictions)

    assert result["overall"] == {"correct": 3, "total": 4, "accuracy": 0.75}
    assert result["positive"] == {"correct": 2, "total": 2, "accuracy": 1.0}
    assert result["keep"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert result["candidate_bearing_keep"] == {
        "correct": 0,
        "total": 1,
        "accuracy": 0.0,
    }
    assert result["multi"] == {"correct": 1, "total": 1, "accuracy": 1.0}
    assert result["per_source"]["sgd"]["correct"] == 2
    assert result["per_source"]["spokenwoz"]["correct"] == 2
