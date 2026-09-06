from scripts.evaluate_full_google_milestones import evaluate_predictions


def test_numb3rs_semantic_metrics_accept_formatting_but_preserve_meaning() -> None:
    rows = [
        {
            "id": "money-format",
            "category": "MONEY",
            "text": "five million dollars",
            "original_text": "$5 million",
        },
        {
            "id": "money-wrong",
            "category": "MONEY",
            "text": "five canadian dollars",
            "original_text": "CAD 5",
        },
        {
            "id": "digits-format",
            "category": "TELEPHONE",
            "text": "three two nine two three two nine seven",
            "original_text": "3292-3297",
        },
    ]

    metrics = evaluate_predictions(
        rows,
        ["$5000000", "USD 5", "329-23297"],
        reference_field="original_text",
        category_field="category",
    )

    assert metrics["exact"] == 0
    assert metrics["semantic"] == {
        "correct": 2,
        "total": 3,
        "accuracy": 2 / 3,
        "per_category": {
            "MONEY": {"correct": 1, "total": 2, "accuracy": 0.5},
            "TELEPHONE": {"correct": 1, "total": 1, "accuracy": 1.0},
        },
    }
