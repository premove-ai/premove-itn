import json
from pathlib import Path

from premove_itn import build_gold_graph, target_is_reachable


def test_every_golden_target_is_reachable() -> None:
    golden_path = Path(__file__).parents[1] / "data" / "golden.json"
    rows = json.loads(golden_path.read_text())

    unreachable = [
        row["id"]
        for row in rows
        if not target_is_reachable(row["text"], row["expected_text"])
    ]
    assert unreachable == []


def test_every_golden_target_has_a_gold_graph() -> None:
    golden_path = Path(__file__).parents[1] / "data" / "golden.json"
    rows = json.loads(golden_path.read_text())

    missing = [
        row["id"]
        for row in rows
        if build_gold_graph(row["text"], row["expected_text"]) is None
    ]
    assert missing == []
