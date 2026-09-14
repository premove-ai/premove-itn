import json
from pathlib import Path

from scripts.generate_time_collision_inventory import (
    format_report,
    generate_inventory,
    number_to_words,
    write_inventory,
)


def test_number_to_words_covers_clock_values() -> None:
    assert [number_to_words(value) for value in (0, 5, 12, 20, 21, 35, 59)] == [
        "zero",
        "five",
        "twelve",
        "twenty",
        "twenty one",
        "thirty five",
        "fifty nine",
    ]


def test_inventory_has_complete_stable_clock_coverage() -> None:
    rows = generate_inventory()
    base_rows = [row for row in rows if row["variant"] == "base"]

    assert len(rows) == 1_920
    assert len(base_rows) == 1_440
    assert len({row["canonical_time"] for row in base_rows}) == 1_440
    assert all(row["time"] == row["canonical_time"] for row in rows)
    assert rows[0]["canonical_time"] == "00:00"
    assert rows[-1]["canonical_time"] == "23:59"


def test_inventory_records_each_realization_path_and_deduplicates_surfaces() -> None:
    rows = generate_inventory()
    by_spoken = {row["spoken"]: row for row in rows}

    assert by_spoken["two thirty"] == {
        "hour": 2,
        "minute": 30,
        "canonical_time": "02:30",
        "spoken": "two thirty",
        "variant": "base",
        "time": "02:30",
        "cardinal": "32",
        "cardinal_options": ["32", "230"],
        "digit_sequence": None,
        "non_time_outputs": [
            {"value": "32", "sources": ["CARDINAL"]},
            {"value": "230", "sources": ["CARDINAL_AVIATION"]},
        ],
        "collision_eligible": True,
    }
    assert by_spoken["one oh five"]["non_time_outputs"] == [
        {"value": "105", "sources": ["CARDINAL", "DIGIT_SEQUENCE"]}
    ]
    assert by_spoken["zero five"]["non_time_outputs"] == [
        {"value": "5", "sources": ["CARDINAL"]},
        {"value": "05", "sources": ["DIGIT_SEQUENCE"]},
    ]


def test_checked_in_inventory_matches_generator(tmp_path) -> None:
    rows = generate_inventory()
    generated = tmp_path / "inventory.jsonl"
    write_inventory(rows, generated)
    checked_in = (
        Path(__file__).resolve().parents[1]
        / "data/collisions/time_numeric_inventory.jsonl"
    )

    assert generated.read_bytes() == checked_in.read_bytes()
    assert (
        len([json.loads(line) for line in generated.read_text().splitlines()]) == 1_920
    )
    assert "Canonical times:                       1440" in format_report(rows)
