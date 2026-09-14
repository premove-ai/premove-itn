"""Generate the deterministic TIME versus numeric realization inventory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from premove_itn import _rust

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "collisions" / "time_numeric_inventory.jsonl"

SMALL_NUMBERS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}
SOURCE_ORDER = ("CARDINAL", "CARDINAL_AVIATION", "DIGIT_SEQUENCE")
VARIANT_RANK = {"base": 0, "oh": 1, "zero": 2, "zero_zero": 3, "oh_oh": 4}


def number_to_words(value: int) -> str:
    """Spell one integer from 0 through 59 with a fixed vocabulary."""
    if not 0 <= value <= 59:
        raise ValueError(f"number is outside 0..59: {value}")
    if value < 20:
        return SMALL_NUMBERS[value]
    tens, ones = divmod(value, 10)
    tens_word = TENS[tens * 10]
    return tens_word if ones == 0 else f"{tens_word} {SMALL_NUMBERS[ones]}"


def spoken_variants(hour: int, minute: int) -> tuple[tuple[str, str], ...]:
    """Return the mandatory base form and ordered optional aliases."""
    hour_words = number_to_words(hour)
    if minute == 0:
        return (
            ("base", f"{hour_words} hundred"),
            ("zero_zero", f"{hour_words} zero zero"),
            ("oh_oh", f"{hour_words} oh oh"),
        )
    minute_words = number_to_words(minute)
    base = ("base", f"{hour_words} {minute_words}")
    if minute >= 10:
        return (base,)
    return (
        base,
        ("oh", f"{hour_words} oh {minute_words}"),
        ("zero", f"{hour_words} zero {minute_words}"),
    )


def build_row(hour: int, minute: int, variant: str, spoken: str) -> dict[str, Any]:
    """Run one spoken form through every relevant Rust realization path."""
    canonical_time = f"{hour:02}:{minute:02}"
    time_value = _rust.realize("TIME", spoken)
    if variant == "base" and time_value != canonical_time:
        raise RuntimeError(
            f"base TIME mismatch for {spoken!r}: {time_value!r} != {canonical_time!r}"
        )

    cardinal = _rust.realize("CARDINAL", spoken)
    cardinal_options = _rust.realize_options("CARDINAL", spoken)
    digit_sequence = _rust.realize("DIGIT_SEQUENCE", spoken)

    outputs: dict[str, list[str]] = {}
    for value, source in (
        *(
            (value, "CARDINAL" if value == cardinal else "CARDINAL_AVIATION")
            for value in cardinal_options
        ),
        (digit_sequence, "DIGIT_SEQUENCE"),
    ):
        if value is None or value == time_value:
            continue
        sources = outputs.setdefault(value, [])
        if source not in sources:
            sources.append(source)

    non_time_outputs = [
        {
            "value": value,
            "sources": sorted(sources, key=SOURCE_ORDER.index),
        }
        for value, sources in outputs.items()
    ]
    return {
        "hour": hour,
        "minute": minute,
        "canonical_time": canonical_time,
        "spoken": spoken,
        "variant": variant,
        "time": time_value,
        "cardinal": cardinal,
        "cardinal_options": cardinal_options,
        "digit_sequence": digit_sequence,
        "non_time_outputs": non_time_outputs,
        "collision_eligible": bool(non_time_outputs),
    }


def generate_inventory() -> list[dict[str, Any]]:
    """Build the complete stable inventory and enforce its core invariants."""
    rows = [
        build_row(hour, minute, variant, spoken)
        for hour in range(24)
        for minute in range(60)
        for variant, spoken in spoken_variants(hour, minute)
    ]
    rows.sort(
        key=lambda row: (row["hour"], row["minute"], VARIANT_RANK[row["variant"]])
    )

    canonical_times = {row["canonical_time"] for row in rows}
    base_rows = [row for row in rows if row["variant"] == "base"]
    row_keys = {(row["canonical_time"], row["spoken"]) for row in rows}
    if len(canonical_times) != 1_440:
        raise RuntimeError(
            f"expected 1,440 canonical times, got {len(canonical_times)}"
        )
    if len(base_rows) != 1_440:
        raise RuntimeError(f"expected 1,440 base rows, got {len(base_rows)}")
    if len(row_keys) != len(rows):
        raise RuntimeError("duplicate canonical_time and spoken pair")
    if any(
        output["value"] == row["time"]
        for row in rows
        for output in row["non_time_outputs"]
    ):
        raise RuntimeError("non-time output duplicates its TIME output")
    return rows


def write_inventory(rows: list[dict[str, Any]], output: Path) -> None:
    """Write compact, deterministic JSON Lines."""
    output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        f"{json.dumps(row, ensure_ascii=True, separators=(',', ':'))}\n" for row in rows
    )
    output.write_text(content, encoding="utf-8")


def format_report(rows: list[dict[str, Any]]) -> str:
    """Summarize the realized collision inventory."""
    canonical_count = len({row["canonical_time"] for row in rows})
    base_count = sum(row["variant"] == "base" for row in rows)
    time_count = sum(row["time"] is not None for row in rows)
    rejected_alias_count = sum(
        row["variant"] != "base" and row["time"] is None for row in rows
    )
    cardinal_count = sum(row["cardinal"] is not None for row in rows)
    digit_sequence_count = sum(row["digit_sequence"] is not None for row in rows)
    collision_count = sum(row["collision_eligible"] for row in rows)
    source_rows = Counter(
        source
        for row in rows
        for output in row["non_time_outputs"]
        for source in output["sources"]
    )
    output_counts = Counter(len(row["non_time_outputs"]) for row in rows)
    aviation_alternates = sum(
        option != row["cardinal"] for row in rows for option in row["cardinal_options"]
    )
    three_plus_count = sum(count for size, count in output_counts.items() if size >= 3)
    return "\n".join(
        (
            f"Canonical times:                       {canonical_count}",
            f"Mandatory base forms:                  {base_count}",
            f"Optional aliases generated:            {len(rows) - base_count}",
            f"Total spoken forms attempted:           {len(rows)}",
            "",
            f"TIME accepted:                         {time_count}",
            f"TIME rejected aliases:                 {rejected_alias_count}",
            "",
            f"CARDINAL accepted:                     {cardinal_count}",
            f"CARDINAL aviation alternatives:        {aviation_alternates}",
            f"DIGIT_SEQUENCE accepted:               {digit_sequence_count}",
            "",
            f"Rows with >=1 non-time collision:      {collision_count}",
            f"Rows with CARDINAL provenance:         {source_rows['CARDINAL']}",
            "Rows with aviation provenance:         "
            f"{source_rows['CARDINAL_AVIATION']}",
            f"Rows with digit sequence provenance:   {source_rows['DIGIT_SEQUENCE']}",
            "",
            f"Rows with 1 distinct non-time output:  {output_counts[1]}",
            f"Rows with 2 distinct non-time outputs: {output_counts[2]}",
            f"Rows with 3+ distinct outputs:         {three_plus_count}",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    rows = generate_inventory()
    write_inventory(rows, arguments.output)
    print(format_report(rows))


if __name__ == "__main__":
    main()
