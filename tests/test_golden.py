import hashlib
import json
from pathlib import Path

from premove_itn import NormalizationResult, NormalizedEdit, SpanKind, TextSpan
from premove_itn.labels import SPAN_KINDS
from premove_itn.tokenize import tokenize

DATA_DIR = Path(__file__).parents[1] / "data"
GOLDEN_PATH = DATA_DIR / "golden.json"
GOLDEN_CHECKSUM_PATH = DATA_DIR / "golden.sha256"


def _load(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text())


def test_golden_file_matches_frozen_checksum() -> None:
    assert hashlib.sha256(GOLDEN_PATH.read_bytes()).hexdigest() == (
        GOLDEN_CHECKSUM_PATH.read_text().strip()
    )


def _validate_records(records: list[dict[str, object]]) -> None:
    assert len({record["id"] for record in records}) == len(records)
    assert len({record["text"] for record in records}) == len(records)

    for record in records:
        text = record["text"]
        tokens = tokenize(text)
        token_starts = {token.start for token in tokens}
        token_ends = {token.end for token in tokens}
        edits: list[NormalizedEdit] = []

        for span in record["spans"]:
            assert span["start"] in token_starts
            assert span["end"] in token_ends
            assert text[span["start"] : span["end"]] == span["source"]
            edits.append(
                NormalizedEdit(
                    kind=SpanKind(span["kind"]),
                    source_text=span["source"],
                    source_span=TextSpan(span["start"], span["end"]),
                    normalized_text=span["replacement"],
                    score=1.0,
                )
            )

        result = NormalizationResult(
            source_text=text,
            normalized_text=record["expected_text"],
            edits=tuple(edits),
        )
        assert all(text[token.start : token.end] == token.text for token in tokens)
        assert result.normalized_text == record["expected_text"]


def test_golden_has_representative_and_adversarial_coverage() -> None:
    records = _load(GOLDEN_PATH)

    assert 150 <= len(records) <= 200
    assert {span["kind"] for record in records for span in record["spans"]} == {
        kind.value for kind in SPAN_KINDS
    }
    assert sum(not record["spans"] for record in records) / len(records) >= 0.25

    tags = {tag for record in records for tag in record["tags"]}
    assert {
        "adjacent_spans",
        "already_normalized",
        "boundary_punctuation",
        "capitalization",
        "cardinal_date_collision",
        "context_collision",
        "digit_phone_collision",
        "digit_time_collision",
        "electronic_literal_collision",
        "leading_zero",
        "long_identifier",
        "malformed",
        "measurement_literal_collision",
        "mixed_asr",
        "multiple_spans",
        "negative",
        "ordinal_literal_collision",
        "punctuation_literal_collision",
        "quantity_money_collision",
        "same_class_spans",
        "three_spans",
    } <= tags

    adjacent_span_records = [
        record for record in records if "adjacent_spans" in record["tags"]
    ]
    assert adjacent_span_records
    assert all(
        any(
            left["end"] + 1 == right["start"]
            for left, right in zip(spans, spans[1:], strict=False)
        )
        for record in adjacent_span_records
        for spans in [record["spans"]]
    )

    pairs: dict[str, list[dict[str, object]]] = {}
    for record in records:
        pair_id = record.get("pair_id")
        if pair_id is not None:
            pairs.setdefault(pair_id, []).append(record)
    assert len(pairs) >= 25
    assert all(len(pair) == 2 for pair in pairs.values())
    assert all(
        len(
            {
                tuple(span["kind"] for span in record["spans"])
                for record in pair
            }
        )
        == 2
        for pair in pairs.values()
    )
    _validate_records(records)
