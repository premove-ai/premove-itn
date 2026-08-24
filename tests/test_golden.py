import hashlib
import json
from collections import Counter
from pathlib import Path

from premove_itn import NormalizationResult, NormalizedEdit, SpanKind, TextSpan
from premove_itn.tokenize import tokenize

DATA_DIR = Path(__file__).parents[1] / "data"
GOLDEN_PATH = DATA_DIR / "golden.json"
GOLDEN_CHECKSUM_PATH = DATA_DIR / "golden.sha256"
GOLDEN_MANIFEST_PATH = DATA_DIR / "golden_manifest.json"
DEFERRED_PATH = DATA_DIR / "deferred.json"
DEFERRED_CHECKSUM_PATH = DATA_DIR / "deferred.sha256"

DATASET_V1_KINDS = {
    SpanKind.CARDINAL,
    SpanKind.DATE,
    SpanKind.DECIMAL,
    SpanKind.DIGIT_SEQUENCE,
    SpanKind.ELECTRONIC,
    SpanKind.MEASUREMENT,
    SpanKind.MONEY,
    SpanKind.ORDINAL,
    SpanKind.PHONE,
    SpanKind.TIME,
}
DEFERRED_KINDS = {
    SpanKind.PUNCTUATION,
    SpanKind.WHITELIST,
    SpanKind.WORD,
}


def _load(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text())


def test_golden_file_matches_frozen_checksum() -> None:
    assert hashlib.sha256(GOLDEN_PATH.read_bytes()).hexdigest() == (
        GOLDEN_CHECKSUM_PATH.read_text().strip()
    )


def test_deferred_file_matches_frozen_checksum() -> None:
    assert hashlib.sha256(DEFERRED_PATH.read_bytes()).hexdigest() == (
        DEFERRED_CHECKSUM_PATH.read_text().strip()
    )


def test_golden_manifest_matches_frozen_benchmark() -> None:
    records = _load(GOLDEN_PATH)
    deferred = _load(DEFERRED_PATH)
    manifest = json.loads(GOLDEN_MANIFEST_PATH.read_text())

    assert manifest["golden_sha256"] == GOLDEN_CHECKSUM_PATH.read_text().strip()
    assert manifest["deferred_sha256"] == DEFERRED_CHECKSUM_PATH.read_text().strip()
    assert manifest["records"] == len(records)
    assert manifest["negative_records"] == sum(
        not record["spans"] for record in records
    )
    assert manifest["multi_span_records"] == sum(
        len(record["spans"]) > 1 for record in records
    )
    assert manifest["spans"] == sum(len(record["spans"]) for record in records)
    assert manifest["spans_by_kind"] == dict(
        sorted(
            Counter(
                span["kind"] for record in records for span in record["spans"]
            ).items()
        )
    )
    assert manifest["minimal_pairs"] == len(
        {record["pair_id"] for record in records if "pair_id" in record}
    )
    assert manifest["class_contract"] == sorted(kind.value for kind in DATASET_V1_KINDS)
    assert manifest["scope_decisions"]["deferred_classes"] == sorted(
        kind.value for kind in DEFERRED_KINDS
    )
    assert manifest["deferred_file"] == DEFERRED_PATH.name
    assert len(deferred) == 25
    assert manifest["leakage_audit"] == {
        "exact_compiled_record_overlap_train": 0,
        "exact_compiled_record_overlap_validation": 0,
        "exact_text_overlap_train": 0,
        "exact_text_overlap_validation": 0,
        "span_values_seen_in_train": 48,
        "span_values_unseen_in_train": 57,
        "unseen_span_values_by_kind": {
            "CARDINAL": 4,
            "DATE": 4,
            "DECIMAL": 6,
            "DIGIT_SEQUENCE": 18,
            "ELECTRONIC": 6,
            "MEASUREMENT": 5,
            "MONEY": 4,
            "ORDINAL": 1,
            "PHONE": 5,
            "TIME": 4,
        },
        "unseen_value_records": 57,
    }
    assert sum("unseen_value" in record["tags"] for record in records) == 57


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

    assert 120 <= len(records) <= 200
    assert {span["kind"] for record in records for span in record["spans"]} == {
        kind.value for kind in DATASET_V1_KINDS
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
    assert len(pairs) >= 20
    assert all(len(pair) == 2 for pair in pairs.values())
    assert all(
        len({tuple(span["kind"] for span in record["spans"]) for record in pair}) == 2
        for pair in pairs.values()
    )
    _validate_records(records)


def test_deferred_cases_are_separate_and_structurally_valid() -> None:
    golden = _load(GOLDEN_PATH)
    deferred = _load(DEFERRED_PATH)

    assert len(deferred) == 25
    assert {span["kind"] for record in deferred for span in record["spans"]} == {
        kind.value for kind in DEFERRED_KINDS
    }
    assert not (
        {record["id"] for record in golden} & {record["id"] for record in deferred}
    )
    assert not (
        {record["text"] for record in golden} & {record["text"] for record in deferred}
    )

    pairs: dict[str, list[dict[str, object]]] = {}
    for record in deferred:
        pair_id = record.get("pair_id")
        if pair_id is not None:
            pairs.setdefault(pair_id, []).append(record)
    assert len(pairs) == 6
    assert all(len(pair) == 2 for pair in pairs.values())
    _validate_records(deferred)


def test_dataset_v1_semantic_scope_is_explicit() -> None:
    records = {record["id"]: record for record in _load(GOLDEN_PATH)}

    room = records["hard_007_negative"]
    assert room["expected_text"] == "the room number is four thirty"
    assert room["spans"] == []
    assert "realizer_rejected" in room["tags"]

    extension = records["phone_004"]
    assert "phone_extension" in extension["tags"]
    assert [span["kind"] for span in extension["spans"]] == ["PHONE"]

    assert "phone_005" not in records
    assert all("network_address" not in record["tags"] for record in records.values())
