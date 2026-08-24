import json
import re
from pathlib import Path

import pytest
from premove_itn_data.enrichment import donors as donor_module
from premove_itn_data.enrichment.donors import (
    DonorExtractionError,
    DonorGenerationError,
    generate_electronic_donors,
    generate_phone_donors,
    iter_google_donors,
    iter_rust_collision_donor_pairs,
)

from premove_itn import _rust
from premove_itn.labels import SpanKind


def _digits(value: str) -> str:
    return re.sub(r"[^0-9]", "", value)


def _google_candidate(
    kind: SpanKind,
    spoken: str,
    replacement: str,
    *,
    sentence_number: int,
    source_file: str = "data/external/google_tn/en.tsv",
) -> dict[str, object]:
    return {
        "record": {
            "text": spoken,
            "spans": [
                {
                    "kind": kind.value,
                    "start": 0,
                    "end": len(spoken),
                    "source": spoken,
                    "replacement": replacement,
                }
            ],
        },
        "provenance": {
            "source": "google_tn",
            "source_file": source_file,
            "sentence_number": sentence_number,
        },
    }


def _write_candidates(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def test_extracts_filtered_google_donors_and_preserves_class_collisions(
    tmp_path: Path,
) -> None:
    candidates_path = tmp_path / "candidates.jsonl"
    _write_candidates(
        candidates_path,
        [
            _google_candidate(SpanKind.TIME, "four thirty", "4:30", sentence_number=20),
            _google_candidate(
                SpanKind.DIGIT_SEQUENCE,
                "four thirty",
                "430",
                sentence_number=30,
            ),
            _google_candidate(
                SpanKind.TIME, "four thirty", "04:30", sentence_number=25
            ),
            _google_candidate(SpanKind.TIME, "four thirty", "4:30", sentence_number=10),
            _google_candidate(SpanKind.MONEY, "five dollars", "$5", sentence_number=40),
        ],
    )

    donors = tuple(
        iter_google_donors(
            candidates_path,
            kinds=frozenset({SpanKind.TIME, SpanKind.DIGIT_SEQUENCE}),
        )
    )

    assert donors == (
        donor_module.EnrichmentDonor(
            SpanKind.DIGIT_SEQUENCE,
            "four thirty",
            "430",
            "google_tn/data/external/google_tn/en.tsv/30/0",
        ),
        donor_module.EnrichmentDonor(
            SpanKind.TIME,
            "four thirty",
            "04:30",
            "google_tn/data/external/google_tn/en.tsv/25/0",
        ),
        donor_module.EnrichmentDonor(
            SpanKind.TIME,
            "four thirty",
            "4:30",
            "google_tn/data/external/google_tn/en.tsv/10/0",
        ),
    )


def test_google_donor_order_and_provenance_winner_are_input_order_independent(
    tmp_path: Path,
) -> None:
    records = [
        _google_candidate(SpanKind.DATE, "may fifth", "May 5", sentence_number=9),
        _google_candidate(SpanKind.DATE, "may fifth", "May 5", sentence_number=3),
        _google_candidate(SpanKind.DATE, "june sixth", "June 6", sentence_number=6),
    ]
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    _write_candidates(first_path, records)
    _write_candidates(second_path, list(reversed(records)))

    first = tuple(iter_google_donors(first_path))
    second = tuple(iter_google_donors(second_path))

    assert first == second
    assert len(first) == 2
    assert first[1].provenance.endswith("/3/0")


def test_google_donor_extraction_rejects_invalid_frozen_records(
    tmp_path: Path,
) -> None:
    invalid_json_path = tmp_path / "invalid.jsonl"
    invalid_json_path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(DonorExtractionError, match=r"invalid\.jsonl:1"):
        tuple(iter_google_donors(invalid_json_path))

    invalid_record = _google_candidate(
        SpanKind.TIME,
        "four thirty",
        "4:30",
        sentence_number=1,
    )
    invalid_record["record"]["spans"][0]["source"] = "wrong"
    invalid_record_path = tmp_path / "invalid-record.jsonl"
    _write_candidates(invalid_record_path, [invalid_record])

    with pytest.raises(DonorExtractionError, match="does not match record text"):
        tuple(iter_google_donors(invalid_record_path))

    machine_specific_record = _google_candidate(
        SpanKind.TIME,
        "four thirty",
        "4:30",
        sentence_number=1,
        source_file="/machine-specific/en.tsv",
    )
    machine_specific_path = tmp_path / "machine-specific.jsonl"
    _write_candidates(machine_specific_path, [machine_specific_record])

    with pytest.raises(DonorExtractionError, match="stable relative path"):
        tuple(iter_google_donors(machine_specific_path))


def test_derives_collision_donors_by_cross_validating_each_spoken_form() -> None:
    donors = (
        donor_module.EnrichmentDonor(
            SpanKind.TIME, "eight o eight", "08:08", "google_tn/time/2/0"
        ),
        donor_module.EnrichmentDonor(
            SpanKind.DIGIT_SEQUENCE,
            "eight o eight",
            "808",
            "google_tn/digits/1/0",
        ),
        donor_module.EnrichmentDonor(
            SpanKind.TIME, "noon", "12:00", "google_tn/time/3/0"
        ),
    )

    pairs = tuple(
        iter_rust_collision_donor_pairs(
            donors,
            first_kind=SpanKind.TIME,
            second_kind=SpanKind.DIGIT_SEQUENCE,
            realize=_rust.realize,
        )
    )

    assert len(pairs) == 1
    assert pairs[0][0].spoken == pairs[0][1].spoken == "eight o eight"
    assert pairs[0][0].replacement == "08:08"
    assert pairs[0][1].replacement == "808"
    assert pairs[0][0].kind is SpanKind.TIME
    assert pairs[0][1].kind is SpanKind.DIGIT_SEQUENCE
    assert "google_tn/digits/1/0" in pairs[0][0].provenance


def test_derives_phone_digit_sequence_collision_donors() -> None:
    spoken = "six five zero five five five one two three four"
    donors = (
        donor_module.EnrichmentDonor(
            SpanKind.PHONE,
            spoken,
            "650-555-1234",
            "generated_phone/nanp/1",
        ),
        donor_module.EnrichmentDonor(
            SpanKind.DIGIT_SEQUENCE,
            spoken,
            "6505551234",
            "google_tn/digits/1/0",
        ),
    )

    [phone_digits] = iter_rust_collision_donor_pairs(
        donors,
        first_kind=SpanKind.PHONE,
        second_kind=SpanKind.DIGIT_SEQUENCE,
        realize=_rust.realize,
    )

    phone, digits = phone_digits
    assert phone.kind is SpanKind.PHONE
    assert phone.replacement == "650-555-1234"
    assert digits.kind is SpanKind.DIGIT_SEQUENCE
    assert digits.replacement == "6505551234"
    assert phone.spoken == digits.spoken == spoken


def test_preserves_semantic_collision_when_rust_outputs_match() -> None:
    spoken = "nine one one"
    donors = (
        donor_module.EnrichmentDonor(
            SpanKind.PHONE,
            spoken,
            "911",
            "generated_phone/emergency/0",
        ),
        donor_module.EnrichmentDonor(
            SpanKind.DIGIT_SEQUENCE,
            spoken,
            "911",
            "google_tn/digits/1/0",
        ),
    )

    [semantic_collision] = iter_rust_collision_donor_pairs(
        donors,
        first_kind=SpanKind.PHONE,
        second_kind=SpanKind.DIGIT_SEQUENCE,
        realize=_rust.realize,
    )

    phone, digits = semantic_collision
    assert phone.kind is SpanKind.PHONE
    assert digits.kind is SpanKind.DIGIT_SEQUENCE
    assert phone.spoken == digits.spoken == spoken
    assert phone.replacement == digits.replacement == "911"


def test_collision_donor_derivation_rejects_invalid_kind_contract() -> None:
    donor = donor_module.EnrichmentDonor(
        SpanKind.DATE, "may fifth", "May 5", "google_tn/date/1/0"
    )

    with pytest.raises(ValueError, match="different"):
        tuple(
            iter_rust_collision_donor_pairs(
                (donor,),
                first_kind=SpanKind.TIME,
                second_kind=SpanKind.TIME,
                realize=_rust.realize,
            )
        )
    with pytest.raises(DonorGenerationError, match="outside requested kinds"):
        tuple(
            iter_rust_collision_donor_pairs(
                (donor,),
                first_kind=SpanKind.TIME,
                second_kind=SpanKind.DIGIT_SEQUENCE,
                realize=_rust.realize,
            )
        )


def test_generates_deterministic_rust_validated_phone_donors() -> None:
    first = generate_phone_donors(100, seed="dataset-v1", realize=_rust.realize)
    second = generate_phone_donors(100, seed="dataset-v1", realize=_rust.realize)

    assert first == second
    assert len(first) == 100
    assert len({donor.spoken for donor in first}) == 100
    assert len({donor.replacement for donor in first}) == 100
    assert all(donor.kind is SpanKind.PHONE for donor in first)
    assert all(
        _rust.realize(donor.kind.value, donor.spoken) == donor.replacement
        for donor in first
    )


def test_phone_donors_cover_required_spoken_shapes() -> None:
    donors = generate_phone_donors(100, seed="coverage", realize=_rust.realize)
    spoken = [donor.spoken for donor in donors]
    provenance = [donor.provenance for donor in donors]

    assert any(source.startswith("generated_phone/emergency/") for source in provenance)
    assert any(
        source.startswith("generated_phone/local_7_digit/") for source in provenance
    )
    assert any(
        source.startswith("generated_phone/nanp_10_digit/") for source in provenance
    )
    assert any(
        source.startswith("generated_phone/nanp_country_code/") for source in provenance
    )
    assert any(
        source.startswith("generated_phone/international_uk/") for source in provenance
    )
    assert any(
        source.startswith("generated_phone/international_india/")
        for source in provenance
    )
    assert any(" zero " in f" {value} " for value in spoken)
    assert any(" oh " in f" {value} " for value in spoken)
    assert any(" o " in f" {value} " for value in spoken)
    assert any(" double " in f" {value} " for value in spoken)
    assert any(" triple " in f" {value} " for value in spoken)


def test_different_seed_changes_generated_numbers_not_emergency_numbers() -> None:
    first = generate_phone_donors(20, seed="first", realize=_rust.realize)
    second = generate_phone_donors(20, seed="second", realize=_rust.realize)

    assert first[:10] == second[:10]
    assert first[10:] != second[10:]


def test_generation_advances_past_duplicate_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = donor_module._generated_phone_candidate
    indexes: list[int] = []

    def with_duplicates(seed: str, index: int) -> object:
        indexes.append(index)
        if index in {1, 2}:
            return original(seed, 0)
        return original(seed, index)

    monkeypatch.setattr(donor_module, "_generated_phone_candidate", with_duplicates)

    donors = generate_phone_donors(20, seed="duplicates", realize=_rust.realize)

    assert len(donors) == 20
    assert len({donor.spoken for donor in donors}) == 20
    assert max(indexes) == 11


def test_generated_nanp_numbers_use_valid_area_and_exchange_prefixes() -> None:
    donors = generate_phone_donors(100, seed="nanp", realize=_rust.realize)

    for donor in donors:
        style = donor.provenance.split("/")[1]
        digits = _digits(donor.replacement)
        if style.startswith("nanp_") and style != "nanp_country_code":
            assert digits[0] in "23456789"
            assert digits[3] in "23456789"
        elif style == "nanp_country_code":
            assert digits[0] == "1"
            assert digits[1] in "23456789"
            assert digits[4] in "23456789"


def test_rejects_invalid_requests() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        generate_phone_donors(-1, seed="seed", realize=_rust.realize)
    with pytest.raises(ValueError, match="non-empty"):
        generate_phone_donors(1, seed="", realize=_rust.realize)


def test_rust_rejection_fails_generation() -> None:
    with pytest.raises(DonorGenerationError, match="Rust rejected"):
        generate_phone_donors(1, seed="seed", realize=lambda _kind, _spoken: None)


def test_rust_digit_change_fails_generation() -> None:
    with pytest.raises(DonorGenerationError, match="changed generated PHONE digits"):
        generate_phone_donors(1, seed="seed", realize=lambda _kind, _spoken: "123")


def test_can_generate_no_donors() -> None:
    assert generate_phone_donors(0, seed="seed", realize=_rust.realize) == ()


def test_generates_deterministic_rust_validated_electronic_donors() -> None:
    first = generate_electronic_donors(200, seed="dataset-v1", realize=_rust.realize)
    second = generate_electronic_donors(200, seed="dataset-v1", realize=_rust.realize)

    assert first == second
    assert len(first) == 200
    assert len({donor.spoken for donor in first}) == 200
    assert len({donor.replacement for donor in first}) == 200
    assert all(donor.kind is SpanKind.ELECTRONIC for donor in first)
    assert all(
        _rust.realize(donor.kind.value, donor.spoken) == donor.replacement
        for donor in first
    )


def test_electronic_donors_cover_required_shapes_and_tlds() -> None:
    donors = generate_electronic_donors(200, seed="coverage", realize=_rust.realize)
    styles = {donor.provenance.split("/")[1] for donor in donors}
    replacements = {donor.replacement for donor in donors}

    assert styles == {
        "domain_bare",
        "domain_hyphenated",
        "domain_www",
        "email_dotted",
        "email_hyphenated",
        "email_numeric",
        "email_simple",
        "email_subdomain",
        "email_underscored",
    }
    assert any("@" in value for value in replacements)
    assert any(value.startswith("www.") for value in replacements)
    assert any("-" in value for value in replacements)
    assert any(".co.uk" in value for value in replacements)
    for tld in (".com", ".org", ".net", ".io", ".ai", ".edu"):
        assert any(value.endswith(tld) for value in replacements)


def test_electronic_generation_advances_past_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = donor_module._generated_electronic_candidate
    indexes: list[int] = []

    def with_duplicates(seed: str, index: int) -> object:
        indexes.append(index)
        if index in {1, 2}:
            return original(seed, 0)
        return original(seed, index)

    monkeypatch.setattr(
        donor_module, "_generated_electronic_candidate", with_duplicates
    )

    donors = generate_electronic_donors(20, seed="duplicates", realize=_rust.realize)

    assert len(donors) == 20
    assert len({donor.replacement for donor in donors}) == 20
    assert max(indexes) == 21


def test_electronic_generation_rejects_rust_failure_or_value_change() -> None:
    with pytest.raises(DonorGenerationError, match="Rust rejected"):
        generate_electronic_donors(1, seed="seed", realize=lambda _kind, _spoken: None)
    with pytest.raises(DonorGenerationError, match="changed generated ELECTRONIC"):
        generate_electronic_donors(
            1, seed="seed", realize=lambda _kind, _spoken: "wrong.example"
        )


def test_electronic_generation_rejects_invalid_requests() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        generate_electronic_donors(-1, seed="seed", realize=_rust.realize)
    with pytest.raises(ValueError, match="non-empty"):
        generate_electronic_donors(1, seed="", realize=_rust.realize)
    assert generate_electronic_donors(0, seed="seed", realize=_rust.realize) == ()
