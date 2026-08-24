import re

import pytest
from premove_itn_data.enrichment import donors as donor_module
from premove_itn_data.enrichment.donors import (
    DonorGenerationError,
    generate_electronic_donors,
    generate_phone_donors,
)

from premove_itn import _rust
from premove_itn.labels import SpanKind


def _digits(value: str) -> str:
    return re.sub(r"[^0-9]", "", value)


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
