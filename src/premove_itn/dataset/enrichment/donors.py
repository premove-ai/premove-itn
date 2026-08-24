from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass

from premove_itn.labels import SpanKind

Realize = Callable[[str, str], str | None]

_DIGIT_WORDS = (
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
)
_ZERO_ALIASES = ("zero", "oh", "o")
_EMERGENCY_NUMBERS = (
    "911",
    "112",
    "999",
    "988",
    "311",
    "411",
    "511",
    "611",
    "711",
    "811",
)
_NON_DIGIT_PATTERN = re.compile(r"[^0-9]")
_ELECTRONIC_GIVEN_NAMES = (
    "alex",
    "arya",
    "casey",
    "chris",
    "devon",
    "jamie",
    "jordan",
    "kai",
    "maya",
    "morgan",
    "nina",
    "noah",
    "priya",
    "riley",
    "robin",
    "sam",
    "sasha",
    "taylor",
    "zara",
)
_ELECTRONIC_SURNAMES = (
    "adams",
    "ali",
    "brown",
    "chen",
    "davis",
    "garcia",
    "gupta",
    "jones",
    "khan",
    "kim",
    "lee",
    "martin",
    "miller",
    "patel",
    "rivera",
    "singh",
    "smith",
    "thomas",
    "wilson",
)
_ELECTRONIC_DOMAIN_MODIFIERS = (
    "bright",
    "clear",
    "cloud",
    "core",
    "digital",
    "direct",
    "first",
    "global",
    "green",
    "hello",
    "local",
    "modern",
    "next",
    "north",
    "open",
    "prime",
    "rapid",
    "smart",
    "true",
)
_ELECTRONIC_DOMAIN_NOUNS = (
    "bridge",
    "circle",
    "desk",
    "group",
    "hub",
    "labs",
    "link",
    "mail",
    "market",
    "network",
    "point",
    "portal",
    "space",
    "studio",
    "systems",
    "team",
    "works",
    "world",
)
_ELECTRONIC_PROVIDERS = (
    ("gmail", "com"),
    ("outlook", "com"),
    ("proton", "me"),
    ("icloud", "com"),
    ("yahoo", "com"),
    ("fastmail", "com"),
)
_ELECTRONIC_TLDS = (
    ("com", "com"),
    ("org", "org"),
    ("net", "net"),
    ("io", "io"),
    ("ai", "ai"),
    ("edu", "edu"),
    ("dev", "dev"),
    ("app", "app"),
    ("me", "me"),
    ("co dot uk", "co.uk"),
)


class DonorGenerationError(ValueError):
    """Raised when a generated donor violates the realization contract."""


@dataclass(frozen=True, slots=True)
class EnrichmentDonor:
    kind: SpanKind
    spoken: str
    replacement: str
    provenance: str


@dataclass(frozen=True, slots=True)
class _PhoneCandidate:
    style: str
    spoken: str
    expected_digits: str


@dataclass(frozen=True, slots=True)
class _ElectronicCandidate:
    style: str
    spoken: str
    expected: str


def _digest(seed: str, index: int, label: str) -> bytes:
    identity = f"phone-donor-v1\0{seed}\0{index}\0{label}"
    return hashlib.sha256(identity.encode("utf-8")).digest()


def _electronic_digest(seed: str, index: int, label: str) -> bytes:
    identity = f"electronic-donor-v1\0{seed}\0{index}\0{label}"
    return hashlib.sha256(identity.encode("utf-8")).digest()


def _electronic_choice(
    values: tuple[str, ...], seed: str, index: int, label: str
) -> str:
    digest = _electronic_digest(seed, index, label)
    return values[int.from_bytes(digest, "big") % len(values)]


def _digit_string(
    seed: str,
    index: int,
    label: str,
    length: int,
    *,
    first_digits: str = "0123456789",
) -> str:
    digest = _digest(seed, index, label)
    digits = [first_digits[digest[0] % len(first_digits)]]
    digits.extend(str(byte % 10) for byte in digest[1:length])
    return "".join(digits)


def _nanp_digits(seed: str, index: int, label: str) -> str:
    digits = list(_digit_string(seed, index, label, 10, first_digits="23456789"))
    digest = _digest(seed, index, f"{label}_exchange")
    digits[3] = "23456789"[digest[0] % 8]
    return "".join(digits)


def _render_digits(digits: str, *, zero_aliases: bool, repetitions: bool) -> str:
    words: list[str] = []
    index = 0
    while index < len(digits):
        digit = digits[index]
        run_end = index + 1
        while run_end < len(digits) and digits[run_end] == digit:
            run_end += 1
        run_length = run_end - index
        if repetitions and run_length >= 2:
            if run_length >= 3:
                words.extend(("triple", _DIGIT_WORDS[int(digit)]))
                index += 3
            else:
                words.extend(("double", _DIGIT_WORDS[int(digit)]))
                index += 2
            continue

        if digit == "0" and zero_aliases:
            words.append(_ZERO_ALIASES[index % len(_ZERO_ALIASES)])
        else:
            words.append(_DIGIT_WORDS[int(digit)])
        index += 1
    return " ".join(words)


def _force_zero(digits: str, index: int, *, protected_positions: frozenset[int]) -> str:
    positions = [
        position
        for position in range(len(digits))
        if position not in protected_positions
    ]
    position = positions[index % len(positions)]
    return f"{digits[:position]}0{digits[position + 1 :]}"


def _force_repetition(digits: str, index: int) -> str:
    run_length = 2 + index % 2
    start = len(digits) - run_length
    repeated_digit = str(_digest(digits, index, "repeat")[0] % 10)
    return f"{digits[:start]}{repeated_digit * run_length}"


def _generated_phone_candidate(seed: str, index: int) -> _PhoneCandidate:
    style_index = index % 8
    if style_index == 0:
        digits = _nanp_digits(seed, index, "nanp_plain")
        style = "nanp_10_digit"
        spoken = _render_digits(digits, zero_aliases=False, repetitions=False)
        expected = digits
    elif style_index == 1:
        digits = _force_zero(
            _nanp_digits(seed, index, "nanp_zero"),
            index,
            protected_positions=frozenset({0, 3}),
        )
        style = "nanp_zero_alias"
        spoken = _render_digits(digits, zero_aliases=True, repetitions=False)
        expected = digits
    elif style_index == 2:
        digits = _force_repetition(
            _nanp_digits(seed, index, "nanp_repeat"),
            index // 8,
        )
        style = "nanp_repetition"
        spoken = _render_digits(digits, zero_aliases=False, repetitions=True)
        expected = digits
    elif style_index == 3:
        digits = _digit_string(seed, index, "local_plain", 7, first_digits="23456789")
        style = "local_7_digit"
        spoken = _render_digits(digits, zero_aliases=False, repetitions=False)
        expected = digits
    elif style_index == 4:
        digits = _force_repetition(
            _digit_string(seed, index, "local_repeat", 7, first_digits="23456789"),
            index // 8,
        )
        style = "local_repetition"
        spoken = _render_digits(digits, zero_aliases=False, repetitions=True)
        expected = digits
    elif style_index == 5:
        digits = _nanp_digits(seed, index, "nanp_country")
        style = "nanp_country_code"
        spoken = f"one {_render_digits(digits, zero_aliases=True, repetitions=False)}"
        expected = f"1{digits}"
    elif style_index == 6:
        digits = f"20{_digit_string(seed, index, 'uk', 8)}"
        style = "international_uk"
        spoken = (
            f"plus forty four "
            f"{_render_digits(digits, zero_aliases=True, repetitions=False)}"
        )
        expected = f"+44{digits}"
    else:
        digits = _digit_string(seed, index, "india", 10, first_digits="6789")
        style = "international_india"
        spoken = (
            f"plus nine one "
            f"{_render_digits(digits, zero_aliases=True, repetitions=False)}"
        )
        expected = f"+91{digits}"
    return _PhoneCandidate(style, spoken, expected)


def _canonical_phone_digits(value: str) -> str:
    prefix = "+" if value.lstrip().startswith("+") else ""
    return f"{prefix}{_NON_DIGIT_PATTERN.sub('', value)}"


def _validate_phone_candidate(
    candidate: _PhoneCandidate,
    *,
    realize: Realize,
    provenance: str,
) -> EnrichmentDonor:
    replacement = realize(SpanKind.PHONE.value, candidate.spoken)
    if replacement is None:
        raise DonorGenerationError(
            f"Rust rejected generated PHONE donor {provenance}: {candidate.spoken!r}"
        )
    if _canonical_phone_digits(replacement) != candidate.expected_digits:
        raise DonorGenerationError(
            f"Rust changed generated PHONE digits for {provenance}: "
            f"expected {candidate.expected_digits!r}, got {replacement!r}"
        )
    return EnrichmentDonor(
        kind=SpanKind.PHONE,
        spoken=candidate.spoken,
        replacement=replacement,
        provenance=provenance,
    )


def generate_phone_donors(
    count: int,
    *,
    seed: str,
    realize: Realize,
) -> tuple[EnrichmentDonor, ...]:
    """Generate unique deterministic PHONE donors accepted by Rust.

    The generator controls telephone semantics and intended digits. Rust owns
    the written replacement and acts as the final acceptance gate.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if not seed:
        raise ValueError("seed must be non-empty")

    emergency_candidates = (
        _PhoneCandidate(
            "emergency",
            _render_digits(number, zero_aliases=False, repetitions=False),
            number,
        )
        for number in _EMERGENCY_NUMBERS
    )

    donors: list[EnrichmentDonor] = []
    seen_spoken: set[str] = set()
    seen_digits: set[str] = set()

    def add_candidate(candidate: _PhoneCandidate, candidate_index: int) -> None:
        canonical_spoken = candidate.spoken.casefold()
        if canonical_spoken in seen_spoken or candidate.expected_digits in seen_digits:
            return
        provenance = f"generated_phone/{candidate.style}/{candidate_index:06d}"
        donor = _validate_phone_candidate(
            candidate, realize=realize, provenance=provenance
        )
        donors.append(donor)
        seen_spoken.add(canonical_spoken)
        seen_digits.add(candidate.expected_digits)

    for candidate_index, candidate in enumerate(emergency_candidates):
        if len(donors) == count:
            break
        add_candidate(candidate, candidate_index)

    generated_index = 0
    maximum_attempts = count * 10 + 100
    while len(donors) < count and generated_index < maximum_attempts:
        candidate = _generated_phone_candidate(seed, generated_index)
        add_candidate(candidate, len(_EMERGENCY_NUMBERS) + generated_index)
        generated_index += 1

    if len(donors) != count:
        raise DonorGenerationError(
            f"could not generate {count} unique PHONE donors after "
            f"{maximum_attempts} generated attempts; generated {len(donors)}"
        )
    return tuple(donors)


def _generated_electronic_candidate(seed: str, index: int) -> _ElectronicCandidate:
    style_index = index % 9
    occurrence = index // 9
    given = _electronic_choice(_ELECTRONIC_GIVEN_NAMES, seed, index, "given")
    surname = _electronic_choice(_ELECTRONIC_SURNAMES, seed, index, "surname")
    modifier = _electronic_choice(_ELECTRONIC_DOMAIN_MODIFIERS, seed, index, "modifier")
    noun = _electronic_choice(_ELECTRONIC_DOMAIN_NOUNS, seed, index, "noun")
    tld_spoken, tld_written = _ELECTRONIC_TLDS[occurrence % len(_ELECTRONIC_TLDS)]
    provider, provider_tld = _ELECTRONIC_PROVIDERS[
        occurrence % len(_ELECTRONIC_PROVIDERS)
    ]

    if style_index == 0:
        return _ElectronicCandidate(
            "email_simple",
            f"{given} {surname} at {provider} dot {provider_tld}",
            f"{given}{surname}@{provider}.{provider_tld}",
        )
    if style_index == 1:
        return _ElectronicCandidate(
            "email_dotted",
            f"{given} dot {surname} at {provider} dot {provider_tld}",
            f"{given}.{surname}@{provider}.{provider_tld}",
        )
    if style_index == 2:
        return _ElectronicCandidate(
            "email_hyphenated",
            f"{given} hyphen {surname} at {provider} dot {provider_tld}",
            f"{given}-{surname}@{provider}.{provider_tld}",
        )
    if style_index == 3:
        return _ElectronicCandidate(
            "email_underscored",
            f"{given} underscore {surname} at {provider} dot {provider_tld}",
            f"{given}_{surname}@{provider}.{provider_tld}",
        )
    if style_index == 4:
        number = (
            100
            + int.from_bytes(_electronic_digest(seed, index, "number")[:2], "big") % 900
        )
        number_spoken = _render_digits(
            str(number), zero_aliases=False, repetitions=False
        )
        return _ElectronicCandidate(
            "email_numeric",
            f"{given} {number_spoken} at {provider} dot {provider_tld}",
            f"{given}{number}@{provider}.{provider_tld}",
        )
    if style_index == 5:
        return _ElectronicCandidate(
            "email_subdomain",
            f"{given} at mail dot {modifier} {noun} dot {tld_spoken}",
            f"{given}@mail.{modifier}{noun}.{tld_written}",
        )
    if style_index == 6:
        return _ElectronicCandidate(
            "domain_bare",
            f"{modifier} {noun} dot {tld_spoken}",
            f"{modifier}{noun}.{tld_written}",
        )
    if style_index == 7:
        return _ElectronicCandidate(
            "domain_hyphenated",
            f"{modifier} hyphen {noun} dot {tld_spoken}",
            f"{modifier}-{noun}.{tld_written}",
        )
    return _ElectronicCandidate(
        "domain_www",
        f"w w w dot {modifier} {noun} dot {tld_spoken}",
        f"www.{modifier}{noun}.{tld_written}",
    )


def _validate_electronic_candidate(
    candidate: _ElectronicCandidate,
    *,
    realize: Realize,
    provenance: str,
) -> EnrichmentDonor:
    replacement = realize(SpanKind.ELECTRONIC.value, candidate.spoken)
    if replacement is None:
        raise DonorGenerationError(
            f"Rust rejected generated ELECTRONIC donor {provenance}: "
            f"{candidate.spoken!r}"
        )
    if replacement != candidate.expected:
        raise DonorGenerationError(
            f"Rust changed generated ELECTRONIC value for {provenance}: "
            f"expected {candidate.expected!r}, got {replacement!r}"
        )
    return EnrichmentDonor(
        kind=SpanKind.ELECTRONIC,
        spoken=candidate.spoken,
        replacement=replacement,
        provenance=provenance,
    )


def generate_electronic_donors(
    count: int,
    *,
    seed: str,
    realize: Realize,
) -> tuple[EnrichmentDonor, ...]:
    """Generate unique deterministic ELECTRONIC donors accepted by Rust."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if not seed:
        raise ValueError("seed must be non-empty")

    donors: list[EnrichmentDonor] = []
    seen_spoken: set[str] = set()
    seen_replacements: set[str] = set()
    generated_index = 0
    maximum_attempts = count * 10 + 100
    while len(donors) < count and generated_index < maximum_attempts:
        candidate = _generated_electronic_candidate(seed, generated_index)
        canonical_spoken = candidate.spoken.casefold()
        if (
            canonical_spoken not in seen_spoken
            and candidate.expected not in seen_replacements
        ):
            provenance = f"generated_electronic/{candidate.style}/{generated_index:06d}"
            donor = _validate_electronic_candidate(
                candidate, realize=realize, provenance=provenance
            )
            donors.append(donor)
            seen_spoken.add(canonical_spoken)
            seen_replacements.add(candidate.expected)
        generated_index += 1

    if len(donors) != count:
        raise DonorGenerationError(
            f"could not generate {count} unique ELECTRONIC donors after "
            f"{maximum_attempts} attempts; generated {len(donors)}"
        )
    return tuple(donors)
