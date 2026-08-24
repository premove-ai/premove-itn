from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

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


class DonorExtractionError(ValueError):
    """Raised when a frozen donor source violates the extraction contract."""


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


def _require_mapping(value: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise DonorExtractionError(f"{location} must be a JSON object")
    return value


def _require_string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise DonorExtractionError(f"{location} must be a non-empty string")
    return value


def _require_int(value: object, *, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DonorExtractionError(f"{location} must be an integer")
    return value


def _require_stable_source_file(value: object, *, location: str) -> str:
    source_file = _require_string(value, location=location)
    posix_path = PurePosixPath(source_file)
    if (
        posix_path.is_absolute()
        or PureWindowsPath(source_file).is_absolute()
        or "\\" in source_file
        or ".." in posix_path.parts
        or posix_path.as_posix() != source_file
    ):
        raise DonorExtractionError(f"{location} must be a stable relative path")
    return source_file


def iter_google_donors(
    candidates_path: Path,
    *,
    kinds: frozenset[SpanKind] | None = None,
) -> Iterator[EnrichmentDonor]:
    """Yield unique trusted span values from frozen Google candidates.

    Donors are deduplicated by exact kind, spoken form, and replacement. The
    lowest source-file, sentence, and span identity wins independent of input
    order. Output is sorted by the same donor key.
    """
    winners: dict[
        tuple[SpanKind, str, str],
        tuple[tuple[str, int, int], EnrichmentDonor],
    ] = {}

    with candidates_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            location = f"{candidates_path}:{line_number}"
            try:
                raw_payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise DonorExtractionError(f"{location}: invalid JSON") from error

            payload = _require_mapping(raw_payload, location=location)
            record = _require_mapping(
                payload.get("record"), location=f"{location}.record"
            )
            provenance = _require_mapping(
                payload.get("provenance"), location=f"{location}.provenance"
            )
            if provenance.get("source") != "google_tn":
                raise DonorExtractionError(
                    f"{location}.provenance.source must be 'google_tn'"
                )
            source_file = _require_stable_source_file(
                provenance.get("source_file"),
                location=f"{location}.provenance.source_file",
            )
            sentence_number = _require_int(
                provenance.get("sentence_number"),
                location=f"{location}.provenance.sentence_number",
            )
            if sentence_number < 0:
                raise DonorExtractionError(
                    f"{location}.provenance.sentence_number must be non-negative"
                )

            text = _require_string(record.get("text"), location=f"{location}.text")
            raw_spans = record.get("spans")
            if not isinstance(raw_spans, list):
                raise DonorExtractionError(f"{location}.spans must be a JSON array")

            for span_index, raw_span in enumerate(raw_spans):
                span_location = f"{location}.spans[{span_index}]"
                span = _require_mapping(raw_span, location=span_location)
                raw_kind = _require_string(
                    span.get("kind"), location=f"{span_location}.kind"
                )
                try:
                    kind = SpanKind(raw_kind)
                except ValueError as error:
                    raise DonorExtractionError(
                        f"{span_location}.kind is unsupported: {raw_kind!r}"
                    ) from error

                start = _require_int(
                    span.get("start"), location=f"{span_location}.start"
                )
                end = _require_int(span.get("end"), location=f"{span_location}.end")
                spoken = _require_string(
                    span.get("source"), location=f"{span_location}.source"
                )
                replacement = _require_string(
                    span.get("replacement"),
                    location=f"{span_location}.replacement",
                )
                if start < 0 or end <= start or end > len(text):
                    raise DonorExtractionError(
                        f"{span_location} has invalid offsets [{start}, {end})"
                    )
                if text[start:end] != spoken:
                    raise DonorExtractionError(
                        f"{span_location}.source does not match record text"
                    )
                if kinds is not None and kind not in kinds:
                    continue

                donor_key = (kind, spoken, replacement)
                source_identity = (source_file, sentence_number, span_index)
                donor = EnrichmentDonor(
                    kind=kind,
                    spoken=spoken,
                    replacement=replacement,
                    provenance=(
                        f"google_tn/{source_file}/{sentence_number}/{span_index}"
                    ),
                )
                current = winners.get(donor_key)
                if current is None or source_identity < current[0]:
                    winners[donor_key] = (source_identity, donor)

    for donor_key in sorted(
        winners,
        key=lambda key: (key[0].value, key[1], key[2]),
    ):
        yield winners[donor_key][1]


def iter_rust_collision_donor_pairs(
    donors: Iterable[EnrichmentDonor],
    *,
    first_kind: SpanKind,
    second_kind: SpanKind,
    realize: Realize,
) -> Iterator[tuple[EnrichmentDonor, EnrichmentDonor]]:
    """Yield spoken forms that Rust accepts under both requested kinds."""
    if first_kind is second_kind:
        raise ValueError("collision kinds must be different")

    requested_kinds = {first_kind, second_kind}
    source_by_spoken: dict[str, EnrichmentDonor] = {}
    for donor in donors:
        if donor.kind not in requested_kinds:
            raise DonorGenerationError(
                f"{donor.kind.value} donor is outside requested kinds"
            )
        if not donor.spoken or not donor.replacement or not donor.provenance:
            raise DonorGenerationError("collision donor fields must be non-empty")
        current = source_by_spoken.get(donor.spoken)
        if current is None or donor.provenance < current.provenance:
            source_by_spoken[donor.spoken] = donor

    for spoken in sorted(source_by_spoken):
        source = source_by_spoken[spoken]
        first_replacement = realize(first_kind.value, spoken)
        second_replacement = realize(second_kind.value, spoken)
        if first_replacement is None or second_replacement is None:
            continue
        provenance = f"rust_collision/{source.provenance}"
        yield (
            EnrichmentDonor(
                kind=first_kind,
                spoken=spoken,
                replacement=first_replacement,
                provenance=f"{provenance}/{first_kind.value.lower()}",
            ),
            EnrichmentDonor(
                kind=second_kind,
                spoken=spoken,
                replacement=second_replacement,
                provenance=f"{provenance}/{second_kind.value.lower()}",
            ),
        )


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
