import pytest
from premove_itn_data.enrichment import generate as generate_module
from premove_itn_data.enrichment.donors import EnrichmentDonor
from premove_itn_data.enrichment.generate import (
    EnrichmentCandidate,
    EnrichmentGenerationError,
    EnrichmentSpan,
    generate_collision_candidates,
    generate_purpose_built_candidate,
    generate_sgd_candidate,
    generate_sgd_context_only_candidate,
)
from premove_itn_data.enrichment.sgd_context import SgdSlotSpan, SgdUserTurn

from premove_itn.labels import SpanKind


def _donor(
    kind: SpanKind,
    spoken: str,
    replacement: str,
    provenance: str,
) -> EnrichmentDonor:
    return EnrichmentDonor(kind, spoken, replacement, provenance)


def _slot(
    utterance: str,
    value: str,
    *,
    service: str,
    slot: str,
    description: str,
) -> SgdSlotSpan:
    start = utterance.index(value)
    return SgdSlotSpan(
        service=service,
        slot=slot,
        description=description,
        start=start,
        end=start + len(value),
        value=value,
    )


def _turn(utterance: str, spans: tuple[SgdSlotSpan, ...]) -> SgdUserTurn:
    return SgdUserTurn(
        source_file="train/dialogues_001.json",
        dialogue_id="1_00000",
        turn_index=3,
        utterance=utterance,
        services=tuple(dict.fromkeys(span.service for span in spans)),
        spans=spans,
    )


def test_generates_sgd_candidate_with_exact_context_and_donor_lineage() -> None:
    utterance = "Book it at 7:30 pm please."
    turn = _turn(
        utterance,
        (
            _slot(
                utterance,
                "7:30 pm",
                service="Restaurants_1",
                slot="time",
                description="Time for the reservation or to find availability",
            ),
        ),
    )
    donor = _donor(SpanKind.TIME, "four thirty", "4:30", "google_tn/time/1/0")

    candidate = generate_sgd_candidate(
        turn,
        {SpanKind.TIME: (donor,)},
        seed="dataset-v1",
    )

    assert candidate == EnrichmentCandidate(
        text="Book it at four thirty please.",
        context_provenance="sgd/train/dialogues_001.json/1_00000/3",
        spans=(
            EnrichmentSpan(
                kind=SpanKind.TIME,
                start=11,
                end=22,
                source="four thirty",
                replacement="4:30",
                context_provenance="sgd_slot/Restaurants_1/time/11/18",
                donor_provenance="google_tn/time/1/0",
            ),
        ),
    )


def test_sgd_generation_rewrites_multiple_spans_and_recomputes_offsets() -> None:
    utterance = "Book May 5 at 7:30."
    date_span = _slot(
        utterance,
        "May 5",
        service="Restaurants_1",
        slot="date",
        description="Date for the reservation or to find availability",
    )
    time_span = _slot(
        utterance,
        "7:30",
        service="Restaurants_1",
        slot="time",
        description="Time for the reservation or to find availability",
    )

    candidate = generate_sgd_candidate(
        _turn(utterance, (time_span, date_span)),
        {
            SpanKind.DATE: (
                _donor(SpanKind.DATE, "june sixth", "June 6", "google_tn/date/1/0"),
            ),
            SpanKind.TIME: (
                _donor(SpanKind.TIME, "four thirty", "4:30", "google_tn/time/1/0"),
            ),
        },
        seed="dataset-v1",
    )

    assert candidate is not None
    assert candidate.text == "Book june sixth at four thirty."
    assert [
        (span.kind, span.start, span.end, candidate.text[span.start : span.end])
        for span in candidate.spans
    ] == [
        (SpanKind.DATE, 5, 15, "june sixth"),
        (SpanKind.TIME, 19, 30, "four thirty"),
    ]


def test_sgd_generation_ignores_unapproved_slots_and_empty_contexts() -> None:
    utterance = "Find food in Boston."
    city = _slot(
        utterance,
        "Boston",
        service="Restaurants_1",
        slot="city",
        description="City where the restaurant is located",
    )

    assert (
        generate_sgd_candidate(_turn(utterance, (city,)), {}, seed="dataset-v1") is None
    )


def test_generates_context_only_candidate_from_conversational_sgd_turn() -> None:
    utterance = "Find food in Boston."
    city = _slot(
        utterance,
        "Boston",
        service="Restaurants_1",
        slot="city",
        description="City where the restaurant is located",
    )

    candidate = generate_sgd_context_only_candidate(_turn(utterance, (city,)))

    assert candidate == EnrichmentCandidate(
        text=utterance,
        context_provenance="sgd/train/dialogues_001.json/1_00000/3",
        spans=(),
    )
    assert generate_sgd_context_only_candidate(_turn("Thanks.", ())) == (
        EnrichmentCandidate(
            text="Thanks.",
            context_provenance="sgd/train/dialogues_001.json/1_00000/3",
            spans=(),
        )
    )


def test_context_only_route_rejects_turns_with_approved_itn_spans() -> None:
    utterance = "Book it at 7:30."
    time_span = _slot(
        utterance,
        "7:30",
        service="Restaurants_1",
        slot="time",
        description="Time for the reservation or to find availability",
    )

    assert generate_sgd_context_only_candidate(_turn(utterance, (time_span,))) is None


def test_sgd_generation_excludes_partial_token_span_from_both_routes() -> None:
    utterance = "I want a ticket for today's event."
    date_span = _slot(
        utterance,
        "today",
        service="Events_2",
        slot="date",
        description="Date of event",
    )
    turn = _turn(utterance, (date_span,))
    donor = _donor(SpanKind.DATE, "july twenty seventh", "July 27", "date")

    assert (
        generate_sgd_candidate(
            turn,
            {SpanKind.DATE: (donor,)},
            seed="dataset-v1",
        )
        is None
    )
    assert generate_sgd_context_only_candidate(turn) is None


def test_sgd_donor_choice_is_independent_of_pool_order() -> None:
    utterance = "Book it at 7:30."
    turn = _turn(
        utterance,
        (
            _slot(
                utterance,
                "7:30",
                service="Restaurants_1",
                slot="time",
                description="Time for the reservation or to find availability",
            ),
        ),
    )
    donors = (
        _donor(SpanKind.TIME, "four thirty", "4:30", "google_tn/time/1/0"),
        _donor(SpanKind.TIME, "six fifteen", "6:15", "google_tn/time/2/0"),
    )

    first = generate_sgd_candidate(turn, {SpanKind.TIME: donors}, seed="dataset-v1")
    second = generate_sgd_candidate(
        turn, {SpanKind.TIME: tuple(reversed(donors))}, seed="dataset-v1"
    )

    assert first == second


def test_sgd_generation_rejects_missing_donors_and_overlapping_spans() -> None:
    utterance = "Book it at 7:30."
    time_span = _slot(
        utterance,
        "7:30",
        service="Restaurants_1",
        slot="time",
        description="Time for the reservation or to find availability",
    )

    with pytest.raises(EnrichmentGenerationError, match="no TIME donors"):
        generate_sgd_candidate(_turn(utterance, (time_span,)), {}, seed="dataset-v1")

    overlapping = SgdSlotSpan(
        service="Restaurants_1",
        slot="date",
        description="Date for the reservation or to find availability",
        start=time_span.start,
        end=time_span.end + 1,
        value=utterance[time_span.start : time_span.end + 1],
    )
    with pytest.raises(
        EnrichmentGenerationError, match="overlapping approved SGD spans"
    ):
        generate_sgd_candidate(
            _turn(utterance, (time_span, overlapping)),
            {
                SpanKind.TIME: (_donor(SpanKind.TIME, "four thirty", "4:30", "time"),),
                SpanKind.DATE: (_donor(SpanKind.DATE, "may fifth", "May 5", "date"),),
            },
            seed="dataset-v1",
        )


def test_generates_purpose_built_phone_email_domain_and_code_contexts() -> None:
    donors = (
        _donor(SpanKind.PHONE, "nine one one", "911", "generated_phone/1"),
        _donor(
            SpanKind.ELECTRONIC,
            "alex at gmail dot com",
            "alex@gmail.com",
            "generated_electronic/email/1",
        ),
        _donor(
            SpanKind.ELECTRONIC,
            "example dot com",
            "example.com",
            "generated_electronic/domain/1",
        ),
        _donor(
            SpanKind.DIGIT_SEQUENCE,
            "four thirty",
            "430",
            "google_tn/digits/1/0",
        ),
    )

    candidates = tuple(
        generate_purpose_built_candidate(
            donor,
            seed="dataset-v1",
            index=index,
        )
        for index, donor in enumerate(donors)
    )

    assert "email" in candidates[1].context_provenance
    assert "domain" in candidates[2].context_provenance
    for candidate, donor in zip(candidates, donors, strict=True):
        assert len(candidate.spans) == 1
        [span] = candidate.spans
        assert candidate.text[span.start : span.end] == donor.spoken
        assert span.kind is donor.kind
        assert span.replacement == donor.replacement
        assert span.donor_provenance == donor.provenance


def test_purpose_built_template_banks_have_v1_context_diversity() -> None:
    template_banks = (
        (generate_module._PHONE_TEMPLATES, 25, 40),
        (generate_module._ELECTRONIC_EMAIL_TEMPLATES, 20, 30),
        (generate_module._ELECTRONIC_DOMAIN_TEMPLATES, 15, 20),
        (generate_module._DIGIT_SEQUENCE_TEMPLATES, 25, 40),
    )

    for templates, minimum, maximum in template_banks:
        assert minimum <= len(templates) <= maximum
        assert len({template.family for template in templates}) == len(templates)
        structures = {(template.prefix, template.suffix) for template in templates}
        assert len(structures) == len(templates)

    assert len(generate_module._PHONE_COLLISION_TEMPLATES) == 8
    assert len(
        {
            (template.prefix, template.suffix)
            for template in generate_module._PHONE_COLLISION_TEMPLATES
        }
    ) == len(generate_module._PHONE_COLLISION_TEMPLATES)


def test_purpose_built_generation_rejects_unsupported_kind() -> None:
    donor = _donor(SpanKind.TIME, "four thirty", "4:30", "google_tn/time/1/0")

    with pytest.raises(EnrichmentGenerationError, match="unsupported purpose-built"):
        generate_purpose_built_candidate(donor, seed="dataset-v1", index=0)


def test_generates_deliberate_time_digit_sequence_collision_pair() -> None:
    time = _donor(SpanKind.TIME, "four thirty", "4:30", "google_tn/time/1/0")
    digits = _donor(
        SpanKind.DIGIT_SEQUENCE,
        "four thirty",
        "430",
        "google_tn/digits/1/0",
    )

    pair = generate_collision_candidates(
        time,
        digits,
        seed="dataset-v1",
        index=7,
    )

    assert len(pair) == 2
    assert {candidate.spans[0].kind for candidate in pair} == {
        SpanKind.TIME,
        SpanKind.DIGIT_SEQUENCE,
    }
    assert {candidate.spans[0].source for candidate in pair} == {"four thirty"}
    assert {candidate.spans[0].replacement for candidate in pair} == {"4:30", "430"}
    assert all("collision" in candidate.context_provenance for candidate in pair)
    assert all(
        candidate.text[span.start : span.end] == "four thirty"
        for candidate in pair
        for span in candidate.spans
    )


def test_generates_deliberate_phone_digit_sequence_collision_pair() -> None:
    spoken = "six five zero five five five one two three four"
    phone = _donor(
        SpanKind.PHONE,
        spoken,
        "650-555-1234",
        "rust_collision/generated_phone/phone",
    )
    digits = _donor(
        SpanKind.DIGIT_SEQUENCE,
        spoken,
        "6505551234",
        "rust_collision/generated_phone/digit_sequence",
    )

    pair = generate_collision_candidates(phone, digits, seed="dataset-v1", index=8)

    assert [candidate.spans[0].kind for candidate in pair] == [
        SpanKind.PHONE,
        SpanKind.DIGIT_SEQUENCE,
    ]
    assert {candidate.spans[0].source for candidate in pair} == {spoken}
    assert {candidate.spans[0].replacement for candidate in pair} == {
        "650-555-1234",
        "6505551234",
    }
    assert "/phone/" in pair[0].context_provenance
    assert "/digit_sequence/" in pair[1].context_provenance
    assert all(
        candidate.text[span.start : span.end] == spoken
        for candidate in pair
        for span in candidate.spans
    )


@pytest.mark.parametrize(
    ("first", "second", "match"),
    [
        (
            _donor(SpanKind.TIME, "four thirty", "4:30", "time"),
            _donor(SpanKind.DIGIT_SEQUENCE, "four twenty", "420", "digits"),
            "same spoken form",
        ),
        (
            _donor(SpanKind.TIME, "four thirty", "4:30", "time"),
            _donor(SpanKind.DATE, "four thirty", "April 30", "date"),
            "unsupported collision",
        ),
    ],
)
def test_collision_generation_fails_closed(
    first: EnrichmentDonor,
    second: EnrichmentDonor,
    match: str,
) -> None:
    with pytest.raises(EnrichmentGenerationError, match=match):
        generate_collision_candidates(first, second, seed="dataset-v1", index=0)
