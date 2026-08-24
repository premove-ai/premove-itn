from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from premove_itn.labels import SpanKind
from premove_itn.tokenize import tokenize
from premove_itn_data.enrichment.donors import EnrichmentDonor
from premove_itn_data.enrichment.sgd_context import SgdSlotSpan, SgdUserTurn
from premove_itn_data.enrichment.sgd_policy import sgd_span_kind


class EnrichmentGenerationError(ValueError):
    """Raised when trusted values cannot be placed into a context safely."""


@dataclass(frozen=True, slots=True)
class EnrichmentSpan:
    kind: SpanKind
    start: int
    end: int
    source: str
    replacement: str
    context_provenance: str
    donor_provenance: str


@dataclass(frozen=True, slots=True)
class EnrichmentCandidate:
    text: str
    context_provenance: str
    spans: tuple[EnrichmentSpan, ...]


@dataclass(frozen=True, slots=True)
class _PurposeBuiltTemplate:
    family: str
    prefix: str
    suffix: str


_PHONE_TEMPLATES = (
    _PurposeBuiltTemplate("phone_number", "My phone number is ", "."),
    _PurposeBuiltTemplate("phone_reach", "You can reach me at ", "."),
    _PurposeBuiltTemplate("phone_call", "Call me on ", "."),
    _PurposeBuiltTemplate("phone_contact", "The contact number is ", "."),
    _PurposeBuiltTemplate("phone_dial", "Please dial ", "."),
    _PurposeBuiltTemplate("phone_callback", "Please use ", " for the callback."),
    _PurposeBuiltTemplate("phone_arrival", "Call ", " when you arrive."),
    _PurposeBuiltTemplate("phone_best", "The best number to contact me on is ", "."),
    _PurposeBuiltTemplate("phone_reachable", "I can be reached at ", "."),
    _PurposeBuiltTemplate("phone_need", "Use ", " if you need to reach me."),
    _PurposeBuiltTemplate("phone_updated", "My updated number is ", "."),
    _PurposeBuiltTemplate("phone_mobile", "My mobile number is ", "."),
    _PurposeBuiltTemplate("phone_contact_me", "Contact me at ", "."),
    _PurposeBuiltTemplate("phone_return", "Return my call at ", "."),
    _PurposeBuiltTemplate("phone_leave", "You can leave a message at ", "."),
    _PurposeBuiltTemplate("phone_form", "Put ", " in the phone number field."),
    _PurposeBuiltTemplate("phone_booking", "Use ", " for the booking contact."),
    _PurposeBuiltTemplate("phone_account", "The number on my account is ", "."),
    _PurposeBuiltTemplate("phone_emergency", "For emergencies, call ", "."),
    _PurposeBuiltTemplate("phone_office", "The office can be reached on ", "."),
    _PurposeBuiltTemplate("phone_support", "Reach support by calling ", "."),
    _PurposeBuiltTemplate("phone_driver", "Give the driver my number, ", "."),
    _PurposeBuiltTemplate("phone_confirm", "Let me confirm the number: ", "."),
    _PurposeBuiltTemplate("phone_repeat", "I said my number was ", "."),
    _PurposeBuiltTemplate("phone_text", "Send a text to ", "."),
    _PurposeBuiltTemplate("phone_reservation", "Attach ", " to the reservation."),
    _PurposeBuiltTemplate("phone_record", "Please record my number as ", "."),
    _PurposeBuiltTemplate("phone_notification", "Send phone updates to ", "."),
    _PurposeBuiltTemplate("phone_after", "After the appointment, call ", "."),
    _PurposeBuiltTemplate("phone_question", "Can you call me back at ", "?"),
)
_ELECTRONIC_EMAIL_TEMPLATES = (
    _PurposeBuiltTemplate("email_address", "My email is ", "."),
    _PurposeBuiltTemplate("email_confirmation", "Send the confirmation to ", "."),
    _PurposeBuiltTemplate("email_contact", "You can email me at ", "."),
    _PurposeBuiltTemplate("email_receipt", "Email the receipt to ", "."),
    _PurposeBuiltTemplate("email_booking", "Use ", " for the booking."),
    _PurposeBuiltTemplate("email_account", "The email on my account is ", "."),
    _PurposeBuiltTemplate("email_updated", "My updated email address is ", "."),
    _PurposeBuiltTemplate("email_reply", "Please reply to ", "."),
    _PurposeBuiltTemplate("email_invoice", "Send the invoice by email to ", "."),
    _PurposeBuiltTemplate("email_ticket", "Forward the ticket to ", "."),
    _PurposeBuiltTemplate("email_form", "Enter ", " in the email field."),
    _PurposeBuiltTemplate("email_notify", "Send email notifications to ", "."),
    _PurposeBuiltTemplate("email_contact_record", "Record my email as ", "."),
    _PurposeBuiltTemplate("email_share", "Share the details with ", "."),
    _PurposeBuiltTemplate("email_team", "The team email is ", "."),
    _PurposeBuiltTemplate("email_support", "Contact support at ", "."),
    _PurposeBuiltTemplate("email_copy", "Copy ", " on the message."),
    _PurposeBuiltTemplate("email_reminder", "Send the reminder to ", "."),
    _PurposeBuiltTemplate("email_delivery", "Use ", " for delivery updates."),
    _PurposeBuiltTemplate("email_profile", "Add ", " to my profile."),
    _PurposeBuiltTemplate("email_confirm", "Let me confirm my email: ", "."),
    _PurposeBuiltTemplate("email_question", "Can you send it to ", "?"),
    _PurposeBuiltTemplate("email_followup", "Follow up with me at ", "."),
    _PurposeBuiltTemplate("email_documents", "Send the documents to ", "."),
    _PurposeBuiltTemplate("email_primary", "My primary email address is ", "."),
)
_ELECTRONIC_DOMAIN_TEMPLATES = (
    _PurposeBuiltTemplate("domain_website", "The website is ", "."),
    _PurposeBuiltTemplate("domain_visit", "Visit ", " for more information."),
    _PurposeBuiltTemplate("domain_location", "You can find it at ", "."),
    _PurposeBuiltTemplate("domain_booking", "Complete the booking at ", "."),
    _PurposeBuiltTemplate("domain_details", "The details are available on ", "."),
    _PurposeBuiltTemplate("domain_support", "Open the support page at ", "."),
    _PurposeBuiltTemplate("domain_account", "Sign in through ", "."),
    _PurposeBuiltTemplate("domain_form", "Enter ", " as the website."),
    _PurposeBuiltTemplate("domain_company", "Their company site is ", "."),
    _PurposeBuiltTemplate("domain_order", "Track the order on ", "."),
    _PurposeBuiltTemplate("domain_menu", "The menu is posted at ", "."),
    _PurposeBuiltTemplate("domain_schedule", "Check the schedule on ", "."),
    _PurposeBuiltTemplate("domain_application", "Submit the application through ", "."),
    _PurposeBuiltTemplate("domain_reference", "The page I used was ", "."),
    _PurposeBuiltTemplate("domain_question", "Can you open ", "?"),
    _PurposeBuiltTemplate("domain_profile", "My profile is hosted at ", "."),
    _PurposeBuiltTemplate("domain_payment", "Make the payment through ", "."),
    _PurposeBuiltTemplate("domain_portal", "Use the customer portal at ", "."),
)
_DIGIT_SEQUENCE_TEMPLATES = (
    _PurposeBuiltTemplate("digit_verification", "My verification code is ", "."),
    _PurposeBuiltTemplate("digit_order", "The order number is ", "."),
    _PurposeBuiltTemplate("digit_reference", "My reference number is ", "."),
    _PurposeBuiltTemplate("digit_entry", "Please enter code ", "."),
    _PurposeBuiltTemplate("digit_confirmation", "The confirmation code is ", "."),
    _PurposeBuiltTemplate("digit_ticket", "My ticket number is ", "."),
    _PurposeBuiltTemplate("digit_case", "The case number is ", "."),
    _PurposeBuiltTemplate("digit_tracking", "The tracking code is ", "."),
    _PurposeBuiltTemplate("digit_last", "The last digits are ", "."),
    _PurposeBuiltTemplate("digit_pin", "Use ", " as the access code."),
    _PurposeBuiltTemplate("digit_security", "My security code is ", "."),
    _PurposeBuiltTemplate("digit_booking", "The booking reference is ", "."),
    _PurposeBuiltTemplate("digit_account", "My account reference is ", "."),
    _PurposeBuiltTemplate("digit_delivery", "The delivery code is ", "."),
    _PurposeBuiltTemplate("digit_unlock", "Enter ", " to unlock it."),
    _PurposeBuiltTemplate("digit_repeat", "Let me repeat the code: ", "."),
    _PurposeBuiltTemplate("digit_readback", "Read back ", " to confirm."),
    _PurposeBuiltTemplate("digit_agent", "Give the agent reference ", "."),
    _PurposeBuiltTemplate("digit_form", "Put ", " in the code field."),
    _PurposeBuiltTemplate("digit_message", "The message includes code ", "."),
    _PurposeBuiltTemplate("digit_lookup", "Look up reservation ", "."),
    _PurposeBuiltTemplate("digit_claim", "My claim number is ", "."),
    _PurposeBuiltTemplate("digit_return", "The return authorization is ", "."),
    _PurposeBuiltTemplate("digit_queue", "My queue number is ", "."),
    _PurposeBuiltTemplate("digit_device", "The device code shown is ", "."),
    _PurposeBuiltTemplate("digit_payment", "The payment reference ends in ", "."),
    _PurposeBuiltTemplate("digit_transfer", "Use reference ", " for the transfer."),
    _PurposeBuiltTemplate("digit_question", "Is the confirmation number ", "?"),
    _PurposeBuiltTemplate("digit_label", "The label shows ", "."),
    _PurposeBuiltTemplate("digit_support", "Support gave me code ", "."),
)
_TIME_COLLISION_TEMPLATES = (
    _PurposeBuiltTemplate("time_arrival", "I will arrive at ", "."),
    _PurposeBuiltTemplate("time_reservation", "Book it for ", "."),
    _PurposeBuiltTemplate("time_meeting", "The meeting starts at ", "."),
)
_PHONE_COLLISION_TEMPLATES = (
    _PurposeBuiltTemplate("phone_call", "Call ", "."),
    _PurposeBuiltTemplate("phone_dial", "Dial ", "."),
    _PurposeBuiltTemplate("phone_reach", "You can reach me at ", "."),
    _PurposeBuiltTemplate("phone_number", "My phone number is ", "."),
    _PurposeBuiltTemplate("phone_need", "Use ", " if you need to call me."),
    _PurposeBuiltTemplate("phone_callback", "Please call me back at ", "."),
    _PurposeBuiltTemplate("phone_contact", "The contact number is ", "."),
    _PurposeBuiltTemplate("phone_text", "Send a text to ", "."),
)
_DIGIT_SEQUENCE_COLLISION_TEMPLATES = (
    _PurposeBuiltTemplate("digit_verification", "My verification code is ", "."),
    _PurposeBuiltTemplate("digit_reference", "The reference code is ", "."),
    _PurposeBuiltTemplate("digit_entry", "Please enter code ", "."),
)


def _require_seed(seed: str) -> None:
    if not seed:
        raise ValueError("seed must be non-empty")


def _deterministic_index(
    namespace: str,
    seed: str,
    identity: tuple[str, ...],
    size: int,
) -> int:
    payload = "\0".join((namespace, seed, *identity))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % size


def _canonical_donor_pool(
    kind: SpanKind,
    donors_by_kind: Mapping[SpanKind, Sequence[EnrichmentDonor]],
) -> tuple[EnrichmentDonor, ...]:
    raw_donors = donors_by_kind.get(kind, ())
    winners: dict[tuple[str, str], EnrichmentDonor] = {}
    for donor in raw_donors:
        if donor.kind is not kind:
            raise EnrichmentGenerationError(
                f"{kind.value} donor pool contains {donor.kind.value} donor"
            )
        if not donor.spoken or not donor.replacement or not donor.provenance:
            raise EnrichmentGenerationError(
                f"{kind.value} donor fields must be non-empty"
            )
        key = (donor.spoken, donor.replacement)
        current = winners.get(key)
        if current is None or donor.provenance < current.provenance:
            winners[key] = donor

    donors = tuple(
        winners[key] for key in sorted(winners, key=lambda value: (value[0], value[1]))
    )
    if not donors:
        raise EnrichmentGenerationError(f"no {kind.value} donors are available")
    return donors


def _approved_sgd_spans(
    turn: SgdUserTurn,
) -> tuple[tuple[SgdSlotSpan, SpanKind], ...]:
    by_offsets: dict[tuple[int, int], tuple[SgdSlotSpan, SpanKind]] = {}
    for span in turn.spans:
        if (
            span.start < 0
            or span.end <= span.start
            or span.end > len(turn.utterance)
            or turn.utterance[span.start : span.end] != span.value
        ):
            raise EnrichmentGenerationError(
                f"invalid SGD span {span.service}.{span.slot} "
                f"at [{span.start}, {span.end})"
            )
        try:
            kind = sgd_span_kind(span.service, span.slot, span.description)
        except ValueError as error:
            raise EnrichmentGenerationError(
                f"SGD policy rejected {span.service}.{span.slot}"
            ) from error
        if kind is None:
            continue

        offsets = (span.start, span.end)
        current = by_offsets.get(offsets)
        if current is not None and current[1] is not kind:
            raise EnrichmentGenerationError(
                "one SGD source span maps to conflicting SpanKind values"
            )
        if current is None or (span.service, span.slot) < (
            current[0].service,
            current[0].slot,
        ):
            by_offsets[offsets] = (span, kind)

    approved = tuple(
        by_offsets[offsets]
        for offsets in sorted(by_offsets, key=lambda value: (value[0], value[1]))
    )
    for (previous, _), (current, _) in zip(approved, approved[1:], strict=False):
        if previous.end > current.start:
            raise EnrichmentGenerationError("overlapping approved SGD spans")
    return approved


def _sgd_context_provenance(turn: SgdUserTurn) -> str:
    return f"sgd/{turn.source_file}/{turn.dialogue_id}/{turn.turn_index}"


def _spans_align_to_token_boundaries(
    turn: SgdUserTurn,
    approved: tuple[tuple[SgdSlotSpan, SpanKind], ...],
) -> bool:
    tokens = tokenize(turn.utterance)
    for span, _ in approved:
        covered = tuple(
            token
            for token in tokens
            if token.start >= span.start and token.end <= span.end
        )
        if (
            not covered
            or covered[0].start != span.start
            or covered[-1].end != span.end
        ):
            return False
    return True


def generate_sgd_candidate(
    turn: SgdUserTurn,
    donors_by_kind: Mapping[SpanKind, Sequence[EnrichmentDonor]],
    *,
    seed: str,
) -> EnrichmentCandidate | None:
    """Replace every approved SGD slot in one USER turn with a trusted donor."""
    _require_seed(seed)
    approved = _approved_sgd_spans(turn)
    if not approved or not _spans_align_to_token_boundaries(turn, approved):
        return None

    pools: dict[SpanKind, tuple[EnrichmentDonor, ...]] = {}
    output_parts: list[str] = []
    output_length = 0
    source_cursor = 0
    generated_spans: list[EnrichmentSpan] = []
    for source_span, kind in approved:
        prefix = turn.utterance[source_cursor : source_span.start]
        output_parts.append(prefix)
        output_length += len(prefix)

        pool = pools.get(kind)
        if pool is None:
            pool = _canonical_donor_pool(kind, donors_by_kind)
            pools[kind] = pool
        context_identity = (
            turn.source_file,
            turn.dialogue_id,
            str(turn.turn_index),
            source_span.service,
            source_span.slot,
            str(source_span.start),
            str(source_span.end),
        )
        donor = pool[
            _deterministic_index(
                "sgd-enrichment-v1",
                seed,
                context_identity,
                len(pool),
            )
        ]
        generated_start = output_length
        generated_end = generated_start + len(donor.spoken)
        output_parts.append(donor.spoken)
        output_length = generated_end
        generated_spans.append(
            EnrichmentSpan(
                kind=kind,
                start=generated_start,
                end=generated_end,
                source=donor.spoken,
                replacement=donor.replacement,
                context_provenance=(
                    f"sgd_slot/{source_span.service}/{source_span.slot}/"
                    f"{source_span.start}/{source_span.end}"
                ),
                donor_provenance=donor.provenance,
            )
        )
        source_cursor = source_span.end

    output_parts.append(turn.utterance[source_cursor:])
    return EnrichmentCandidate(
        text="".join(output_parts),
        context_provenance=_sgd_context_provenance(turn),
        spans=tuple(generated_spans),
    )


def generate_sgd_context_only_candidate(
    turn: SgdUserTurn,
) -> EnrichmentCandidate | None:
    """Keep one SGD USER turn only when it has no approved ITN span."""
    if _approved_sgd_spans(turn):
        return None
    return EnrichmentCandidate(
        text=turn.utterance,
        context_provenance=_sgd_context_provenance(turn),
        spans=(),
    )


def _purpose_built_templates(
    donor: EnrichmentDonor,
) -> tuple[_PurposeBuiltTemplate, ...]:
    if donor.kind is SpanKind.PHONE:
        return _PHONE_TEMPLATES
    if donor.kind is SpanKind.ELECTRONIC:
        if "@" in donor.replacement:
            return _ELECTRONIC_EMAIL_TEMPLATES
        return _ELECTRONIC_DOMAIN_TEMPLATES
    if donor.kind is SpanKind.DIGIT_SEQUENCE:
        return _DIGIT_SEQUENCE_TEMPLATES
    raise EnrichmentGenerationError(
        f"unsupported purpose-built kind: {donor.kind.value}"
    )


def generate_purpose_built_candidate(
    donor: EnrichmentDonor,
    *,
    seed: str,
    index: int,
) -> EnrichmentCandidate:
    """Place one trusted donor in a deterministic class-specific context."""
    _require_seed(seed)
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("index must be a non-negative integer")
    if not donor.spoken or not donor.replacement or not donor.provenance:
        raise EnrichmentGenerationError("donor fields must be non-empty")

    templates = _purpose_built_templates(donor)
    template = templates[
        _deterministic_index(
            "purpose-built-enrichment-v1",
            seed,
            (
                str(index),
                donor.kind.value,
                donor.spoken,
                donor.replacement,
                donor.provenance,
            ),
            len(templates),
        )
    ]
    context_provenance = (
        f"purpose_built/{donor.kind.value.lower()}/{template.family}/{index:06d}"
    )
    return _candidate_from_template(
        donor,
        template,
        context_provenance=context_provenance,
    )


def _candidate_from_template(
    donor: EnrichmentDonor,
    template: _PurposeBuiltTemplate,
    *,
    context_provenance: str,
) -> EnrichmentCandidate:
    start = len(template.prefix)
    end = start + len(donor.spoken)
    return EnrichmentCandidate(
        text=f"{template.prefix}{donor.spoken}{template.suffix}",
        context_provenance=context_provenance,
        spans=(
            EnrichmentSpan(
                kind=donor.kind,
                start=start,
                end=end,
                source=donor.spoken,
                replacement=donor.replacement,
                context_provenance=context_provenance,
                donor_provenance=donor.provenance,
            ),
        ),
    )


def generate_collision_candidates(
    first: EnrichmentDonor,
    second: EnrichmentDonor,
    *,
    seed: str,
    index: int,
) -> tuple[EnrichmentCandidate, EnrichmentCandidate]:
    """Generate an approved contextual collision pair for one spoken form."""
    _require_seed(seed)
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("index must be a non-negative integer")
    if first.spoken != second.spoken:
        raise EnrichmentGenerationError(
            "collision donors must have the same spoken form"
        )
    donors = {first.kind: first, second.kind: second}
    kinds = set(donors)
    if kinds == {SpanKind.TIME, SpanKind.DIGIT_SEQUENCE}:
        contextual_kind = SpanKind.TIME
        contextual_templates = _TIME_COLLISION_TEMPLATES
    elif kinds == {SpanKind.PHONE, SpanKind.DIGIT_SEQUENCE}:
        contextual_kind = SpanKind.PHONE
        contextual_templates = _PHONE_COLLISION_TEMPLATES
    else:
        raise EnrichmentGenerationError(
            "unsupported collision; expected TIME or PHONE paired with "
            "DIGIT_SEQUENCE"
        )

    identity = (
        str(index),
        first.spoken,
        donors[contextual_kind].provenance,
        donors[SpanKind.DIGIT_SEQUENCE].provenance,
    )
    contextual_template = contextual_templates[
        _deterministic_index(
            f"collision-{contextual_kind.value.lower()}-v1",
            seed,
            identity,
            len(contextual_templates),
        )
    ]
    digit_template = _DIGIT_SEQUENCE_COLLISION_TEMPLATES[
        _deterministic_index(
            "collision-digit-sequence-v1",
            seed,
            identity,
            len(_DIGIT_SEQUENCE_COLLISION_TEMPLATES),
        )
    ]
    collision_identity = hashlib.sha256(
        "\0".join((seed, *identity)).encode("utf-8")
    ).hexdigest()[:16]
    return (
        _candidate_from_template(
            donors[contextual_kind],
            contextual_template,
            context_provenance=(
                f"purpose_built/collision/{collision_identity}/"
                f"{contextual_kind.value.lower()}/{index:06d}"
            ),
        ),
        _candidate_from_template(
            donors[SpanKind.DIGIT_SEQUENCE],
            digit_template,
            context_provenance=(
                f"purpose_built/collision/{collision_identity}/"
                f"digit_sequence/{index:06d}"
            ),
        ),
    )
