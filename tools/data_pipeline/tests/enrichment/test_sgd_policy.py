import pytest
from premove_itn_data.enrichment.sgd_policy import (
    SGD_IGNORED_SLOTS,
    SGD_SPAN_POLICIES,
    sgd_span_kind,
)

from premove_itn.labels import SpanKind


def test_policy_covers_the_audited_sgd_train_inventory() -> None:
    assert len(SGD_SPAN_POLICIES) == 39
    assert len(SGD_IGNORED_SLOTS) == 55
    assert not (SGD_SPAN_POLICIES.keys() & SGD_IGNORED_SLOTS)
    assert len(SGD_SPAN_POLICIES.keys() | SGD_IGNORED_SLOTS) == 94


@pytest.mark.parametrize(
    ("service", "slot", "description", "kind"),
    [
        (
            "Restaurants_1",
            "time",
            "Time for the reservation or to find availability",
            SpanKind.TIME,
        ),
        (
            "Flights_1",
            "departure_date",
            "Start date for the trip",
            SpanKind.DATE,
        ),
        (
            "Banks_1",
            "amount",
            "The amount of money to transfer",
            SpanKind.MONEY,
        ),
        (
            "Hotels_1",
            "number_of_days",
            "Number of days in the reservation",
            SpanKind.CARDINAL,
        ),
        (
            "Hotels_2",
            "rating",
            "Review rating of the house",
            SpanKind.DECIMAL,
        ),
    ],
)
def test_maps_only_exact_approved_semantics(
    service: str, slot: str, description: str, kind: SpanKind
) -> None:
    assert sgd_span_kind(service, slot, description) is kind


def test_lexical_slot_is_explicitly_ignored() -> None:
    assert sgd_span_kind("Movies_1", "movie_name", "Name of the movie") is None


def test_approved_description_drift_fails_closed() -> None:
    with pytest.raises(ValueError, match="schema description changed"):
        sgd_span_kind("Restaurants_1", "time", "An unrelated number")


def test_unknown_pair_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown SGD slot"):
        sgd_span_kind("Restaurants_1", "confirmation_number", "Booking reference")


def test_no_phone_electronic_or_digit_sequence_is_inferred() -> None:
    kinds = {policy.kind for policy in SGD_SPAN_POLICIES.values()}

    assert SpanKind.PHONE not in kinds
    assert SpanKind.ELECTRONIC not in kinds
    assert SpanKind.DIGIT_SEQUENCE not in kinds
