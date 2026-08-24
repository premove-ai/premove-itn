from __future__ import annotations

from dataclasses import dataclass

from premove_itn.labels import SpanKind


@dataclass(frozen=True, slots=True)
class SgdSpanPolicy:
    description: str
    kind: SpanKind


SGD_SPAN_POLICIES: dict[tuple[str, str], SgdSpanPolicy] = {
    ("Banks_1", "amount"): SgdSpanPolicy(
        "The amount of money to transfer", SpanKind.MONEY
    ),
    ("Buses_1", "leaving_date"): SgdSpanPolicy(
        "Date of bus leaving for journey", SpanKind.DATE
    ),
    ("Buses_1", "leaving_time"): SgdSpanPolicy(
        "Time of bus leaving for journey", SpanKind.TIME
    ),
    ("Buses_2", "departure_date"): SgdSpanPolicy(
        "Date of bus departure", SpanKind.DATE
    ),
    ("Buses_2", "departure_time"): SgdSpanPolicy(
        "Time of bus departure", SpanKind.TIME
    ),
    ("Calendar_1", "event_date"): SgdSpanPolicy(
        "Date of event or for checking availability", SpanKind.DATE
    ),
    ("Calendar_1", "event_time"): SgdSpanPolicy("Start time of event", SpanKind.TIME),
    ("Events_1", "date"): SgdSpanPolicy("Date of occurrence of event", SpanKind.DATE),
    ("Events_2", "date"): SgdSpanPolicy("Date of event", SpanKind.DATE),
    ("Flights_1", "departure_date"): SgdSpanPolicy(
        "Start date for the trip", SpanKind.DATE
    ),
    ("Flights_1", "inbound_departure_time"): SgdSpanPolicy(
        "Departure time for the return leg flight", SpanKind.TIME
    ),
    ("Flights_1", "outbound_departure_time"): SgdSpanPolicy(
        "Departure time for the outbound leg flight", SpanKind.TIME
    ),
    ("Flights_1", "return_date"): SgdSpanPolicy(
        "Date of the return flight", SpanKind.DATE
    ),
    ("Flights_2", "departure_date"): SgdSpanPolicy(
        "Date of departure flight on the ticket", SpanKind.DATE
    ),
    ("Flights_2", "return_date"): SgdSpanPolicy(
        "Date of return flight on the ticket", SpanKind.DATE
    ),
    ("Homes_1", "visit_date"): SgdSpanPolicy(
        "Date for the visit to the apartment", SpanKind.DATE
    ),
    ("Hotels_1", "check_in_date"): SgdSpanPolicy(
        "Start date for the reservation", SpanKind.DATE
    ),
    ("Hotels_1", "number_of_days"): SgdSpanPolicy(
        "Number of days in the reservation", SpanKind.CARDINAL
    ),
    ("Hotels_2", "check_in_date"): SgdSpanPolicy(
        "Start date for the reservation or to find the house", SpanKind.DATE
    ),
    ("Hotels_2", "check_out_date"): SgdSpanPolicy(
        "End date for the reservation or to find the house", SpanKind.DATE
    ),
    ("Hotels_2", "rating"): SgdSpanPolicy(
        "Review rating of the house", SpanKind.DECIMAL
    ),
    ("Hotels_3", "check_in_date"): SgdSpanPolicy(
        "Start date for the hotel reservation", SpanKind.DATE
    ),
    ("Hotels_3", "check_out_date"): SgdSpanPolicy(
        "End date for the hotel reservation", SpanKind.DATE
    ),
    ("Movies_1", "show_date"): SgdSpanPolicy("Date of the show", SpanKind.DATE),
    ("RentalCars_1", "dropoff_date"): SgdSpanPolicy(
        "Date of rental car drop-off", SpanKind.DATE
    ),
    ("RentalCars_1", "pickup_date"): SgdSpanPolicy(
        "Date of rental car pickup", SpanKind.DATE
    ),
    ("RentalCars_1", "pickup_time"): SgdSpanPolicy(
        "Time of rental car pickup", SpanKind.TIME
    ),
    ("RentalCars_2", "dropoff_date"): SgdSpanPolicy(
        "End date of car rental reservation", SpanKind.DATE
    ),
    ("RentalCars_2", "pickup_date"): SgdSpanPolicy(
        "Date of pickup for car rental", SpanKind.DATE
    ),
    ("RentalCars_2", "pickup_time"): SgdSpanPolicy(
        "Time of pickup for car rental", SpanKind.TIME
    ),
    ("Restaurants_1", "date"): SgdSpanPolicy(
        "Date for the reservation or to find availability", SpanKind.DATE
    ),
    ("Restaurants_1", "time"): SgdSpanPolicy(
        "Time for the reservation or to find availability", SpanKind.TIME
    ),
    ("Services_1", "appointment_date"): SgdSpanPolicy(
        "Date for the appointment", SpanKind.DATE
    ),
    ("Services_1", "appointment_time"): SgdSpanPolicy(
        "Time of the appointment", SpanKind.TIME
    ),
    ("Services_2", "appointment_date"): SgdSpanPolicy(
        "Date for the appointment", SpanKind.DATE
    ),
    ("Services_2", "appointment_time"): SgdSpanPolicy(
        "Time for the appointment", SpanKind.TIME
    ),
    ("Services_3", "appointment_date"): SgdSpanPolicy(
        "Date for scheduling the appointment with the doctor", SpanKind.DATE
    ),
    ("Services_3", "appointment_time"): SgdSpanPolicy(
        "Time for the appointment with the doctor", SpanKind.TIME
    ),
    ("Weather_1", "date"): SgdSpanPolicy("Date for the weather", SpanKind.DATE),
}

SGD_IGNORED_SLOTS = frozenset(
    {
        ("Banks_1", "recipient_account_name"),
        ("Buses_1", "from_location"),
        ("Buses_1", "to_location"),
        ("Buses_2", "destination"),
        ("Buses_2", "origin"),
        ("Calendar_1", "event_location"),
        ("Calendar_1", "event_name"),
        ("Events_1", "city_of_event"),
        ("Events_1", "event_name"),
        ("Events_1", "subcategory"),
        ("Events_2", "category"),
        ("Events_2", "city"),
        ("Events_2", "event_name"),
        ("Flights_1", "destination_city"),
        ("Flights_1", "origin_city"),
        ("Flights_2", "destination"),
        ("Flights_2", "origin"),
        ("Homes_1", "area"),
        ("Hotels_1", "destination"),
        ("Hotels_1", "hotel_name"),
        ("Hotels_2", "where_to"),
        ("Hotels_3", "hotel_name"),
        ("Hotels_3", "location"),
        ("Media_1", "directed_by"),
        ("Media_1", "genre"),
        ("Media_1", "title"),
        ("Movies_1", "genre"),
        ("Movies_1", "location"),
        ("Movies_1", "movie_name"),
        ("Movies_1", "theater_name"),
        ("Music_1", "album"),
        ("Music_1", "artist"),
        ("Music_1", "genre"),
        ("Music_1", "song_name"),
        ("Music_2", "album"),
        ("Music_2", "artist"),
        ("Music_2", "genre"),
        ("Music_2", "song_name"),
        ("RentalCars_1", "pickup_city"),
        ("RentalCars_1", "pickup_location"),
        ("RentalCars_2", "pickup_city"),
        ("RentalCars_2", "pickup_location"),
        ("Restaurants_1", "city"),
        ("Restaurants_1", "cuisine"),
        ("Restaurants_1", "restaurant_name"),
        ("RideSharing_1", "destination"),
        ("RideSharing_2", "destination"),
        ("Services_1", "city"),
        ("Services_1", "stylist_name"),
        ("Services_2", "city"),
        ("Services_2", "dentist_name"),
        ("Services_3", "city"),
        ("Services_3", "doctor_name"),
        ("Travel_1", "location"),
        ("Weather_1", "city"),
    }
)


def sgd_span_kind(service: str, slot: str, description: str) -> SpanKind | None:
    """Return the approved ITN kind for one exact SGD schema slot."""
    key = (service, slot)
    policy = SGD_SPAN_POLICIES.get(key)
    if policy is not None:
        if description != policy.description:
            raise ValueError(
                f"SGD schema description changed for {service}.{slot}: {description!r}"
            )
        return policy.kind
    if key in SGD_IGNORED_SLOTS:
        return None
    raise ValueError(f"unknown SGD slot: {service}.{slot}")
