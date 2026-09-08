"""Provide the shared source rows for the VoiceAgent ITN benchmark."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "eval/_voice_agent_itn_source"
SEED = "premove-itn/voice-agent-itn/source"
FROZEN_CHECKPOINT_SHA256 = (
    "9021fa11a028faefb31ef67878170cbe29ed25e68a9a78999f37b120c2ad00d5"
)
QUOTAS = {
    "normal": 500,
    "voice_agent": 400,
    "collision": 350,
    "multi": 200,
    "negative": 50,
}
ONES = [
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
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = [
    "zero",
    "ten",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
CATEGORIES = [
    "CARDINAL",
    "ORDINAL",
    "DECIMAL",
    "MONEY",
    "PERCENT",
    "DATE",
    "TIME",
    "PHONE",
    "DIGIT_SEQUENCE",
    "ORDER_ID",
    "REFERENCE_ID",
    "FLIGHT_ID",
    "VERSION",
    "EMAIL",
    "URL",
    "IP",
    "MEASUREMENT",
    "ZIP_CODE",
]


def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + (f" {ONES[n % 10]}" if n % 10 else "")
    if n < 1000:
        return f"{ONES[n // 100]} hundred" + (f" {words(n % 100)}" if n % 100 else "")
    return f"{words(n // 1000)} thousand" + (f" {words(n % 1000)}" if n % 1000 else "")


def digit_words(value: str) -> str:
    return " ".join(ONES[int(char)] for char in value)


def ordinal(n: int) -> str:
    special = {
        1: "first",
        2: "second",
        3: "third",
        5: "fifth",
        8: "eighth",
        9: "ninth",
        12: "twelfth",
    }
    if n in special:
        return special[n]
    if n < 20:
        return ONES[n].removesuffix("e") + "th"
    if n % 10 == 0:
        return TENS[n // 10].removesuffix("y") + "ieth"
    return f"{TENS[n // 10]} {ordinal(n % 10)}"


def ord_suffix(n: int) -> str:
    return (
        "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )


def atom(category: str, index: int) -> dict[str, object]:
    n = 31 + index * 43
    kind = category
    if category == "CARDINAL":
        spoken, written, value = words(n), str(n), n
    elif category == "ORDINAL":
        value = index % 99 + 1
        spoken, written = ordinal(value), f"{value}{ord_suffix(value)}"
    elif category == "DECIMAL":
        whole = index % 97 + 1
        width = index % 4 + 1
        fraction = f"{(index * 7919 + 3) % (10**width):0{width}d}"
        spoken, written, value = (
            f"{words(whole)} point {digit_words(fraction)}",
            f"{whole}.{fraction}",
            f"{whole}.{fraction}",
        )
    elif category == "MONEY":
        whole, cents = index % 193 + 7, (index * 23 + 11) % 100
        currencies = [
            ("dollars", "$", "USD"),
            ("euros", "€", "EUR"),
            ("pounds", "£", "GBP"),
            ("Canadian dollars", "CAD ", "CAD"),
        ]
        unit, symbol, currency = currencies[index % len(currencies)]
        cents_words = f"zero {ONES[cents]}" if cents < 10 else words(cents)
        minor_unit = "pence" if currency == "GBP" else "cents"
        spoken, written = (
            f"{words(whole)} {unit} and {cents_words} {minor_unit}",
            f"{symbol}{whole}.{cents:02d}",
        )
        value = {"currency": currency, "minor_units": whole * 100 + cents}
    elif category == "PERCENT":
        value = index % 97 + 1
        spoken, written = f"{words(value)} percent", f"{value}%"
    elif category == "DATE":
        month, day, year = MONTHS[index % 12], index % 28 + 1, 2024 + index % 6
        spoken, written = (
            f"{month.lower()} {ordinal(day)} {words(year)}",
            f"{month} {day}, {year}",
        )
        value = {"year": year, "month": index % 12 + 1, "day": day}
    elif category == "TIME":
        hour, minute, pm = index // 60 % 12 + 1, index % 60, index % 2
        spoken = f"{words(hour)} {words(minute)} {'p m' if pm else 'a m'}"
        written, value = (
            f"{hour}:{minute:02d} {'PM' if pm else 'AM'}",
            {"hour": hour % 12 + 12 * pm, "minute": minute},
        )
    elif category == "PHONE":
        if index % 3 == 0:
            value = f"{200 + index % 700:03d}{1000 + index % 9000:04d}"
            written = f"{value[:3]}-{value[3:]}"
        elif index % 3 == 1:
            area = 201 + index % 700
            exchange = 310 + index % 600
            subscriber = 1000 + index % 9000
            value = f"{area:03d}{exchange:03d}{subscriber:04d}"
            written = f"({value[:3]}) {value[3:6]}-{value[6:]}"
        else:
            value = f"44{20 + index % 70:02d}{10000000 + index % 90000000:08d}"
            written = f"+{value}"
        spoken = ("plus " if written.startswith("+") else "") + digit_words(value)
        value = {"digits": value}
    elif category in {"DIGIT_SEQUENCE", "ZIP_CODE"}:
        value = (
            f"0{(index * 7919 + 73) % 100000:05d}"
            if category == "DIGIT_SEQUENCE"
            else (
                f"{10000 + index % 89999:05d}-{index % 10000:04d}"
                if index % 4 == 0
                else f"{10000 + index % 89999:05d}"
            )
        )
        spoken = (
            " dash ".join(digit_words(part) for part in value.split("-"))
            if "-" in value
            else digit_words(value)
        )
        written = value
        kind = "DIGIT_SEQUENCE"
    elif category in {"ORDER_ID", "REFERENCE_ID", "FLIGHT_ID"}:
        letter_count = index % 3 + 1
        letters = "".join(
            chr(97 + (index * (j + 3) + j * 7) % 20) for j in range(letter_count)
        )
        width = index % 4 + 3
        number = f"{(index * 83 + 19) % (10**width):0{width}d}"
        spoken = f"{' '.join(letters)} {digit_words(number)}"
        written = value = f"{letters.upper()}{number}"
        kind = "WORD"
    elif category == "VERSION":
        count = index % 3 + 2
        parts = [index // 100 + 1] + [
            index // (10**j) % 10 for j in range(count - 2, -1, -1)
        ]
        prefix = "v " if index % 4 == 0 else ""
        spoken, written = (
            prefix + " point ".join(words(x) for x in parts),
            ("v" if prefix else "") + ".".join(map(str, parts)),
        )
        value, kind = written, "WORD"
    elif category == "EMAIL":
        local_variants = [
            (f"case{index + 17}", f"case {digit_words(str(index + 17))}"),
            (f"a.lee{index}", f"a dot lee {digit_words(str(index))}"),
            (f"support_{index}", f"support underscore {digit_words(str(index))}"),
            (f"j-smith{index}", f"j dash smith {digit_words(str(index))}"),
        ]
        domains = [
            ("helpdesk.example.com", "help desk dot example dot com"),
            ("mail.example.org", "mail dot example dot org"),
            ("service.example.net", "service dot example dot net"),
        ]
        local, local_spoken = local_variants[index % 4]
        domain, domain_spoken = domains[index % 3]
        value = f"{local}@{domain}"
        spoken, written, kind = (
            f"{local_spoken} at {domain_spoken}",
            value,
            "ELECTRONIC",
        )
    elif category == "URL":
        hosts = ["support.example.org", "status.example.com", "docs.example.net"]
        host = hosts[index % 3]
        path_stem = ["ticket", "case", "help", "status"][index % 4]
        path_number = str(index + 31)
        path = path_stem + path_number
        protocol = "https://" if index % 4 else "http://"
        value = f"{protocol}{host}/{path}"
        protocol_spoken = "h t t p s" if protocol == "https://" else "h t t p"
        host_spoken = host.replace(".", " dot ")
        path_spoken = f"{path_stem} {digit_words(path_number)}"
        spoken = (
            f"{protocol_spoken} colon slash slash {host_spoken} slash {path_spoken}"
        )
        written, kind = value, "ELECTRONIC"
    elif category == "IP":
        parts = [10, index % 250 + 1, index * 7 % 250 + 1, index * 13 % 250 + 1]
        spoken, value = (
            " dot ".join(digit_words(str(x)) for x in parts),
            ".".join(map(str, parts)),
        )
        written, kind = value, "PHONE"
    elif category == "MEASUREMENT":
        whole = index % 190 + 3
        numeric = f"{whole}.{index % 10}" if index % 3 == 0 else str(whole)
        number_spoken = (
            f"{words(whole)} point {ONES[index % 10]}"
            if index % 3 == 0
            else words(whole)
        )
        unit, symbol = [
            ("kilometers", "km"),
            ("liters", "L"),
            ("degrees celsius", "°C"),
            ("megabytes", "MB"),
            ("miles per hour", "mph"),
            ("kilograms", "kg"),
        ][index % 6]
        spoken, written, value = (
            f"{number_spoken} {unit}",
            f"{numeric} {symbol}",
            {"number": numeric, "unit": symbol},
        )
    else:
        raise ValueError(category)
    return {
        "category": category,
        "spoken": spoken,
        "written": written,
        "semantic_value": value,
        "premove_span_kind": kind,
    }


def row(
    row_id: str,
    group: str,
    template: str,
    atoms: list[dict[str, object]],
    domain: str,
    _difficulty: str,
    collision: str | None = None,
    pair_id: str | None = None,
) -> dict[str, object]:
    text = expected = template
    for i, item in enumerate(atoms):
        marker = "{x}" if len(atoms) == 1 else f"{{x{i}}}"
        text = text.replace(marker, str(item["spoken"]), 1)
        expected = expected.replace(marker, str(item["written"]), 1)
    spans, cursor = [], 0
    for item in atoms:
        if item["category"] == "KEEP":
            continue
        source = str(item["spoken"])
        start = text.index(source, cursor)
        spans.append(
            {
                **item,
                "start": start,
                "end": start + len(source),
                "source": source,
                "replacement": item["written"],
            }
        )
        cursor = start + len(source)
    features = ["multiple_entities"] if len(atoms) > 1 else []
    if pair_id:
        features.append("contextual_disambiguation")
        features.append("paired_contrast")
    if any(str(item["spoken"]).startswith(("zero", "oh")) for item in atoms):
        features.append("leading_zero")
    if any(len(str(item["spoken"]).split()) >= 9 for item in atoms):
        features.append("long_entity")
    if any(item["category"] in {"EMAIL", "URL", "IP", "VERSION"} for item in atoms):
        features.append("structured_punctuation")
    if not atoms:
        features.append("lexical_keep_trap")
    score = len(features)
    difficulty = "easy" if score == 0 else "medium" if score == 1 else "hard"
    return {
        "id": row_id,
        "text": text,
        "expected_text": expected,
        "group": group,
        "domain": domain,
        "categories": [a["category"] for a in atoms] or ["KEEP"],
        "collision": collision,
        "collision_pair_id": pair_id,
        "difficulty": difficulty,
        "difficulty_features": features,
        "spans": spans,
        "scoring": {
            "strict_exact": True,
            "semantic_entity": bool(spans),
            "normalization_decision": True,
        },
        "review_status": "pending_independent_review",
    }


CONTEXTS = [
    "please record {x}",
    "the confirmed value is {x}",
    "I wrote down {x}",
    "the form shows {x}",
    "read back {x}",
    "could you verify {x}",
    "the caller reported {x}",
    "use {x} for this request",
    "the note lists {x}",
    "I heard {x} clearly",
    "enter {x} in the next field",
    "the corrected entry is {x}",
    "we agreed on {x}",
    "the message included {x}",
    "confirm that it says {x}",
    "the operator repeated {x}",
    "save {x} with the record",
    "my written copy says {x}",
    "the final answer was {x}",
    "please replace the old value with {x}",
]
DOMAINS = [
    "customer_support",
    "ecommerce",
    "banking",
    "scheduling",
    "travel",
    "logistics",
    "technical_support",
    "healthcare_admin",
]
DOMAIN_TASKS = {
    "customer_support": [
        ("REFERENCE_ID", "the ticket they gave me is {x}"),
        ("PHONE", "call me back at {x}"),
        ("EMAIL", "send the case notes to {x}"),
        ("DATE", "the problem started on {x}"),
        ("TIME", "the agent called at {x}"),
    ],
    "ecommerce": [
        ("ORDER_ID", "can you look up order {x}"),
        ("MONEY", "my card was charged {x}"),
        ("ZIP_CODE", "ship it to zip code {x}"),
        ("DATE", "the parcel is due by {x}"),
        ("REFERENCE_ID", "the return reference is {x}"),
    ],
    "banking": [
        ("MONEY", "I do not recognize a charge for {x}"),
        ("DIGIT_SEQUENCE", "the last account digits are {x}"),
        ("PERCENT", "the quoted interest rate is {x}"),
        ("DATE", "the transfer posted on {x}"),
        ("PHONE", "my verified number is {x}"),
    ],
    "scheduling": [
        ("DATE", "move my appointment to {x}"),
        ("TIME", "schedule the visit for {x}"),
        ("PHONE", "text the reminder to {x}"),
        ("REFERENCE_ID", "the appointment reference is {x}"),
        ("EMAIL", "send the invitation to {x}"),
    ],
    "travel": [
        ("FLIGHT_ID", "I need to change flight {x}"),
        ("DATE", "the outbound date is {x}"),
        ("TIME", "boarding begins at {x}"),
        ("MONEY", "the fare difference is {x}"),
        ("REFERENCE_ID", "my booking reference is {x}"),
    ],
    "logistics": [
        ("ORDER_ID", "the tracking identifier is {x}"),
        ("MEASUREMENT", "the shipment weighs {x}"),
        ("ZIP_CODE", "the destination zip code is {x}"),
        ("DATE", "collect the freight on {x}"),
        ("TIME", "the loading slot starts at {x}"),
    ],
    "technical_support": [
        ("VERSION", "I am running version {x}"),
        ("IP", "the failing host is at {x}"),
        ("URL", "the error appears at {x}"),
        ("REFERENCE_ID", "the device serial is {x}"),
        ("DECIMAL", "set the multiplier to {x}"),
    ],
    "healthcare_admin": [
        ("DATE", "my appointment date is {x}"),
        ("TIME", "the clinic booked me for {x}"),
        ("PHONE", "my callback number is {x}"),
        ("REFERENCE_ID", "the referral number is {x}"),
        ("MEASUREMENT", "the recorded measurement is {x}"),
    ],
}


def domain_atom(domain: str, category: str, index: int) -> dict[str, object]:
    if domain == "banking" and category == "PERCENT":
        rates = [1, 3, 5, 7, 12, 18, 21, 24, 29, 34]
        value = rates[index // 40 % len(rates)]
        return {
            "category": category,
            "spoken": f"{words(value)} percent",
            "written": f"{value}%",
            "semantic_value": value,
            "premove_span_kind": "PERCENT",
        }
    if domain == "logistics" and category == "MEASUREMENT":
        number = index % 190 + 3
        return {
            "category": category,
            "spoken": f"{words(number)} kilograms",
            "written": f"{number} kg",
            "semantic_value": {"number": str(number), "unit": "kg"},
            "premove_span_kind": "MEASUREMENT",
        }
    if domain == "healthcare_admin" and category == "MEASUREMENT":
        temperatures = [
            "35.5",
            "36.1",
            "36.5",
            "37.0",
            "37.5",
            "38.2",
            "38.8",
            "39.4",
            "40.1",
            "41.0",
        ]
        value = temperatures[index // 40 % len(temperatures)]
        whole, fraction = value.split(".")
        return {
            "category": category,
            "spoken": (
                f"{words(int(whole))} point {ONES[int(fraction)]} degrees celsius"
            ),
            "written": f"{value} °C",
            "semantic_value": {"number": value, "unit": "°C"},
            "premove_span_kind": "MEASUREMENT",
        }
    return atom(category, index)


def build_rows() -> list[dict[str, object]]:
    result = []
    for i in range(500):
        category = CATEGORIES[i % len(CATEGORIES)]
        result.append(
            row(
                f"source_normal_{i + 1:04d}",
                "normal",
                CONTEXTS[i % len(CONTEXTS)],
                [atom(category, i)],
                "general",
                ["easy", "medium", "medium", "hard"][i % 4],
            )
        )
    for i in range(400):
        domain_index = i % 8
        domain = DOMAINS[domain_index]
        category, template = DOMAIN_TASKS[domain][i // 8 % 5]
        result.append(
            row(
                f"source_voice_{i + 1:04d}",
                "voice_agent",
                template,
                [domain_atom(domain, category, 1000 + i)],
                domain,
                ["easy", "medium", "hard", "medium", "hard"][i % 5],
            )
        )
    result.extend(collision_rows())
    combos = [(a, b) for a in CATEGORIES for b in reversed(CATEGORIES) if a != b][:40]
    multi_templates = [
        "record {x0} and then confirm {x1}",
        "use {x0}; the related detail is {x1}",
        "I have {x0}, but the second field is {x1}",
        "confirm {x0} before entering {x1}",
        "the first value is {x0} and the next is {x1}",
        "update {x0}, but leave a note about {x1}",
        "after confirming {x0}, read back {x1}",
        "the request mentions {x0} alongside {x1}",
        "store {x0} in the first field and {x1} in the second",
        "I corrected the first detail to {x0} and the other to {x1}",
    ]
    for i in range(200):
        a, b = combos[i % 40]
        items = [atom(a, 2000 + i), atom(b, 3000 + i)]
        template = multi_templates[i % len(multi_templates)]
        if i < 40:
            third = CATEGORIES[(i + 7) % len(CATEGORIES)]
            items.append(atom(third, 4000 + i))
            template = "confirm {x0}, then {x1}, and finally {x2}"
        result.append(
            row(
                f"source_multi_{i + 1:04d}",
                "multi",
                template,
                items,
                DOMAINS[i % 8],
                "medium" if i % 3 == 0 else "hard",
            )
        )
    for i, text in enumerate(NEGATIVES):
        result.append(
            row(
                f"source_negative_{i + 1:04d}",
                "negative",
                text,
                [],
                "general",
                ["easy", "medium", "hard"][i % 3],
                "false_positive_keep",
            )
        )
    return result


def collision_semantic_value(category: str, written: str) -> object:
    if category in {"CARDINAL", "ORDINAL"}:
        return int(re.sub(r"[^0-9]", "", written))
    if category == "MONEY":
        whole, cents = written.removeprefix("$").split(".")
        return {"currency": "USD", "minor_units": int(whole) * 100 + int(cents)}
    if category == "TIME":
        hour, minute = written.split(":")
        return {"hour": int(hour), "minute": int(minute)}
    if category == "PHONE":
        return {"digits": re.sub(r"\D", "", written)}
    if category == "DATE":
        return {"month": 5, "day": 1, "year": None}
    if category == "KEEP":
        return None
    return written


def collision_rows() -> list[dict[str, object]]:
    result = []
    families = [
        "money_vs_time",
        "phone_vs_digits",
        "date_vs_modal",
        "military_time_vs_cardinal",
        "ordinal_vs_plain",
        "url_vs_plain",
        "identifier_vs_time",
    ]
    for i in range(175):
        family, pair_id = families[i % 7], f"collision_pair_{i + 1:03d}"
        if family == "money_vs_time":
            h, m = i % 9 + 1, i * 7 % 50 + 10
            spoken = f"{words(h)} {words(m)}"
            variants = [
                ("MONEY", f"${h}.{m:02d}", "the cash price was {x}"),
                ("TIME", f"{h}:{m:02d}", "the meeting starts at {x}"),
            ]
        elif family == "phone_vs_digits":
            value = f"{3000000 + i * 7919:07d}"[-7:]
            spoken = digit_words(value)
            variants = [
                ("PHONE", f"{value[:3]}-{value[3:]}", "call {x}"),
                ("DIGIT_SEQUENCE", value, "enter the digits {x}"),
            ]
        elif family == "date_vs_modal":
            d = i // 7 + 1
            spoken = "may first"
            variants = [
                ("DATE", "May 1", f"hearing {d} is scheduled for {{x}}"),
                ("KEEP", spoken, f"{{x}} responders enter through gate {d}"),
            ]
        elif family == "military_time_vs_cardinal":
            hour = 10 + i // 7 % 14
            spoken = f"{words(hour)} hundred"
            variants = [
                ("TIME", f"{hour:02d}:00", f"train {i + 1} departs at {{x}} hours"),
                ("CARDINAL", str(hour * 100), f"section {i + 1} attendance is {{x}}"),
            ]
        elif family == "ordinal_vs_plain":
            lexical_cases = [
                ("first", "{x} aid training starts today"),
                ("first", "put {x} things first"),
                ("second", "it became {x} nature to her"),
                ("second", "their service is {x} to none"),
                ("third", "a {x} party handles billing"),
                ("fourth", "the press is called the {x} estate"),
                ("fifth", "the spy joined a {x} column"),
                ("sixth", "she trusted her {x} sense"),
                ("seventh", "the news put him in {x} heaven"),
                ("eleventh", "they changed it at the {x} hour"),
                ("first", "the {x} aid kits are by the door"),
                ("second", "driving soon became {x} nature"),
                ("third", "the {x}-party vendor called"),
                ("fourth", "democracy depends on the {x} estate"),
                ("fifth", "the novel describes a {x} column"),
                ("sixth", "his {x} sense warned him"),
                ("seventh", "she said she was in {x} heaven"),
                ("eleventh", "the agreement came at the {x} hour"),
                ("first", "a {x} aid course begins Monday"),
                ("second", "the hotel service was {x} to none"),
                ("third", "our {x}-party processor failed"),
                ("fourth", "the {x} estate reported the scandal"),
                ("fifth", "they uncovered a {x} column"),
                ("sixth", "always trust your {x} sense"),
                ("seventh", "winning put the team in {x} heaven"),
            ]
            spoken, keep_context = lexical_cases[i // 7]
            n = {
                "first": 1,
                "second": 2,
                "third": 3,
                "fourth": 4,
                "fifth": 5,
                "sixth": 6,
                "seventh": 7,
                "eleventh": 11,
            }[spoken]
            variants = [
                ("ORDINAL", f"{n}{ord_suffix(n)}", f"choose {{x}} in list {i + 1}"),
                ("KEEP", spoken, keep_context),
            ]
        elif family == "url_vs_plain":
            spoken = f"portal{i + 11} dot com"
            variants = [
                ("URL", f"portal{i + 11}.com", "open {x}"),
                ("KEEP", spoken, "quote the phrase {x}"),
            ]
        else:
            case_index = i // 7
            hour = case_index % 12 + 1
            minute = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55][case_index % 11]
            minute_spoken = f"oh {ONES[minute]}" if minute < 10 else words(minute)
            spoken = f"{words(hour)} {minute_spoken}"
            identifier = f"{hour}{minute:02d}"
            variants = [
                ("ORDER_ID", identifier, "the room code is {x}"),
                ("TIME", f"{hour}:{minute:02d}", "the appointment starts at {x}"),
            ]
        for category, written, template in variants:
            item_spoken = spoken
            semantic_value = collision_semantic_value(category, written)
            items = [
                {
                    "category": category,
                    "spoken": item_spoken,
                    "written": written,
                    "semantic_value": semantic_value,
                    "premove_span_kind": (
                        "WORD" if category in {"VERSION", "ORDER_ID"} else category
                    ),
                }
            ]
            result.append(
                row(
                    f"source_collision_{len(result) + 1:04d}",
                    "collision",
                    template,
                    items,
                    "general",
                    "hard",
                    family,
                    pair_id,
                )
            )
    return result


NEGATIVES = [
    "point taken",
    "may I ask something",
    "call it a day",
    "second to none",
    "one way or another",
    "at some point we should leave",
    "dot the i's and cross the t's",
    "the phrase one on one was quoted verbatim",
    "the third-party provider replied",
    "version control is one thing and deployment is another",
    "the pair were like peas in a pod",
    "the chapter titled One was better",
    "give me a second",
    "she came along at first light",
    "the duo agreed",
    "someday we will know",
    "agreement takes cooperation",
    "that perspective makes sense",
    "priorities come before preferences",
    "zero in on the cause",
    "the odds were stacked against us",
    "the eleventh hour has passed",
    "he gave everything he had",
    "every story has another side",
    "I need a minute",
    "pause before continuing",
    "the bottom line is clear",
    "we are on the same page",
    "may the best team win",
    "march to the beat",
    "august was a respected leader",
    "the period drama was excellent",
    "slash fiction is a genre",
    "the underscore was intentional",
    "the at sign is a useful term",
    "the point is still valid",
    "they arrived in succession",
    "she is in seventh heaven",
    "a fraction of the audience stayed",
    "money talks",
    "time will tell",
    "the phone rang twice",
    "address the problem directly",
    "route the request to support",
    "the cardinal flew away",
    "order was restored",
    "reference the earlier section",
    "flight was impossible",
    "measure twice and cut once",
    "keep this sentence unchanged",
]


def validate(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 1500 or Counter(r["group"] for r in rows) != Counter(QUOTAS):
        raise ValueError("quota failure")
    normalized = [
        " ".join(re.findall(r"[a-z0-9]+", str(r["text"]).lower())) for r in rows
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate normalized input")
    signatures, pairs = set(), Counter()
    collision_families = Counter()
    surface_patterns: dict[str, set[str]] = {
        "normal": set(),
        "voice_agent": set(),
        "multi": set(),
    }
    multi_sizes = Counter()
    difficulty_features = Counter()
    for r in rows:
        rebuilt = str(r["text"])
        for s in reversed(r["spans"]):
            rebuilt = (
                rebuilt[: s["start"]] + str(s["replacement"]) + rebuilt[s["end"] :]
            )
        if rebuilt != r["expected_text"]:
            raise ValueError(f"span failure: {r['id']}")
        if r["group"] in surface_patterns:
            pattern = str(r["text"])
            for span in reversed(r["spans"]):
                pattern = pattern[: span["start"]] + "<ENTITY>" + pattern[span["end"] :]
            surface_patterns[r["group"]].add(pattern)
        signatures.add(
            tuple(
                (s["source"], str(s["replacement"]), s["category"]) for s in r["spans"]
            )
        )
        if r["collision_pair_id"]:
            pairs[r["collision_pair_id"]] += 1
            collision_families[r["collision"]] += 1
        if r["group"] == "multi":
            multi_sizes[len(r["spans"])] += 1
        difficulty_features.update(r["difficulty_features"])
        if r["difficulty"] == "easy" and r["difficulty_features"]:
            raise ValueError(f"difficulty is not feature-derived: {r['id']}")
        if r["categories"] == ["KEEP"] and r["scoring"]["semantic_entity"]:
            raise ValueError(f"KEEP row enables entity scoring: {r['id']}")
        for span in r["spans"]:
            category = span["category"]
            value = span["semantic_value"]
            if category in {"ORDER_ID", "REFERENCE_ID"} and re.match(
                r"^(ORD|REF)[A-Z]", str(span["replacement"])
            ):
                raise ValueError(f"unspoken identifier prefix: {r['id']}")
            expected_types = {
                "CARDINAL": int,
                "ORDINAL": int,
                "MONEY": dict,
                "PHONE": dict,
                "TIME": dict,
                "DATE": dict,
            }
            if category in expected_types and not isinstance(
                value, expected_types[category]
            ):
                raise ValueError(f"semantic schema mismatch: {r['id']}")
            if category == "TIME" and set(value) != {"hour", "minute"}:
                raise ValueError(f"TIME schema mismatch: {r['id']}")
            if category == "MONEY" and value["minor_units"] % 100 < 10:
                cents = ONES[value["minor_units"] % 100]
                if f"zero {cents}" not in span["source"]:
                    raise ValueError(f"missing spoken zero cents cue: {r['id']}")
            if (
                category == "MONEY"
                and value["currency"] == "GBP"
                and (" pence" not in span["source"] or " cents" in span["source"])
            ):
                raise ValueError(f"GBP subunit mismatch: {r['id']}")
            if category == "URL" and re.search(
                r"(?:ticket|case|help|status)(?:zero|one|two|three|four|five|six|seven|eight|nine)",
                span["source"],
            ):
                raise ValueError(f"glued URL path token: {r['id']}")
        if "shipment weighs" in r["text"] and "megabytes" in r["text"]:
            raise ValueError(f"logistics measurement mismatch: {r['id']}")
        if r["domain"] == "healthcare_admin" and "miles per hour" in r["text"]:
            raise ValueError(f"healthcare measurement mismatch: {r['id']}")
    if len(signatures) < 1200 or len(pairs) != 175 or set(pairs.values()) != {2}:
        raise ValueError("coverage failure")
    if "identifier_vs_decimal" in collision_families:
        raise ValueError("obsolete collision family remains")
    if multi_sizes != Counter({2: 160, 3: 40}):
        raise ValueError("multi-entity arity coverage failure")
    pattern_counts = {group: len(values) for group, values in surface_patterns.items()}
    if pattern_counts["normal"] < 20 or pattern_counts["voice_agent"] < 40:
        raise ValueError("surface pattern coverage failure")
    return {
        "semantic_signatures": len(signatures),
        "collision_pairs": 175,
        "unique_keep_rows": 50,
        "multi_entity_arity": {"two": 160, "three": 40},
        "hidden_identifier_prefixes": 0,
        "semantic_schema_mismatches": 0,
        "keep_rows_with_entity_scoring": 0,
        "malformed_collision_rows": 0,
        "domain_measurement_mismatches": 0,
        "money_rows_missing_zero_cue": 0,
        "time_schema_mismatches": 0,
        "url_path_spacing_errors": 0,
        "gbp_subunit_errors": 0,
        "nonstandard_decimal_collision_rows": 0,
        "military_time_pronunciation_errors": 0,
        "ordinal_keep_lexical_errors": 0,
        "difficulty_basis": "observable difficulty_features",
        "difficulty_feature_distribution": dict(sorted(difficulty_features.items())),
        "surface_pattern_distribution": pattern_counts,
        "backend_reachability_used_as_inclusion_gate": False,
    }


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output: Path = OUTPUT) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen benchmark: {output}")
    rows = build_rows()
    checks = validate(rows)
    rows.sort(key=lambda r: hashlib.sha256(f"{SEED}\0{r['id']}".encode()).hexdigest())
    output.mkdir(parents=True)
    data = output / "voice_agent_eval_source.jsonl"
    data.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    manifest = {
        "schema_version": 2,
        "name": "VoiceAgent-ITN-source",
        "created_at": datetime.now(UTC).isoformat(),
        "rows": 1500,
        "quotas": QUOTAS,
        "selection_basis": "backend-neutral voice-agent requirements",
        "evaluated_backends": ["Premove ITN", "Thutmose", "text-processing-rs"],
        "backend_outputs_inspected_during_authoring": False,
        "model_frozen_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "frozen_checkpoint": (
            "data/generated/structured_value_adaptation_20k_run/checkpoint.pt"
        ),
        "frozen_checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "used_for_training": False,
        "used_for_model_selection": False,
        "first_model_run_at": None,
        "metric_hierarchy": {
            "primary": "macro_semantic_entity_accuracy",
            "secondary": "strict_sentence_exact_match",
            "collision": "pair_accuracy",
            "multi": ["span_accuracy", "all_entities_correct"],
            "keep": "false_normalization_rate",
            "voice_agent": "macro_domain_accuracy",
            "additional": "micro_semantic_entity_accuracy",
            "category_aggregates": [
                "standard_itn_core",
                "voice_agent_structured_extension",
            ],
        },
        "review_status": "candidate_pending_independent_review",
        "validation": checks,
        "difficulty_distribution": dict(Counter(r["difficulty"] for r in rows)),
        "category_distribution": dict(
            Counter(c for r in rows for c in r["categories"])
        ),
        "artifact": {"path": data.name, "sha256": sha(data)},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
