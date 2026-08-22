"""Repair the first synthetic corpus without hand-editing derived offsets."""

# Long template strings are intentionally kept readable as data.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from premove_itn.labels import BIO_LABELS
from premove_itn.tokenize import tokenize

MAX_SOURCE_TOKENS = 48
PROVENANCE_SUFFIX = "|dataset_cleanup_v1"

ROLE_BY_KIND = {
    "DATE": "date",
    "TIME": "time",
    "MONEY": "amount",
    "DECIMAL": "reading",
    "PHONE": "phone number",
    "ELECTRONIC": "address",
    "MEASUREMENT": "measurement",
    "DIGIT_SEQUENCE": "reference",
    "CARDINAL": "quantity",
    "ORDINAL": "position",
    "PUNCTUATION": "punctuation mark",
    "WHITELIST": "label",
    "WORD": "code",
}

SINGLE_TEMPLATES = {
    "train": [
        "The {role} is recorded as __SPAN_0__.",
        "Please use __SPAN_0__ for the {role}.",
        "The form gives __SPAN_0__ as the {role}.",
        "According to the note, the {role} is __SPAN_0__.",
        "We set the {role} to __SPAN_0__.",
        "The caller reported __SPAN_0__ for the {role}.",
        "I entered __SPAN_0__ in the {role} field.",
        "The next update concerns the {role}: __SPAN_0__.",
        "The {role} appears as __SPAN_0__ on the form.",
        "For this record, the {role} is __SPAN_0__.",
        "The message gives __SPAN_0__ under {role}.",
        "Use __SPAN_0__; it is the recorded {role}.",
    ],
    "validation": [
        "The {role} reads __SPAN_0__.",
        "The note gives __SPAN_0__ under {role}.",
        "This entry has __SPAN_0__ as its {role}.",
        "The record identifies the {role} with __SPAN_0__.",
        "On the submitted form, the {role} is __SPAN_0__.",
        "The request names __SPAN_0__ for the {role}.",
        "In the account, __SPAN_0__ is the {role}.",
        "The printed line shows __SPAN_0__ as the {role}.",
        "The file stores the {role} as __SPAN_0__.",
        "A caller gave __SPAN_0__ for the {role}.",
        "The summary lists __SPAN_0__ in the {role} field.",
        "The {role} on this request is __SPAN_0__.",
    ],
}

START_TEMPLATES = {
    "train": [
        "__SPAN_0__ is the recorded {role}.",
        "__SPAN_0__ appears in the {role} field.",
        "__SPAN_0__; use it as the {role}.",
        "__SPAN_0__ is what the form gives for the {role}.",
        "__SPAN_0__ belongs in the {role} field.",
        "__SPAN_0__ was supplied as the {role}.",
    ],
    "validation": [
        "__SPAN_0__ is the {role} on this request.",
        "__SPAN_0__ appears first in the {role} entry.",
        "__SPAN_0__ is listed under the {role}.",
        "__SPAN_0__; the form uses it for the {role}.",
        "__SPAN_0__ was entered as the {role}.",
        "__SPAN_0__ identifies the {role} here.",
    ],
}

MULTI_TEMPLATES = {
    "train": {
        2: [
            "The {subject} records the {role0} as __SPAN_0__ and the {role1} as __SPAN_1__.",
            "For the {subject}, __SPAN_0__ is the {role0}; __SPAN_1__ is the {role1}.",
            "The {subject} uses __SPAN_0__ for the {role0}, followed by __SPAN_1__ for the {role1}.",
            "The {subject} puts __SPAN_0__ beside the {role1}; __SPAN_1__ fills the {role0} entry.",
            "The {subject} gives __SPAN_0__ before __SPAN_1__ in the submitted line.",
            "The note for the {subject} starts with __SPAN_0__ and ends with __SPAN_1__.",
            "The {subject} includes __SPAN_0__ near __SPAN_1__ in the same entry.",
            "On the {subject}, __SPAN_0__ comes first and __SPAN_1__ follows.",
            "The {subject} shows __SPAN_0__ __SPAN_1__ in the paired fields.",
            "The caller gave __SPAN_0__, then added __SPAN_1__ for the {subject}.",
            "The {subject} lists __SPAN_0__; the next value is __SPAN_1__.",
            "The handoff note carries __SPAN_0__ and later mentions __SPAN_1__.",
        ],
        3: [
            "The {subject} records __SPAN_0__, __SPAN_1__, and __SPAN_2__ in that order.",
            "For the {subject}, __SPAN_0__ comes first, __SPAN_1__ follows, and __SPAN_2__ closes the line.",
            "The {subject} gives __SPAN_0__ with __SPAN_1__ nearby and __SPAN_2__ at the end.",
            "The note for the {subject} starts with __SPAN_0__, then includes __SPAN_1__ and __SPAN_2__.",
            "On the {subject}, __SPAN_0__ is followed by __SPAN_1__; __SPAN_2__ appears below.",
            "The caller supplied __SPAN_0__, added __SPAN_1__, and finished with __SPAN_2__.",
            "The {subject} places __SPAN_0__ before __SPAN_1__ and keeps __SPAN_2__ in the final field.",
            "The submitted {subject} contains __SPAN_0__ near __SPAN_1__, with __SPAN_2__ after both.",
            "The record begins __SPAN_0__; it continues with __SPAN_1__ and ends at __SPAN_2__.",
            "The {subject} has __SPAN_0__ beside __SPAN_1__ and __SPAN_2__ on the next line.",
        ],
        4: [
            "The {subject} records __SPAN_0__, __SPAN_1__, __SPAN_2__, and __SPAN_3__.",
            "For the {subject}, __SPAN_0__ starts the line, followed by __SPAN_1__, __SPAN_2__, and __SPAN_3__.",
            "The note for the {subject} gives __SPAN_0__ before __SPAN_1__; __SPAN_2__ and __SPAN_3__ follow.",
            "The submitted {subject} contains __SPAN_0__, then __SPAN_1__, with __SPAN_2__ and __SPAN_3__ below.",
            "The caller supplied __SPAN_0__, added __SPAN_1__, recorded __SPAN_2__, and finished with __SPAN_3__.",
            "On the {subject}, __SPAN_0__ and __SPAN_1__ share the first line; __SPAN_2__ and __SPAN_3__ follow.",
            "The {subject} begins with __SPAN_0__; the remaining values are __SPAN_1__, __SPAN_2__, and __SPAN_3__.",
            "The record places __SPAN_0__ near __SPAN_1__, then lists __SPAN_2__ and __SPAN_3__.",
        ],
    },
    "validation": {
        2: [
            "The {subject} shows __SPAN_0__ for one entry and __SPAN_1__ for another.",
            "Within the {subject}, __SPAN_0__ comes before __SPAN_1__.",
            "The {subject} puts __SPAN_0__ in the first line and __SPAN_1__ in the next.",
            "The submitted {subject} contains __SPAN_0__ beside __SPAN_1__.",
            "A note about the {subject} says __SPAN_0__, followed by __SPAN_1__.",
            "The {subject} starts with __SPAN_0__ and continues with __SPAN_1__.",
            "The caller mentions __SPAN_0__ and then gives __SPAN_1__ for the {subject}.",
            "The record places __SPAN_0__ near __SPAN_1__ in the same sentence.",
            "The {subject} has __SPAN_0__ __SPAN_1__ on the printed line.",
            "The file keeps __SPAN_0__ ahead of __SPAN_1__ for the {subject}.",
            "The request lists __SPAN_0__; __SPAN_1__ follows in the entry.",
            "The {subject} mentions __SPAN_0__ and later returns to __SPAN_1__.",
        ],
        3: [
            "The {subject} lists __SPAN_0__, then __SPAN_1__, followed by __SPAN_2__.",
            "Within the {subject}, __SPAN_0__ begins the entry, __SPAN_1__ continues it, and __SPAN_2__ ends it.",
            "The submitted {subject} contains __SPAN_0__ beside __SPAN_1__, with __SPAN_2__ after them.",
            "A note about the {subject} says __SPAN_0__; it adds __SPAN_1__ and __SPAN_2__ afterward.",
            "The caller mentions __SPAN_0__, supplies __SPAN_1__, and closes with __SPAN_2__.",
            "The {subject} starts at __SPAN_0__, moves to __SPAN_1__, and finishes at __SPAN_2__.",
            "The record places __SPAN_0__ before __SPAN_1__; __SPAN_2__ is in the final field.",
            "The file contains __SPAN_0__, __SPAN_1__, and __SPAN_2__ on successive lines.",
            "The request gives __SPAN_0__ near __SPAN_1__ and records __SPAN_2__ below.",
            "The {subject} carries __SPAN_0__, followed by __SPAN_1__ and then __SPAN_2__.",
        ],
        4: [
            "The {subject} lists __SPAN_0__, __SPAN_1__, __SPAN_2__, and __SPAN_3__.",
            "Within the {subject}, __SPAN_0__ starts the record; __SPAN_1__, __SPAN_2__, and __SPAN_3__ follow.",
            "The submitted {subject} begins with __SPAN_0__, continues through __SPAN_1__ and __SPAN_2__, and ends at __SPAN_3__.",
            "A note about the {subject} gives __SPAN_0__ before __SPAN_1__, then records __SPAN_2__ and __SPAN_3__.",
            "The caller mentions __SPAN_0__, supplies __SPAN_1__, adds __SPAN_2__, and closes with __SPAN_3__.",
            "The record puts __SPAN_0__ beside __SPAN_1__; __SPAN_2__ and __SPAN_3__ are listed afterward.",
            "The file contains __SPAN_0__, followed by __SPAN_1__, __SPAN_2__, and __SPAN_3__.",
            "The request starts at __SPAN_0__ and continues with __SPAN_1__, __SPAN_2__, and __SPAN_3__.",
        ],
    },
}

SUBJECTS = {
    "train": [
        "booking",
        "invoice",
        "shipment",
        "account",
        "reservation",
        "service note",
        "delivery record",
        "case file",
        "customer profile",
        "travel plan",
        "order form",
        "appointment record",
        "support ticket",
        "registration",
        "message thread",
        "handoff note",
    ],
    "validation": [
        "itinerary",
        "receipt",
        "parcel record",
        "member file",
        "calendar entry",
        "work order",
        "contact sheet",
        "claim form",
        "visit record",
        "dispatch note",
        "client record",
        "service request",
        "reservation file",
        "account note",
        "booking sheet",
        "case summary",
    ],
}

ROOM_TEMPLATES = [
    "The site perimeter measures __SPAN_0__.",
    "The corridor length comes to __SPAN_0__.",
    "The venue grounds measure __SPAN_0__.",
    "The facilities plan records __SPAN_0__ along the outer boundary.",
    "The access route is __SPAN_0__ from the loading area to the entrance.",
    "The hall footprint covers a length of __SPAN_0__.",
]

NEGATIVE_CONTEXTS = {
    "train": [
        "The handout says {payload}.",
        "The caller's note includes {payload}.",
        "A line in the manual reads {payload}.",
        "The catalog entry states {payload}.",
        "The team discussed {payload} during the review.",
        "The message quotes {payload} from the guide.",
        "The reference sheet contains {payload}.",
        "The draft records {payload} in the margin.",
        "The archive includes {payload} under the heading.",
        "The printed page mentions {payload}.",
        "I found {payload} in the submitted note.",
        "The support log contains {payload}.",
        "The checklist refers to {payload}.",
        "The form has a line for {payload}.",
        "The transcript includes {payload} near the opening.",
        "The editor marked {payload} in the source.",
        "The customer wrote {payload} in the comment.",
        "The record begins with {payload}.",
        "The page heading mentions {payload}.",
        "The training note gives {payload} as an example.",
        "The document contains {payload} in a footnote.",
        "The reviewer cited {payload} from the old form.",
        "The file preserves {payload} from the original note.",
        "The account history mentions {payload}.",
        "The sentence ends with {payload}.",
    ],
    "validation": [
        "The memo records {payload}.",
        "The submitted page contains {payload}.",
        "A note from the caller says {payload}.",
        "The reference card includes {payload}.",
        "The file quotes {payload}.",
    ],
}

NEGATIVE_PAYLOADS = {
    "literal": {
        "train": [
            "the word period in the style guide",
            "the word comma in the punctuation table",
            "the term percent in the glossary",
            "mister in the printed salutation",
            "the word ampersand in the keyboard notes",
            "the label plus on the diagram",
            "the word slash in the filename guide",
            "colon in the heading examples",
            "the term dash in the typesetting notes",
            "question mark in the copy-editing guide",
            "the word bracket in the layout notes",
            "underscore in the database manual",
            "the label dot on the map legend",
            "hyphen in the address instructions",
            "the phrase at sign in the glossary",
            "the word semicolon in the reference card",
            "the label star on the keyboard chart",
            "the term pipe in the command guide",
        ],
        "validation": [
            "the word quotation mark in the style guide",
            "the term apostrophe in the glossary",
            "the label backslash on the diagram",
            "the word brace in the layout notes",
            "the phrase equal sign in the reference card",
            "the term caret in the typesetting notes",
            "the label percent sign in the handbook",
            "the word parenthesis in the copy guide",
            "the term ellipsis in the punctuation table",
            "the label ampersand in the catalog notes",
        ],
    },
    "malformed": {
        "train": [
            "the amount was twenty point",
            "the total ended at thirty point",
            "the phone number begins with plus",
            "the address starts with plus",
            "the appointment is on june",
            "the visit was scheduled for april",
            "the time was half past",
            "the call began at quarter to",
            "the domain is example dot",
            "the link ends at example dot",
            "the code starts with double",
            "the quantity was four hundred and",
            "the date was the fifth of",
            "the amount reads one hundred point",
            "the phone line ends after seven",
            "the address contains example at",
            "the measurement was two point",
            "the time was six oh",
        ],
        "validation": [
            "the price stopped at forty point",
            "the phone line starts with plus",
            "the appointment falls on october",
            "the time ends at quarter past",
            "the domain stops at sample dot",
            "the count was twelve and",
            "the date ends after march",
            "the code begins with triple",
            "the measurement ends at five point",
            "the address starts with sample at",
        ],
    },
    "already": {
        "train": [
            "the receipt prints 09:15 a.m.",
            "the report shows 0.75",
            "the reference is 430",
            "the contact line is 807-492-1301",
            "the address is example.com",
            "the calendar lists June 5",
            "the account uses dr. shah",
            "the invoice total is $24.50",
            "the form contains 18 units",
            "the version is 2.4",
            "the ticket number is 4819",
            "the package weighs 3 kg",
            "the code is A-17",
            "the meeting starts at 14:30",
            "the note names st. mary",
            "the page lists 7th place",
            "the balance is $0.91",
            "the file points to r.smea@example.com",
        ],
        "validation": [
            "the receipt shows 08:40 a.m.",
            "the reading is 1.25",
            "the reference is 712",
            "the contact number is 415-555-0138",
            "the address is sample.org",
            "the schedule says July 9",
            "the label is mr. lee",
            "the total is $18.20",
            "the form lists 24 items",
            "the code is B-08",
        ],
    },
    "ordinary": {
        "train": [
            "the billing note mentions a late fee",
            "the account owner changed the mailing address",
            "the customer asked for a paper receipt",
            "the delivery window moved to next week",
            "the support ticket is waiting for a reply",
            "the clerk attached the signed form",
            "the order includes a replacement part",
            "the visitor left a message at reception",
            "the team approved the updated schedule",
            "the archive contains the original invoice",
            "the caller described a damaged package",
            "the manager reviewed the open request",
            "the shipment arrived without a note",
            "the profile has a secondary contact",
            "the meeting room is near the lobby",
            "the technician checked the power cable",
            "the customer prefers email contact",
            "the form needs a signature",
        ],
        "validation": [
            "the billing note mentions a refund",
            "the member changed the delivery address",
            "the customer requested a phone call",
            "the service window moved to Friday",
            "the case is waiting for approval",
            "the clerk attached a photo",
            "the order needs a replacement label",
            "the visitor left a message with security",
            "the team approved the new route",
            "the archive contains the signed receipt",
        ],
    },
}


def record_number(record_id: str) -> int:
    match = re.search(r"(\d+)$", record_id)
    if match is None:
        raise ValueError(f"record ID has no numeric suffix: {record_id}")
    return int(match.group(1))


def replace_span_payload(record: dict[str, Any]) -> list[dict[str, Any]]:
    spans = [dict(span) for span in record["spans"]]
    family = record["template_family"]
    for span in spans:
        if span["kind"] != "MEASUREMENT":
            continue
        source = span["source"]
        if "parcel weighs" in record["text"] and source.endswith(" hours"):
            span["source"] = f"{source[:-6]} kilograms"
            span["replacement"] = re.sub(r"\s+h$", " kg", span["replacement"])
        elif "room_measure" in family:
            number_phrase = source.rsplit(" ", 1)[0]
            span["source"] = f"{number_phrase} meters"
            span["replacement"] = re.sub(r"\s+(?:mi|m)$", " m", span["replacement"])
    return spans


def render(template: str, sources: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text = template
    offsets: list[tuple[int, int]] = []
    for index, source in enumerate(sources):
        marker = f"__SPAN_{index}__"
        start = text.index(marker)
        text = text.replace(marker, source, 1)
        offsets.append((start, start + len(source)))
    return text, offsets


def rebuild_derived_fields(text: str, spans: list[dict[str, Any]]) -> None:
    for span in spans:
        assert text[span["start"] : span["end"]] == span["source"]

    expected = text
    for span in reversed(spans):
        expected = (
            expected[: span["start"]] + span["replacement"] + expected[span["end"] :]
        )

    tokens = tokenize(text)
    labels = ["O"] * len(tokens)
    for span in spans:
        covered = [
            index
            for index, token in enumerate(tokens)
            if token.start >= span["start"] and token.end <= span["end"]
        ]
        assert covered, (span, text)
        assert tokens[covered[0]].start == span["start"]
        assert tokens[covered[-1]].end == span["end"]
        for offset, index in enumerate(covered):
            assert labels[index] == "O"
            labels[index] = ("B-" if offset == 0 else "I-") + span["kind"]

    assert all(label in BIO_LABELS for label in labels)
    assert len(tokens) <= MAX_SOURCE_TOKENS, (len(tokens), text)

    return {
        "tokens": [
            {"text": token.text, "start": token.start, "end": token.end}
            for token in tokens
        ],
        "bio_labels": labels,
        "expected_text": expected,
    }


def negative_text(split: str, category: str, ordinal: int) -> str:
    payloads = NEGATIVE_PAYLOADS[category][split]
    contexts = NEGATIVE_CONTEXTS[split]
    payload = payloads[ordinal // len(contexts)]
    context = contexts[ordinal % len(contexts)]
    return context.format(payload=payload)


def positive_template(
    record: dict[str, Any], spans: list[dict[str, Any]], ordinal: int
) -> str:
    split = record["split"]
    family = record["template_family"]
    if "room_measure" in family:
        return ROOM_TEMPLATES[ordinal % len(ROOM_TEMPLATES)]

    if len(spans) == 1:
        kind = spans[0]["kind"]
        role = ROLE_BY_KIND[kind]
        if family.startswith("whole_") or (split == "train" and ordinal < 19):
            return "__SPAN_0__"
        if ordinal % 7 == 0:
            return START_TEMPLATES[split][ordinal % len(START_TEMPLATES[split])].format(
                role=role
            )
        return SINGLE_TEMPLATES[split][ordinal % len(SINGLE_TEMPLATES[split])].format(
            role=role
        )

    templates = MULTI_TEMPLATES[split][len(spans)]
    roles = [ROLE_BY_KIND[span["kind"]] for span in spans]
    subjects = SUBJECTS[split]
    template = templates[ordinal % len(templates)]
    return template.format(
        subject=subjects[ordinal % len(subjects)],
        role0=roles[0],
        role1=roles[1] if len(roles) > 1 else "value",
    )


def repair_record(
    record: dict[str, Any],
    negative_ordinals: dict[tuple[str, str], int],
    positive_ordinals: dict[tuple[str, int], int],
) -> dict[str, Any]:
    spans = replace_span_payload(record)
    split = record["split"]
    family = record["template_family"]
    if not spans:
        category = family.rsplit("_", 1)[-1]
        ordinal = negative_ordinals[(split, category)]
        negative_ordinals[(split, category)] += 1
        text = negative_text(split, category, ordinal)
    else:
        key = (split, len(spans))
        ordinal = positive_ordinals[key]
        positive_ordinals[key] += 1
        template = positive_template(record, spans, ordinal)
        text, offsets = render(template, [span["source"] for span in spans])
        for span, (start, end) in zip(spans, offsets, strict=True):
            span["start"] = start
            span["end"] = end
        if len(tokenize(text)) > MAX_SOURCE_TOKENS:
            compact = {
                2: "Record: __SPAN_0__; __SPAN_1__.",
                3: "Record: __SPAN_0__; __SPAN_1__; __SPAN_2__.",
                4: "Record: __SPAN_0__; __SPAN_1__; __SPAN_2__; __SPAN_3__.",
            }[len(spans)]
            text, offsets = render(compact, [span["source"] for span in spans])
            for span, (start, end) in zip(spans, offsets, strict=True):
                span["start"] = start
                span["end"] = end

    derived = rebuild_derived_fields(text, spans)
    derived.pop("_expected", None)
    repaired = dict(record)
    repaired.update({"text": text, "spans": spans, **derived})
    provenance = repaired.get("provenance", "synthetic")
    if PROVENANCE_SUFFIX not in provenance:
        repaired["provenance"] = provenance + PROVENANCE_SUFFIX
    return repaired


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/generated/records.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/records.jsonl"),
    )
    args = parser.parse_args()

    records = load_records(args.input)
    negative_ordinals: dict[tuple[str, str], int] = defaultdict(int)
    positive_ordinals: dict[tuple[str, int], int] = defaultdict(int)
    repaired = [
        repair_record(record, negative_ordinals, positive_ordinals)
        for record in records
    ]

    assert len(repaired) == 10_000
    assert {record["split"] for record in repaired} == {"train", "validation"}
    assert all(len(record["tokens"]) <= MAX_SOURCE_TOKENS for record in repaired)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in repaired:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(args.output)
    print(f"wrote {len(repaired)} repaired records to {args.output}")


if __name__ == "__main__":
    main()
