"""Rebuild Batch 01 contexts and all derived dataset annotations."""

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
PROVENANCE_SUFFIX = "|dataset_cleanup_v2"

PEOPLE = {
    "train": (
        "Mara",
        "Jonah",
        "Priya",
        "Luis",
        "Nora",
        "Evan",
        "Tessa",
        "Owen",
        "Leila",
        "Marcus",
        "Rina",
        "Caleb",
        "Asha",
        "Dylan",
        "Mina",
        "Theo",
        "Iris",
        "Samir",
        "June",
        "Ravi",
        "Elena",
        "Noah",
        "Sofia",
        "Arun",
    ),
    "validation": (
        "Amara",
        "Felix",
        "Mei",
        "Andre",
        "Lena",
        "Isaac",
        "Nadia",
        "Cole",
        "Anika",
        "Peter",
        "Yara",
        "Mateo",
        "Sana",
        "Eli",
        "Maya",
        "Julian",
        "Inez",
        "Omar",
        "Clara",
        "Dev",
        "Hana",
        "Victor",
        "Zoe",
        "Kiran",
    ),
}

PLACES = {
    "train": (
        "the clinic",
        "the station",
        "the south entrance",
        "the repair shop",
        "the school office",
        "the harbor",
        "the loading dock",
        "the west gate",
        "the service desk",
        "the community hall",
        "the market",
        "the depot",
        "the hotel lobby",
        "the warehouse",
        "the library",
        "the theater",
        "the lab",
        "the apartment office",
    ),
    "validation": (
        "the museum",
        "the north platform",
        "the pharmacy",
        "the front counter",
        "the ferry terminal",
        "the garden gate",
        "the design studio",
        "the conference room",
        "the delivery bay",
        "the courthouse",
        "the archive",
        "the main lobby",
        "the ticket window",
        "the field office",
        "the gallery",
        "the test kitchen",
        "the west stairwell",
        "the travel office",
    ),
}

OBJECTS = {
    "train": (
        "appointment",
        "shipment",
        "repair",
        "reservation",
        "membership",
        "invoice",
        "inspection",
        "delivery",
        "registration",
        "claim",
        "schedule",
        "order",
        "return",
        "application",
        "handoff",
        "visit",
        "account",
        "booking",
        "parcel",
        "service call",
    ),
    "validation": (
        "itinerary",
        "receipt",
        "parcel",
        "membership",
        "work order",
        "claim",
        "visit",
        "dispatch",
        "reservation",
        "account note",
        "delivery",
        "inspection",
        "application",
        "return",
        "invoice",
        "booking",
        "service request",
        "registration",
        "handoff",
        "travel plan",
    ),
}

LEAD_SUBJECTS = {
    "train": (
        "The appointment for {person}",
        "The {person} shipment",
        "The repair for {person}",
        "The reservation under {person}",
        "The membership for {person}",
        "The invoice from {person}",
        "The {person} inspection",
        "The delivery for {person}",
        "The registration for {person}",
        "The claim from {person}",
        "The order for {person}",
        "The visit arranged by {person}",
    ),
    "validation": (
        "The itinerary for {person}",
        "The receipt from {person}",
        "The parcel assigned to {person}",
        "The membership under {person}",
        "The work order for {person}",
        "The claim filed by {person}",
        "The visit planned by {person}",
        "The dispatch for {person}",
        "The reservation under {person}",
        "The account note from {person}",
        "The delivery requested by {person}",
        "The inspection for {person}",
    ),
}

LEAD_VERBS = (
    "needs checking",
    "is nearly ready",
    "awaits approval",
    "needs one detail",
    "changes this afternoon",
    "is ready for review",
    "needs attention",
    "awaits pickup",
    "fits the morning plan",
    "is being arranged",
    "can move forward",
    "is set for handoff",
    "needs a quick call",
    "has a new deadline",
    "needs follow-up",
    "needs a route change",
    "is ready to print",
    "waits on one answer",
    "has a missing detail",
    "is queued for dispatch",
)

LEAD_LOCATIONS = (
    "near {place}",
    "at {place}",
    "outside {place}",
    "beside {place}",
    "by {place}",
    "past {place}",
    "from {place}",
    "inside {place}",
    "around {place}",
    "behind {place}",
    "across from {place}",
    "alongside {place}",
    "next to {place}",
    "under {place}",
    "beyond {place}",
    "toward {place}",
    "through {place}",
    "between {place} and the road",
    "at {place}'s edge",
)

ROLE_FREE_CONTEXTS = {
    "DATE": (
        "{person} can meet on {value} near {place}",
        "Move the visit to {value} after the school event",
        "The handoff belongs on {value}, not the earlier draft",
        "We will leave on {value} if the ferry is running",
        "Book the inspection for {value} at {place}",
        "The return is set for {value} unless plans change",
        "Keep {value} for the appointment with {person}",
        "The delivery should arrive on {value} before sunset",
    ),
    "TIME": (
        "{person} can call at {value} from {place}",
        "The gate opens at {value}; arrive before the crowd",
        "We should leave at {value} to catch the train",
        "Set the reminder for {value} after the meeting",
        "The service begins at {value} near {place}",
        "I can meet you at {value} by the west entrance",
        "The pickup is due at {value}, so keep the path clear",
        "The evening session starts at {value} for {person}",
    ),
    "MONEY": (
        "{person} can spend {value} on the {object}",
        "The refund should be {value}, sent to the original card",
        "Reserve {value} for {person}'s deposit at {place}",
        "The vendor quoted {value} for the replacement",
        "I can approve {value} if delivery is included",
        "The repair is worth {value} to {person}",
        "Charge {value} to the account before the handoff",
        "The budget leaves {value} for the final visit",
    ),
    "DECIMAL": (
        "The sensor settled at {value} after calibration",
        "Use {value} as the rate for the {object}",
        "The reading near {place} came back as {value}",
        "The mixture needs {value} before the next test",
        "The gauge showed {value} while {person} watched",
        "Keep {value} as the threshold for the repair",
        "The measured ratio is {value} on this sample",
        "The new setting uses {value} for the device",
    ),
    "PHONE": (
        "Call {person} at {value} after the meeting",
        "Use {value} when the driver reaches {place}",
        "The office can be reached at {value} before noon",
        "Send the technician to {place} and call {value}",
        "I left {value} for the return call",
        "The spare contact for {person} is {value}",
        "Ring {value} if the delivery misses the gate",
        "The clinic should use {value} for the reminder",
    ),
    "ELECTRONIC": (
        "Send the receipt to {value} for {person}",
        "The confirmation should reach {value} before the visit",
        "Forward the itinerary to {value} after booking",
        "Use {value} for the account notice",
        "The vendor wrote to {value} about the repair",
        "I will send the file to {value} from {place}",
        "The contact address for {person} is {value}",
        "Put {value} on the delivery notice",
    ),
    "DIGIT_SEQUENCE": (
        "The locker code is {value} for {person}",
        "Use {value} to open the cabinet at {place}",
        "The booking reference reads {value} on the ticket",
        "Enter {value} when the service desk asks",
        "I wrote {value} on the parcel label",
        "The access sequence for the {object} is {value}",
        "Give {value} to the clerk at the counter",
        "The {object} code confirms {value} for {person}",
    ),
    "CARDINAL": (
        "There are {value} guests waiting near {place}",
        "The order needs {value} parts for {person}",
        "Set aside {value} seats for {person}",
        "The shipment contains {value} boxes for the depot",
        "We expect {value} visitors during the afternoon",
        "The repair uses {value} screws from the kit",
        "The kitchen prepared {value} meals for the event",
        "The account covers {value} service calls",
    ),
    "ORDINAL": (
        "{person} takes {value} with the {object}",
        "Use the {value} stop when the train arrives",
        "The {value} attempt worked for {person}",
        "Choose the {value} shelf in the archive",
        "The {value} item belongs in the delivery crate",
        "We are waiting in the {value} position",
        "The {value} step starts the inspection",
        "Choose {value} for {person}'s form",
    ),
    "PUNCTUATION": (
        "Put {value} after the greeting in the draft",
        "The editor marked {value} before the quoted line",
        "Add {value} between the two clauses",
        "The {object} note includes {value} for {person}",
        "Use {value} before the closing sentence",
        "The heading ends with {value} on the printed page",
        "Insert {value} where the reply changes direction",
        "The caption needs {value} before the location",
    ),
    "WORD": (
        "The identifier starts with {value} on the label",
        "Use {value} for the model name at {place}",
        "The part code begins with {value} for {person}",
        "I wrote {value} beside the device serial",
        "The catalog lists {value} as the product code",
        "Keep {value} at the start of the account tag",
        "The technician marked {value} on the repair sheet",
        "The {object} package lists {value} for {person}",
    ),
}

MULTI_CONTEXTS = {
    "DATE": (
        "{person} schedules the {object} for {value} near {place}",
        "The {object} uses {value} for {person}",
        "Meet {person} on {value} at {place}",
        "We move the {object} to {value} before {person}",
        "The {person} handoff uses {value} with {object}",
        "Keep {value} for {person}'s {object}",
        "The {person} leaves for {object} on {value}",
        "The inspection for {person} falls on {value} near {object}",
    ),
    "TIME": (
        "{person} starts the {object} at {value} near {place}",
        "The {object} begins at {value} with {person}",
        "Meet {person} at {value} by {place}",
        "For {person}, leave {value} with {object}",
        "The {person} handoff uses {value} with {object}",
        "Keep {value} for {person}'s {object}",
        "The {person} departs for {object} at {value}",
        "The inspection for {person} starts at {value} near {object}",
    ),
    "MONEY": (
        "{person} budgets {value} for the {object}",
        "The {object} costs {value} for {person}",
        "Reserve {value} with {person} at {place}",
        "We quote {value} to {person} for the repair",
        "The {person} account holds {value} near {place}",
        "Pay {value} from the {object} budget",
        "The deposit for {person} is {value} at {place}",
        "Keep {value} with the {object} receipt",
    ),
    "DECIMAL": (
        "{person} records {value} for the {object}",
        "The {object} settles at {value} near {place}",
        "Use {value} at {place} for {person}",
        "The {person} reading reaches {value} on the device",
        "We set the {object} threshold to {value} beside {place}",
        "The gauge near {place} shows {value} for {person}",
        "Keep {value} with the {object} notes",
        "The sample from {person} measures {value} at {place}",
    ),
    "PHONE": (
        "Call {person} at {value} from {place}",
        "The {object} desk uses {value} for {person}",
        "Send {person} to {place} and call {value}",
        "The return call for {person} uses {value}",
        "Keep {value} for {person}'s contact",
        "The {person} reminder goes to {value} near {place}",
        "Ring {value} for the {object} at {place}",
        "The clinic lists {value} beside {person}",
    ),
    "ELECTRONIC": (
        "Send the {object} receipt to {value} for {person}",
        "The confirmation for {person} reaches {value} near {place}",
        "Forward the {object} file to {value} after booking",
        "Use {value} for the {person} account notice",
        "The vendor writes to {value} about the {object}",
        "The itinerary from {place} goes to {value} for {person}",
        "The contact address for {person} is {value} beside {place}",
        "Put {value} on the {object} delivery notice",
    ),
    "DIGIT_SEQUENCE": (
        "The {object} code for {person} is {value}",
        "Use {value} at {place} for the {object}",
        "The {person} booking reference reads {value} on the ticket",
        "Enter {value} when the {object} desk asks",
        "I wrote {value} on {person}'s parcel label",
        "The access sequence for {object} is {value} near {place}",
        "Give {value} to {person} at the counter",
        "The {object} code confirms {value} for {person}",
    ),
    "CARDINAL": (
        "{person} has {value} guests at {place}",
        "The {object} order has {value} for {person}",
        "Set {value} seats aside for {person}",
        "The shipment for {person} contains {value} boxes",
        "We expect {value} visitors at {place}",
        "The {object} repair uses {value} screws",
        "The kitchen made {value} meals for {person}",
        "The account for {person} covers {value} service calls",
    ),
    "ORDINAL": (
        "For {person}, take {value} with {object}",
        "Use {value} for {person}'s train stop",
        "The {person} attempt was {value} for the {object}",
        "Choose the {value} shelf for the {object}",
        "The {object} item is {value} for {person}",
        "We are in position {value} with {person}",
        "The {object} inspection reaches step {value} for {person}",
        "Choose {value} for {person}'s form",
    ),
    "PUNCTUATION": (
        "{person} puts {value} after the greeting",
        "The {object} editor marks {value} before the quoted line",
        "Add {value} between {person}'s two clauses",
        "The {object} note includes {value} for {person}",
        "Use {value} before {person}'s closing sentence",
        "The {object} heading ends with {value} on the page",
        "Insert {value} where {person}'s reply changes direction",
        "The {object} caption needs {value} before the location",
    ),
    "WORD": (
        "{person} starts the identifier with {value} on the label",
        "Use {value} for the {object} model at {place}",
        "The {person} part code begins with {value}",
        "I wrote {value} beside the {object} serial",
        "The catalog lists {value} as {person}'s product code",
        "Keep {value} at the start of the {object} account tag",
        "The technician marked {value} on {person}'s repair sheet",
        "The {object} package lists {value} for {person}",
    ),
    "MEASUREMENT": (
        "The {object} for {person} measures {value} near {place}",
        "The shipment weighs {value} at {place} for {person}",
        "The {object} route covers {value} beyond {place}",
        "Pack {value} with {object}",
        "The {person} sample records {value} for {object}",
        "The crate for {object} carries {value} near {person}",
        "The service lasts {value} for {person} at {place}",
        "The gap by {place} measures {value} for the {object}",
    ),
    "WHITELIST": (
        "The {object} contact for {person} is {value}",
        "Ask for {value} at {place} about the {object}",
        "The badge for {person} should show {value}",
        "{value} is handling the {object} handoff for {person}",
        "Send the {object} invitation to {value} at {place}",
        "The account for {person} lists {value} near {place}",
        "The catalog puts {value} beside the {object}",
        "The technician marked {value} on {person}'s package",
    ),
}

COMPACT_CLAUSES = {
    "DATE": "meet on {value}",
    "TIME": "leave at {value}",
    "MONEY": "pay {value}",
    "DECIMAL": "set {value}",
    "PHONE": "call {value}",
    "ELECTRONIC": "email {value}",
    "DIGIT_SEQUENCE": "use code {value}",
    "CARDINAL": "send {value} items",
    "ORDINAL": "take {value}",
    "PUNCTUATION": "add {value}",
    "WORD": "use {value}",
}

MULTI_CONNECTORS = {
    2: (
        "{lead}; {a}, and {b}.",
        "{lead}. {a}. Then {b}.",
        "{lead}: {a}; {b}.",
        "{lead}. {a}, and {b}.",
        "{lead}. First, {a}. Next, {b}.",
        "{lead}. {a}; the next detail is {b}.",
        "{lead}. The notes include {a}; they also include {b}.",
        "{lead}. {a}. A second detail is {b}.",
        "{lead}: first {a}; then {b}.",
        "{lead}. Two details follow: {a}; {b}.",
        "{lead}. One detail is {a}; another is {b}.",
        "{lead}. The record contains {a}; it also contains {b}.",
    ),
    3: (
        "{lead}; {a}, {b}, and {c}.",
        "{lead}. {a}. Then {b}. Finally, {c}.",
        "{lead}: {a}; {b}; {c}.",
        "{lead}. First, {a}. Next, {b}. Last, {c}.",
        "{lead}. {a}, and {b}; {c}.",
        "{lead}. Three details follow: {a}; {b}; {c}.",
        "{lead}. The notes include {a}; {b}; and {c}.",
        "{lead}. One detail is {a}; another is {b}; the last is {c}.",
        "{lead}. {a}. A second detail is {b}. A third is {c}.",
        "{lead}. The record contains {a}; it also contains {b} and {c}.",
    ),
    4: (
        "{lead}; {a}, {b}, {c}, and {d}.",
        "{lead}. First, {a}. Next, {b}. Then, {c}. Last, {d}.",
        "{lead}: {a}; {b}; {c}; and {d}.",
        "{lead}. Four details follow: {a}; {b}; {c}; {d}.",
        "{lead}. {a}. Then {b}. Next, {c}. Finally, {d}.",
        "{lead}. The notes include {a}; {b}; {c}; and {d}.",
        "{lead}. One detail is {a}; another is {b}; then {c}; finally {d}.",
        "{lead}. The record contains {a}; {b}; {c}; and {d}.",
    ),
}

NEGATIVE_TERMS = {
    "literal": {
        "train": (
            "the word period",
            "the word comma",
            "the term percent",
            "mister",
            "the word ampersand",
            "the label plus",
            "the word slash",
            "colon",
            "the term dash",
            "question mark",
            "the word bracket",
            "underscore",
            "the label dot",
            "hyphen",
            "the phrase at sign",
            "the word semicolon",
            "the label star",
            "the term pipe",
        ),
        "validation": (
            "the word quotation mark",
            "the term apostrophe",
            "the label backslash",
            "the word brace",
            "the phrase equal sign",
            "the term caret",
            "the label percent sign",
            "the word parenthesis",
            "the term ellipsis",
        ),
    },
    "malformed": {
        "train": (
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
        ),
        "validation": (
            "the price stopped at forty point",
            "the phone line starts with plus",
            "the appointment falls on october",
            "the time ends at quarter past",
            "the domain stops at sample dot",
            "the count was twelve and",
            "the date ends after march",
            "the code begins with triple",
            "the measurement ends at five point",
        ),
    },
    "already": {
        "train": (
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
        ),
        "validation": (
            "the receipt shows 08:40 a.m.",
            "the reading is 1.25",
            "the reference is 712",
            "the contact number is 415-555-0138",
            "the address is sample.org",
            "the schedule says July 9",
            "the label is mr. lee",
            "the total is $18.20",
            "the form lists 24 items",
        ),
    },
    "ordinary": {
        "train": (
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
        ),
        "validation": (
            "the billing note mentions a refund",
            "the member changed the delivery address",
            "the customer requested a phone call",
            "the service window moved to Friday",
            "the case is waiting for approval",
            "the clerk attached a photo",
            "the order needs a replacement label",
            "the visitor left a message with security",
            "the team approved the new route",
        ),
    },
}

NEGATIVE_DETAILS = {
    "train": (
        ("handbook", "the opening section"),
        ("style guide", "the punctuation table"),
        ("catalog", "the product notes"),
        ("archive", "the old correspondence"),
        ("manual", "the keyboard diagram"),
        ("invoice", "the payment paragraph"),
        ("transcript", "the caller's quotation"),
        ("reference card", "the examples column"),
        ("draft", "the heading"),
        ("support log", "the first comment"),
        ("checklist", "the handoff note"),
        ("editorial file", "the margin note"),
        ("training memo", "the second example"),
        ("account history", "the attached message"),
        ("review sheet", "the closing paragraph"),
        ("printed form", "the instructions"),
        ("meeting notes", "the action item"),
        ("delivery record", "the driver's remark"),
        ("customer letter", "the quoted sentence"),
        ("website copy", "the navigation text"),
        ("case summary", "the background section"),
        ("product sheet", "the feature list"),
        ("office note", "the reminder line"),
        ("mailing label", "the address block"),
        ("service report", "the final observation"),
    ),
    "validation": (
        ("museum guide", "the exhibit note"),
        ("pharmacy card", "the dosage example"),
        ("travel leaflet", "the route description"),
        ("visitor log", "the arrival comment"),
        ("dispatch sheet", "the driver note"),
    ),
}

NEGATIVE_FRAMES = {
    "literal": (
        "The {document} uses {term} in {section}.",
        "In {section}, the {document} includes {term}.",
        "The {section} of the {document} mentions {term}.",
        "A line about {term} appears in {section} of the {document}.",
        "The {document} puts {term} under {section}.",
        "The editor found {term} in {section} of the {document}.",
        "The {document} quotes {term} beside {section}.",
        "Under {section}, the {document} names {term}.",
    ),
    "malformed": (
        "The {document} ends after the caller says {term} in {section}.",
        "The recording for {document} stops at {term} near {section}.",
        "In {section}, the caller leaves the {document} saying {term}.",
        "The {document} cuts off with {term} in the {section}.",
        "The last line of {section} has the caller say {term} in the {document}.",
        "The {document} trails off at {term} beside {section}.",
        "The caller's {document} note ends with {term} under {section}.",
        "At {section}, the {document} stops after {term}.",
    ),
    "already": (
        "The {document} records this line in {section}: {term}.",
        "In {section}, the {document} prints this line: {term}.",
        "The {section} of the {document} contains this line: {term}.",
        "A printed {document} gives this line beside {section}: {term}.",
        "The {document} has this line under {section}: {term}.",
        "The clerk copied this line into {section} of the {document}: {term}.",
        "The {document} displays this line near {section}: {term}.",
        "At {section}, the {document} records this line: {term}.",
    ),
    "ordinary": (
        "The {document} mentions that {term} while discussing {section}.",
        "During {section}, the {document} notes that {term}.",
        "The {section} of the {document} says that {term}.",
        "A comment in the {document} says {term} during {section}.",
        "The {document} explains that {term} beside {section}.",
        "The clerk wrote that {term} in the {section} of the {document}.",
        "The {document} links {term} with the discussion in {section}.",
        "While reading {section}, the team heard that {term}.",
    ),
}


def record_number(record_id: str) -> int:
    match = re.search(r"(\d+)$", record_id)
    if match is None:
        raise ValueError(f"record ID has no numeric suffix: {record_id}")
    return int(match.group(1))


def repair_spans(record: dict[str, Any]) -> list[dict[str, Any]]:
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


def scenario(split: str, ordinal: int, salt: int = 0) -> dict[str, str]:
    ordinal += salt * 100_003
    people, places, objects = PEOPLE[split], PLACES[split], OBJECTS[split]
    person = people[ordinal % len(people)]
    place = places[(ordinal // len(people)) % len(places)]
    object_name = objects[(ordinal // (len(people) * len(places))) % len(objects)]
    subjects = LEAD_SUBJECTS[split]
    subject = subjects[(ordinal + salt) % len(subjects)]
    verb = LEAD_VERBS[(ordinal // len(subjects) + salt) % len(LEAD_VERBS)]
    location = LEAD_LOCATIONS[
        (ordinal // (len(subjects) * len(LEAD_VERBS)) + salt) % len(LEAD_LOCATIONS)
    ]
    return {
        "person": person,
        "place": place,
        "object": object_name,
        "lead": (
            subject.format(person=person, place=place, object=object_name)
            + " "
            + verb
            + " "
            + location.format(person=person, place=place, object=object_name)
        ),
    }


def measurement_context(
    source: str, value: str, s: dict[str, str], ordinal: int
) -> str:
    if source.endswith(" hours"):
        choices = (
            "The inspection lasts {value} from opening to close",
            "Allow {value} for the repair at {place}",
            "The appointment takes {value} before {person} leaves",
            "The class runs for {value} near {place}",
            "The service window spans {value} on the schedule",
            "The journey takes {value} if traffic stays light",
        )
    elif source.endswith((" grams", " kilograms")):
        choices = (
            "The parcel weighs {value} before sealing",
            "The sample contains {value} after drying",
            "The shipment is {value} at the loading dock",
            "Pack {value} with the replacement parts",
            "The crate holds {value} for the delivery",
            "The kitchen portion weighs {value} for {person}",
        )
    else:
        choices = (
            "The trail covers {value} beyond {place}",
            "The panel measures {value} along the outer edge",
            "The route extends {value} past the station",
            "The sample spans {value} beside the {object}",
            "The fence runs {value} around the site",
            "The gap between the doors is {value}",
        )
    return choices[ordinal % len(choices)].format(value=value, **s)


def whitelist_context(value: str, s: dict[str, str], ordinal: int) -> str:
    lower = value.lower()
    if any(title in lower for title in ("doctor", "mister", "misses", "saint")):
        choices = (
            "The contact name on the visitor list is {value}",
            "Ask for {value} at the front desk",
            "The badge should be printed for {value}",
            "{value} is the person handling the handoff",
            "Send the invitation to {value} at {place}",
        )
    elif any(term in lower for term in ("five hundred", "seven eleven", "p c i e")):
        choices = (
            "The product listed for {person} is {value}",
            "Look for {value} on the equipment shelf",
            "The store entry for {value} is near {place}",
            "The compatibility note mentions {value}",
            "The catalog puts {value} beside the {object}",
        )
    else:
        choices = (
            "The device option selected by {person} is {value}",
            "The account label for the {object} reads {value}",
            "The technician marked {value} on the package",
            "The model listed at {place} is {value}",
            "The parts list includes {value} for the repair",
        )
    return choices[ordinal % len(choices)].format(value=value, **s)


def clause(kind: str, source: str, value: str, s: dict[str, str], ordinal: int) -> str:
    if kind == "MEASUREMENT":
        return measurement_context(source, value, s, ordinal)
    if kind == "WHITELIST":
        return whitelist_context(source, s, ordinal).replace(source, value, 1)
    choices = ROLE_FREE_CONTEXTS[kind]
    return choices[ordinal % len(choices)].format(value=value, **s)


def multi_clause(kind: str, value: str, s: dict[str, str], ordinal: int) -> str:
    choices = MULTI_CONTEXTS[kind]
    rendered = choices[ordinal % len(choices)].format(value=value, **s)
    if rendered[0].isupper() and not rendered.startswith((s["person"], "I ")):
        rendered = rendered[0].lower() + rendered[1:]
    return rendered


def render(template: str, sources: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text = template
    offsets = []
    for index, source in enumerate(sources):
        marker = f"__SPAN_{index}__"
        start = text.index(marker)
        text = text.replace(marker, source, 1)
        offsets.append((start, start + len(source)))
    return text, offsets


def positive_text(
    record: dict[str, Any], spans: list[dict[str, Any]], ordinal: int
) -> tuple[str, list[tuple[int, int]]]:
    split = record["split"]
    family = record["template_family"]
    sources = [span["source"] for span in spans]
    if len(spans) == 1 and (
        family.startswith("whole_") or (split == "train" and ordinal < 19)
    ):
        return render("__SPAN_0__", sources)

    scenario_salt = len(spans) + record_number(record["id"]) % 997
    s = scenario(split, ordinal, scenario_salt)
    values = [f"__SPAN_{index}__" for index in range(len(spans))]
    if len(spans) > 1:
        clauses = [
            multi_clause(span["kind"], value, s, ordinal + index * 3)
            for index, (span, value) in enumerate(zip(spans, values, strict=True))
        ]
    else:
        clauses = [
            clause(span["kind"], span["source"], value, s, ordinal)
            for span, value in zip(spans, values, strict=True)
        ]
    if len(spans) == 1:
        text = f"{s['lead']}: {clauses[0]}."
        if len(tokenize(text)) > MAX_SOURCE_TOKENS:
            text = f"{s['person']} says: {clauses[0]}."
        return render(text, sources)

    connectors = MULTI_CONNECTORS[len(spans)]
    template = connectors[ordinal % len(connectors)].format(
        lead=s["lead"], **dict(zip("abcd"[: len(clauses)], clauses, strict=True))
    )
    text, offsets = render(template, sources)
    if len(tokenize(text)) > MAX_SOURCE_TOKENS:
        compact = {
            "DATE": "meet {value}, {person}",
            "TIME": "leave {value}, {person}",
            "MONEY": "pay {value} for {person}",
            "DECIMAL": "set {value} for {person}",
            "PHONE": "call {value} for {person}",
            "ELECTRONIC": "email {value} for {person}",
            "DIGIT_SEQUENCE": "use {value} for {person}",
            "CARDINAL": "send {value} to {person}",
            "ORDINAL": "take {value} for {person}",
            "PUNCTUATION": "add {value} for {person}",
            "MEASUREMENT": "use {value} for {person}",
            "WHITELIST": "use {value} for {person}",
            "WORD": "use {value} for {person}",
        }
        compact_clauses = [
            compact[span["kind"]].format(value=value, **s)
            if span["kind"] in compact
            else clause(span["kind"], span["source"], value, s, ordinal + index)
            for index, (span, value) in enumerate(zip(spans, values, strict=True))
        ]
        compact_text = f"{'; '.join(compact_clauses)}."
        return render(compact_text, sources)
    return text, offsets


def negative_text(split: str, category: str, ordinal: int) -> str:
    terms = NEGATIVE_TERMS[category][split]
    details = NEGATIVE_DETAILS[split]
    term = terms[ordinal % len(terms)]
    document, section = details[(ordinal // len(terms)) % len(details)]
    frames = NEGATIVE_FRAMES[category]
    frame = frames[(ordinal // len(terms)) % len(frames)]
    return frame.format(document=document, section=section, term=term)


def rebuild_derived_fields(text: str, spans: list[dict[str, Any]]) -> dict[str, Any]:
    expected = text
    for span in reversed(spans):
        assert text[span["start"] : span["end"]] == span["source"]
        expected = (
            expected[: span["start"]] + span["replacement"] + expected[span["end"] :]
        )

    tokens = tokenize(text)
    labels = ["O"] * len(tokens)
    previous_end = -1
    for span in spans:
        assert span["start"] >= previous_end
        covered = [
            index
            for index, token in enumerate(tokens)
            if token.start >= span["start"] and token.end <= span["end"]
        ]
        assert covered
        assert tokens[covered[0]].start == span["start"]
        assert tokens[covered[-1]].end == span["end"]
        for offset, index in enumerate(covered):
            assert labels[index] == "O"
            labels[index] = ("B-" if offset == 0 else "I-") + span["kind"]
        previous_end = span["end"]

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


def repair_record(
    record: dict[str, Any],
    negative_ordinals: dict[tuple[str, str], int],
    positive_ordinals: dict[tuple[str, int], int],
) -> dict[str, Any]:
    spans = repair_spans(record)
    split = record["split"]
    family = record["template_family"]
    if spans:
        key = (split, len(spans))
        ordinal = positive_ordinals[key]
        positive_ordinals[key] += 1
        text, offsets = positive_text(record, spans, ordinal)
        for span, (start, end) in zip(spans, offsets, strict=True):
            span["start"] = start
            span["end"] = end
    else:
        category = family.rsplit("_", 1)[-1]
        ordinal = negative_ordinals[(split, category)]
        negative_ordinals[(split, category)] += 1
        text = negative_text(split, category, ordinal)

    derived = rebuild_derived_fields(text, spans)
    repaired = dict(record)
    repaired.update({"text": text, "spans": spans, **derived})
    provenance = repaired.get("provenance", "synthetic")
    if "dataset_cleanup_v2" not in provenance:
        repaired["provenance"] = provenance + PROVENANCE_SUFFIX
    return repaired


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/generated/records.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/generated/records.jsonl")
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
