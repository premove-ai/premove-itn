from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DIALOGUE_FILE_PATTERN = re.compile(r"dialogues_[0-9]+\.json")


class SgdContextError(ValueError):
    """Raised when an SGD train artifact violates the context contract."""


@dataclass(frozen=True, slots=True)
class SgdSlotSpan:
    service: str
    slot: str
    description: str
    start: int
    end: int
    value: str


@dataclass(frozen=True, slots=True)
class SgdUserTurn:
    source_file: str
    dialogue_id: str
    turn_index: int
    utterance: str
    services: tuple[str, ...]
    spans: tuple[SgdSlotSpan, ...]


@dataclass(frozen=True, slots=True)
class _SgdSchemaIndex:
    services: frozenset[str]
    non_categorical_slot_descriptions: dict[tuple[str, str], str]


def _require_mapping(value: object, *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SgdContextError(f"{location} must be a JSON object")
    return value


def _require_list(value: object, *, location: str) -> list[object]:
    if not isinstance(value, list):
        raise SgdContextError(f"{location} must be a JSON array")
    return value


def _require_string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise SgdContextError(f"{location} must be a non-empty string")
    return value


def _require_int(value: object, *, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SgdContextError(f"{location} must be an integer")
    return value


def _load_json(path: Path) -> object:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise SgdContextError(
            f"{path}:{error.lineno}:{error.colno}: invalid JSON"
        ) from error


def _load_schema_index(schema_path: Path) -> _SgdSchemaIndex:
    raw_services = _require_list(_load_json(schema_path), location=f"{schema_path}")
    known_services: set[str] = set()
    known_slots: set[tuple[str, str]] = set()
    non_categorical_slot_descriptions: dict[tuple[str, str], str] = {}

    for service_index, raw_service in enumerate(raw_services):
        location = f"{schema_path}:service[{service_index}]"
        service = _require_mapping(raw_service, location=location)
        service_name = _require_string(
            service.get("service_name"), location=f"{location}.service_name"
        )
        if service_name in known_services:
            raise SgdContextError(
                f"{location}.service_name duplicates {service_name!r}"
            )
        known_services.add(service_name)

        raw_slots = _require_list(service.get("slots"), location=f"{location}.slots")
        for slot_index, raw_slot in enumerate(raw_slots):
            slot_location = f"{location}.slots[{slot_index}]"
            slot = _require_mapping(raw_slot, location=slot_location)
            slot_name = _require_string(
                slot.get("name"), location=f"{slot_location}.name"
            )
            key = (service_name, slot_name)
            if key in known_slots:
                raise SgdContextError(f"{slot_location}.name duplicates {slot_name!r}")
            known_slots.add(key)

            is_categorical = slot.get("is_categorical")
            if not isinstance(is_categorical, bool):
                raise SgdContextError(
                    f"{slot_location}.is_categorical must be a boolean"
                )
            if not is_categorical:
                non_categorical_slot_descriptions[key] = _require_string(
                    slot.get("description"),
                    location=f"{slot_location}.description",
                )

    return _SgdSchemaIndex(frozenset(known_services), non_categorical_slot_descriptions)


def _parse_user_turn(
    raw_turn: Mapping[str, Any],
    *,
    path: Path,
    dialogue_id: str,
    turn_index: int,
    schema: _SgdSchemaIndex,
) -> SgdUserTurn:
    location = f"{path}:{dialogue_id}:turn[{turn_index}]"
    utterance = _require_string(
        raw_turn.get("utterance"), location=f"{location}.utterance"
    )
    raw_frames = _require_list(raw_turn.get("frames"), location=f"{location}.frames")

    services: list[str] = []
    spans: list[SgdSlotSpan] = []
    for frame_index, raw_frame in enumerate(raw_frames):
        frame_location = f"{location}.frames[{frame_index}]"
        frame = _require_mapping(raw_frame, location=frame_location)
        service = _require_string(
            frame.get("service"), location=f"{frame_location}.service"
        )
        if service not in schema.services:
            raise SgdContextError(
                f"{frame_location}.service references unknown service {service!r}"
            )
        if service not in services:
            services.append(service)

        raw_spans = _require_list(
            frame.get("slots"), location=f"{frame_location}.slots"
        )
        for span_index, raw_span in enumerate(raw_spans):
            span_location = f"{frame_location}.slots[{span_index}]"
            span = _require_mapping(raw_span, location=span_location)
            slot = _require_string(span.get("slot"), location=f"{span_location}.slot")
            key = (service, slot)
            if key not in schema.non_categorical_slot_descriptions:
                raise SgdContextError(
                    f"{span_location} references unknown or categorical slot "
                    f"{service}.{slot}"
                )

            start = _require_int(span.get("start"), location=f"{span_location}.start")
            end = _require_int(
                span.get("exclusive_end"),
                location=f"{span_location}.exclusive_end",
            )
            if start < 0 or end <= start or end > len(utterance):
                raise SgdContextError(
                    f"{span_location} has invalid offsets [{start}, {end}) "
                    f"for an utterance of length {len(utterance)}"
                )
            spans.append(
                SgdSlotSpan(
                    service=service,
                    slot=slot,
                    description=schema.non_categorical_slot_descriptions[key],
                    start=start,
                    end=end,
                    value=utterance[start:end],
                )
            )

    return SgdUserTurn(
        source_file=f"train/{path.name}",
        dialogue_id=dialogue_id,
        turn_index=turn_index,
        utterance=utterance,
        services=tuple(services),
        spans=tuple(spans),
    )


def iter_sgd_train_user_turns(train_directory: Path) -> Iterator[SgdUserTurn]:
    """Yield validated SGD USER turns in deterministic source order.

    The directory must be the original SGD train split. Context-only USER turns
    are included. Slot spans are restricted to schema-declared non-categorical
    slots and retain SGD's character offsets. Overlapping spans are preserved
    because ontology mapping happens later.
    """
    if train_directory.name != "train":
        raise SgdContextError("SGD context extraction is restricted to train/")
    if not train_directory.is_dir():
        raise FileNotFoundError(
            f"SGD train directory does not exist: {train_directory}"
        )

    schema_path = train_directory / "schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"SGD train schema does not exist: {schema_path}")
    schema = _load_schema_index(schema_path)

    dialogue_paths = sorted(
        path
        for path in train_directory.iterdir()
        if path.is_file() and _DIALOGUE_FILE_PATTERN.fullmatch(path.name)
    )
    if not dialogue_paths:
        raise SgdContextError(
            f"SGD train directory has no dialogue files: {train_directory}"
        )

    seen_dialogue_ids: set[str] = set()
    for path in dialogue_paths:
        raw_dialogues = _require_list(_load_json(path), location=f"{path}")
        for dialogue_index, raw_dialogue in enumerate(raw_dialogues):
            location = f"{path}:dialogue[{dialogue_index}]"
            dialogue = _require_mapping(raw_dialogue, location=location)
            dialogue_id = _require_string(
                dialogue.get("dialogue_id"), location=f"{location}.dialogue_id"
            )
            if dialogue_id in seen_dialogue_ids:
                raise SgdContextError(
                    f"{location} duplicates dialogue_id {dialogue_id!r}"
                )
            seen_dialogue_ids.add(dialogue_id)

            raw_turns = _require_list(
                dialogue.get("turns"), location=f"{location}.turns"
            )
            for turn_index, raw_turn in enumerate(raw_turns):
                turn_location = f"{location}.turns[{turn_index}]"
                turn = _require_mapping(raw_turn, location=turn_location)
                speaker = _require_string(
                    turn.get("speaker"), location=f"{turn_location}.speaker"
                )
                if speaker == "SYSTEM":
                    continue
                if speaker != "USER":
                    raise SgdContextError(
                        f"{turn_location}.speaker must be USER or SYSTEM"
                    )
                yield _parse_user_turn(
                    turn,
                    path=path,
                    dialogue_id=dialogue_id,
                    turn_index=turn_index,
                    schema=schema,
                )
