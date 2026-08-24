import json
from pathlib import Path

import pytest

from premove_itn.dataset.enrichment.sgd_context import (
    SgdContextError,
    SgdSlotSpan,
    SgdUserTurn,
    iter_sgd_train_user_turns,
)


def _write_train(
    root: Path,
    dialogues: list[dict[str, object]],
    *,
    is_categorical: bool = False,
) -> Path:
    train = root / "train"
    train.mkdir()
    (train / "schema.json").write_text(
        json.dumps(
            [
                {
                    "service_name": "Restaurants_1",
                    "slots": [
                        {
                            "name": "time",
                            "description": "Time of the restaurant reservation",
                            "is_categorical": is_categorical,
                        }
                    ],
                }
            ]
        )
    )
    (train / "dialogues_001.json").write_text(json.dumps(dialogues))
    return train


def _dialogue(
    *,
    dialogue_id: str = "1_00000",
    utterance: str = "Book it at 7:30 pm.",
    slots: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "dialogue_id": dialogue_id,
        "turns": [
            {
                "speaker": "SYSTEM",
                "utterance": "What time?",
                "frames": [{"service": "Restaurants_1", "slots": []}],
            },
            {
                "speaker": "USER",
                "utterance": utterance,
                "frames": [
                    {
                        "service": "Restaurants_1",
                        "slots": (
                            [{"slot": "time", "start": 11, "exclusive_end": 18}]
                            if slots is None
                            else slots
                        ),
                    }
                ],
            },
        ],
    }


def test_iterates_user_turns_with_non_categorical_spans(tmp_path: Path) -> None:
    train = _write_train(tmp_path, [_dialogue()])

    turns = list(iter_sgd_train_user_turns(train))

    assert turns == [
        SgdUserTurn(
            source_file="train/dialogues_001.json",
            dialogue_id="1_00000",
            turn_index=1,
            utterance="Book it at 7:30 pm.",
            services=("Restaurants_1",),
            spans=(
                SgdSlotSpan(
                    service="Restaurants_1",
                    slot="time",
                    description="Time of the restaurant reservation",
                    start=11,
                    end=18,
                    value="7:30 pm",
                ),
            ),
        )
    ]


def test_includes_context_only_user_turns(tmp_path: Path) -> None:
    train = _write_train(tmp_path, [_dialogue(utterance="Yes please.", slots=[])])

    [turn] = iter_sgd_train_user_turns(train)

    assert turn.utterance == "Yes please."
    assert turn.spans == ()


def test_preserves_overlapping_sgd_spans(tmp_path: Path) -> None:
    utterance = "Watch it at Century at Hayward."
    train = _write_train(
        tmp_path,
        [
            _dialogue(
                utterance=utterance,
                slots=[
                    {"slot": "time", "start": 12, "exclusive_end": 30},
                    {"slot": "time", "start": 23, "exclusive_end": 30},
                ],
            )
        ],
    )

    [turn] = iter_sgd_train_user_turns(train)

    assert [(span.start, span.end, span.value) for span in turn.spans] == [
        (12, 30, "Century at Hayward"),
        (23, 30, "Hayward"),
    ]


@pytest.mark.parametrize(
    ("slot", "match"),
    [
        ({"slot": "unknown", "start": 0, "exclusive_end": 4}, "unknown or categorical"),
        ({"slot": "time", "start": -1, "exclusive_end": 4}, "invalid offsets"),
        ({"slot": "time", "start": 0, "exclusive_end": 99}, "invalid offsets"),
        ({"slot": "time", "start": 4, "exclusive_end": 4}, "invalid offsets"),
    ],
)
def test_rejects_invalid_slot_annotations(
    tmp_path: Path, slot: dict[str, object], match: str
) -> None:
    train = _write_train(tmp_path, [_dialogue(utterance="time", slots=[slot])])

    with pytest.raises(SgdContextError, match=match):
        list(iter_sgd_train_user_turns(train))


def test_rejects_categorical_span_annotations(tmp_path: Path) -> None:
    train = _write_train(tmp_path, [_dialogue()], is_categorical=True)

    with pytest.raises(SgdContextError, match="unknown or categorical"):
        list(iter_sgd_train_user_turns(train))


def test_rejects_unknown_frame_service_without_spans(tmp_path: Path) -> None:
    dialogue = _dialogue(utterance="Yes please.", slots=[])
    dialogue["turns"][1]["frames"][0]["service"] = "Unknown_1"  # type: ignore[index]
    train = _write_train(tmp_path, [dialogue])

    with pytest.raises(SgdContextError, match="unknown service"):
        list(iter_sgd_train_user_turns(train))


def test_rejects_non_train_split(tmp_path: Path) -> None:
    dev = tmp_path / "dev"
    dev.mkdir()

    with pytest.raises(SgdContextError, match="restricted to train"):
        list(iter_sgd_train_user_turns(dev))


def test_rejects_duplicate_dialogue_ids_across_files(tmp_path: Path) -> None:
    train = _write_train(tmp_path, [_dialogue()])
    (train / "dialogues_002.json").write_text(json.dumps([_dialogue()]))

    with pytest.raises(SgdContextError, match="duplicates dialogue_id"):
        list(iter_sgd_train_user_turns(train))
