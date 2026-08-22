import json
from collections import Counter
from pathlib import Path

from premove_itn.labels import BIO_LABELS
from premove_itn.tokenize import tokenize

DATASET_PATH = Path(__file__).parents[1] / "data/generated/records.jsonl"
NEGATIVE_META_CUES = (
    "no normalization",
    "don't normalize",
    "leave that wording literal",
    "dictation command",
    "itn edit",
)


def _load_records() -> list[dict[str, object]]:
    with DATASET_PATH.open() as handle:
        return [json.loads(line) for line in handle]


def test_generated_dataset_contract_and_cleanup_invariants() -> None:
    records = _load_records()
    assert len(records) == 10_000
    assert len({record["id"] for record in records}) == len(records)
    assert Counter(record["split"] for record in records) == {
        "train": 9_000,
        "validation": 1_000,
    }
    assert not (
        {record["template_family"] for record in records if record["split"] == "train"}
        & {
            record["template_family"]
            for record in records
            if record["split"] == "validation"
        }
    )

    train_cores: set[str] = set()
    validation_cores: set[str] = set()
    for record in records:
        text = record["text"]
        spans = record["spans"]
        tokens = tokenize(text)
        assert len(tokens) <= 48
        assert len(tokens) == len(record["tokens"]) == len(record["bio_labels"])
        assert all(
            stored == {"text": token.text, "start": token.start, "end": token.end}
            for stored, token in zip(record["tokens"], tokens, strict=True)
        )

        expected = text
        previous_start = len(text) + 1
        labels = ["O"] * len(tokens)
        for span in reversed(spans):
            assert 0 <= span["start"] < span["end"] <= len(text)
            assert span["end"] <= previous_start
            assert text[span["start"] : span["end"]] == span["source"]
            expected = (
                expected[: span["start"]]
                + span["replacement"]
                + expected[span["end"] :]
            )
            previous_start = span["start"]

        for span in spans:
            covered = [
                index
                for index, token in enumerate(tokens)
                if token.start >= span["start"] and token.end <= span["end"]
            ]
            assert covered
            assert tokens[covered[0]].start == span["start"]
            assert tokens[covered[-1]].end == span["end"]
            for offset, index in enumerate(covered):
                labels[index] = ("B-" if offset == 0 else "I-") + span["kind"]

        assert expected == record["expected_text"]
        assert all(label in BIO_LABELS for label in record["bio_labels"])
        assert labels == record["bio_labels"]

        if not spans:
            lowered = text.lower()
            assert not any(cue in lowered for cue in NEGATIVE_META_CUES)
        else:
            core = text
            for span in reversed(spans):
                core = core[: span["start"]] + f"<{span['kind']}>" + core[span["end"] :]
            if record["split"] == "train":
                train_cores.add(core)
            else:
                validation_cores.add(core)

        if "room_measure" in record["template_family"]:
            assert all(span["source"].endswith(" meters") for span in spans)
        if "parcel weighs" in text:
            assert all(not span["source"].endswith(" hours") for span in spans)

    assert not train_cores & validation_cores
