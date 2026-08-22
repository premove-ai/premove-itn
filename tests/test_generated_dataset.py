import json
import re
from collections import Counter
from pathlib import Path

from premove_itn.labels import BIO_LABELS, SpanKind
from premove_itn.tokenize import tokenize

DATASET_PATH = Path(__file__).parents[1] / "data/generated/records.jsonl"
GOLDEN_PATH = Path(__file__).parents[1] / "data/golden.json"
MAX_POSITIVE_CORE_FREQUENCY = 50
MAX_FIVE_GRAM_FREQUENCY = 120
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


def _redacted_core(record: dict[str, object], include_kind: bool = False) -> str:
    text = record["text"]
    for span in reversed(record["spans"]):
        marker = f"<{span['kind']}>" if include_kind else "<SPAN>"
        text = text[: span["start"]] + marker + text[span["end"] :]
    return text.lower()


def _span_skeleton(record: dict[str, object]) -> tuple[str, ...]:
    spans = record["spans"]
    return tuple(
        "<SPAN>"
        if any(
            token.start >= span["start"] and token.end <= span["end"] for span in spans
        )
        else token.text.lower()
        for token in tokenize(record["text"])
    )


def _five_grams(record: dict[str, object]) -> list[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+|<span>", _redacted_core(record))
    return [tuple(words[index : index + 5]) for index in range(max(0, len(words) - 4))]


def test_generated_dataset_contract_and_cleanup_invariants() -> None:
    records = _load_records()
    assert len(records) == 10_000
    assert len({record["id"] for record in records}) == len(records)
    assert len({record["text"] for record in records}) == len(records)
    assert len(
        {
            json.dumps(
                {"text": record["text"], "spans": record["spans"]},
                sort_keys=True,
            )
            for record in records
        }
    ) == len(records)
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
    positive_core_counts = {"train": Counter(), "validation": Counter()}
    five_gram_counts: Counter[tuple[str, ...]] = Counter()
    observed_kinds: set[str] = set()
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
            observed_kinds.add(span["kind"])
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
            positive_core_counts[record["split"]][core] += 1

        five_gram_counts.update(_five_grams(record))

        if "room_measure" in record["template_family"]:
            assert all(span["source"].endswith(" meters") for span in spans)
        if "parcel weighs" in text:
            assert all(not span["source"].endswith(" hours") for span in spans)

    assert not train_cores & validation_cores
    assert observed_kinds == {kind.value for kind in SpanKind}
    assert all(
        max(counts.values(), default=0) <= MAX_POSITIVE_CORE_FREQUENCY
        for counts in positive_core_counts.values()
    )
    assert max(five_gram_counts.values(), default=0) <= MAX_FIVE_GRAM_FREQUENCY

    golden = json.loads(GOLDEN_PATH.read_text())
    generated_texts = {record["text"] for record in records}
    golden_texts = {record["text"] for record in golden}
    assert not generated_texts & golden_texts
    assert not {_redacted_core(record, include_kind=True) for record in records} & {
        _redacted_core(record, include_kind=True) for record in golden
    }
    assert not {_span_skeleton(record) for record in records} & {
        _span_skeleton(record) for record in golden
    }
