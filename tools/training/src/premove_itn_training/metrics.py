from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from premove_itn_training.labels import ID_TO_LABEL, TRAINED_KINDS

_IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class WordSpan:
    start: int
    end: int
    kind: str


def extract_spans(label_ids: Sequence[int]) -> tuple[WordSpan, ...]:
    """Decode BIO IDs into strict word-coordinate spans with deterministic repair."""
    spans: list[WordSpan] = []
    active_start: int | None = None
    active_kind: str | None = None

    def close(end: int) -> None:
        nonlocal active_start, active_kind
        if active_start is not None and active_kind is not None:
            spans.append(WordSpan(active_start, end, active_kind))
        active_start = None
        active_kind = None

    for index, label_id in enumerate(label_ids):
        try:
            label = ID_TO_LABEL[label_id]
        except KeyError as error:
            raise ValueError(f"unknown Model V1 label ID: {label_id}") from error
        if label == "O":
            close(index)
            continue

        prefix, kind = label.split("-", 1)
        if prefix == "B" or active_kind != kind:
            close(index)
            active_start = index
            active_kind = kind
    close(len(label_ids))
    return tuple(spans)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(
    true_positive: int, false_positive: int, false_negative: int
) -> tuple[float, float, float]:
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return precision, recall, f1


def score_word_sequences(
    predicted_sequences: Sequence[Sequence[int]],
    gold_sequences: Sequence[Sequence[int]],
) -> dict[str, float]:
    """Score word-level BIO predictions using exact span boundaries and kinds."""
    if len(predicted_sequences) != len(gold_sequences):
        raise ValueError("prediction and gold sequence counts must match")

    true_positive: Counter[str] = Counter()
    false_positive: Counter[str] = Counter()
    false_negative: Counter[str] = Counter()
    sentence_exact = 0
    context_only_total = 0
    context_only_exact = 0
    multi_span_total = 0
    multi_span_exact = 0

    for predicted, gold in zip(predicted_sequences, gold_sequences, strict=True):
        if len(predicted) != len(gold):
            raise ValueError("prediction and gold word counts must match")
        predicted_tuple = tuple(predicted)
        gold_tuple = tuple(gold)
        predicted_spans = set(extract_spans(predicted_tuple))
        gold_spans = set(extract_spans(gold_tuple))

        exact = predicted_tuple == gold_tuple
        sentence_exact += exact
        if not gold_spans:
            context_only_total += 1
            context_only_exact += exact
        if len(gold_spans) > 1:
            multi_span_total += 1
            multi_span_exact += exact

        for span in predicted_spans & gold_spans:
            true_positive[span.kind] += 1
        for span in predicted_spans - gold_spans:
            false_positive[span.kind] += 1
        for span in gold_spans - predicted_spans:
            false_negative[span.kind] += 1

    total_tp = sum(true_positive.values())
    total_fp = sum(false_positive.values())
    total_fn = sum(false_negative.values())
    precision, recall, f1 = _prf(total_tp, total_fp, total_fn)
    record_count = len(gold_sequences)
    metrics = {
        "span_precision": precision,
        "span_recall": recall,
        "span_f1": f1,
        "sentence_exact_bio_accuracy": _ratio(sentence_exact, record_count),
        "context_only_exact_accuracy": _ratio(
            context_only_exact, context_only_total
        ),
        "multi_span_sentence_exact_accuracy": _ratio(
            multi_span_exact, multi_span_total
        ),
    }
    for kind in TRAINED_KINDS:
        kind_precision, kind_recall, kind_f1 = _prf(
            true_positive[kind], false_positive[kind], false_negative[kind]
        )
        metrics[f"{kind}_precision"] = kind_precision
        metrics[f"{kind}_recall"] = kind_recall
        metrics[f"{kind}_f1"] = kind_f1
    return metrics


def score_aligned_predictions(
    prediction_ids: Any,
    aligned_gold_ids: Any,
) -> dict[str, float]:
    """Reduce first-subword predictions to words, then compute strict metrics."""
    predictions = prediction_ids.tolist()
    gold = aligned_gold_ids.tolist()
    if len(predictions) != len(gold):
        raise ValueError("prediction and gold batch sizes must match")

    word_predictions: list[list[int]] = []
    word_gold: list[list[int]] = []
    for predicted_row, gold_row in zip(predictions, gold, strict=True):
        if len(predicted_row) != len(gold_row):
            raise ValueError("prediction and gold subword counts must match")
        kept = [
            (predicted, expected)
            for predicted, expected in zip(predicted_row, gold_row, strict=True)
            if expected != _IGNORE_INDEX
        ]
        word_predictions.append([predicted for predicted, _ in kept])
        word_gold.append([expected for _, expected in kept])
    return score_word_sequences(word_predictions, word_gold)
