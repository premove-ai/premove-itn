from premove_itn_training.labels import LABEL_TO_ID
from premove_itn_training.metrics import (
    extract_spans,
    score_aligned_predictions,
    score_word_sequences,
)


def _ids(*labels: str) -> list[int]:
    return [LABEL_TO_ID[label] for label in labels]


class _Rows:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    def tolist(self) -> list[list[int]]:
        return self.rows


def test_extract_spans_repairs_malformed_inside_label_deterministically() -> None:
    assert extract_spans(_ids("O", "I-TIME", "I-TIME", "B-DATE")) == (
        extract_spans(_ids("O", "B-TIME", "I-TIME", "B-DATE"))
    )


def test_strict_span_metrics_require_matching_boundaries_and_kind() -> None:
    gold = [
        _ids("O", "B-DIGIT_SEQUENCE", "I-DIGIT_SEQUENCE"),
        _ids("O", "B-TIME", "I-TIME"),
    ]
    predictions = [
        _ids("O", "B-TIME", "I-TIME"),
        _ids("O", "O", "B-TIME"),
    ]

    metrics = score_word_sequences(predictions, gold)

    assert metrics["span_precision"] == 0.0
    assert metrics["span_recall"] == 0.0
    assert metrics["span_f1"] == 0.0
    assert metrics["sentence_exact_bio_accuracy"] == 0.0


def test_metrics_report_context_only_and_multi_span_exact_accuracy() -> None:
    gold = [
        _ids("O", "O"),
        _ids("B-DATE", "O", "B-MONEY"),
        _ids("B-TIME", "I-TIME"),
    ]
    predictions = [
        _ids("O", "O"),
        _ids("B-DATE", "O", "B-MONEY"),
        _ids("B-TIME", "I-TIME"),
    ]

    metrics = score_word_sequences(predictions, gold)

    assert metrics["span_precision"] == 1.0
    assert metrics["span_recall"] == 1.0
    assert metrics["span_f1"] == 1.0
    assert metrics["sentence_exact_bio_accuracy"] == 1.0
    assert metrics["context_only_exact_accuracy"] == 1.0
    assert metrics["multi_span_sentence_exact_accuracy"] == 1.0


def test_aligned_metrics_ignore_special_padding_and_continuation_subwords() -> None:
    predictions = _Rows(
        [[0, LABEL_TO_ID["B-TIME"], LABEL_TO_ID["O"], 999, 999]]
    )
    gold = _Rows([[-100, LABEL_TO_ID["B-TIME"], -100, -100, -100]])

    metrics = score_aligned_predictions(predictions, gold)

    assert metrics["span_f1"] == 1.0
    assert metrics["sentence_exact_bio_accuracy"] == 1.0
