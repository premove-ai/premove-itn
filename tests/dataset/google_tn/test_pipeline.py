import pytest

from premove_itn.dataset.google_tn.parser import GoogleTnRow, GoogleTnSentence
from premove_itn.dataset.google_tn.pipeline import (
    GoogleTnSentenceOutcome,
    iter_google_tn_outcomes,
    process_google_tn_sentence,
)
from premove_itn.dataset.google_tn.validation import GoogleTnRejectionReason


def test_accepted_sentence_produces_training_record() -> None:
    sentence = GoogleTnSentence(
        "sample.tsv",
        3,
        (
            GoogleTnRow("sample.tsv", 7, "PLAIN", "call", "<self>"),
            GoogleTnRow("sample.tsv", 8, "TIME", "4:30", "four thirty"),
        ),
    )

    outcome = process_google_tn_sentence(sentence, lambda kind, spoken: "4:30")

    assert outcome.sentence is sentence
    assert outcome.record is not None
    assert outcome.record.text == "call four thirty"
    assert outcome.record.expected_text == "call 4:30"
    assert outcome.record.bio_labels == ("O", "B-TIME", "I-TIME")
    assert outcome.quarantined_rows == ()
    assert outcome.candidate_rejections == ()
    assert outcome.is_accepted


def test_quarantined_sentence_produces_no_record() -> None:
    fraction = GoogleTnRow("sample.tsv", 9, "FRACTION", "1/2", "one half")
    sentence = GoogleTnSentence(
        "sample.tsv",
        3,
        (
            GoogleTnRow("sample.tsv", 8, "TIME", "4:30", "four thirty"),
            fraction,
        ),
    )

    outcome = process_google_tn_sentence(sentence, lambda kind, spoken: "4:30")

    assert outcome.record is None
    assert outcome.quarantined_rows == (fraction,)
    assert outcome.candidate_rejections == ()
    assert not outcome.is_accepted


@pytest.mark.parametrize(
    ("realized", "reason"),
    [
        (None, GoogleTnRejectionReason.REALIZER_REJECTED),
        ("04:30", GoogleTnRejectionReason.WRITTEN_MISMATCH),
    ],
)
def test_candidate_rejection_produces_no_record(
    realized: str | None, reason: GoogleTnRejectionReason
) -> None:
    sentence = GoogleTnSentence(
        "sample.tsv",
        3,
        (GoogleTnRow("sample.tsv", 8, "TIME", "4:30", "four thirty"),),
    )

    outcome = process_google_tn_sentence(sentence, lambda kind, spoken: realized)

    assert outcome.record is None
    assert outcome.quarantined_rows == ()
    assert len(outcome.candidate_rejections) == 1
    assert outcome.candidate_rejections[0].reason is reason
    assert outcome.candidate_rejections[0].realized == realized
    assert not outcome.is_accepted


def test_outcome_rejects_record_with_quarantine() -> None:
    accepted_sentence = GoogleTnSentence(
        "sample.tsv",
        1,
        (GoogleTnRow("sample.tsv", 1, "PLAIN", "call", "<self>"),),
    )
    accepted = process_google_tn_sentence(accepted_sentence, lambda kind, spoken: None)
    fraction = GoogleTnRow("sample.tsv", 2, "FRACTION", "1/2", "one half")

    with pytest.raises(ValueError, match="record exists if and only if"):
        GoogleTnSentenceOutcome(accepted_sentence, accepted.record, (fraction,), ())


def test_outcome_rejects_empty_rejection_without_record() -> None:
    sentence = GoogleTnSentence(
        "sample.tsv",
        1,
        (GoogleTnRow("sample.tsv", 1, "PLAIN", "call", "<self>"),),
    )

    with pytest.raises(ValueError, match="record exists if and only if"):
        GoogleTnSentenceOutcome(sentence, None, (), ())


def test_unknown_class_propagates() -> None:
    sentence = GoogleTnSentence(
        "sample.tsv",
        1,
        (GoogleTnRow("sample.tsv", 1, "NEW_CLASS", "x", "x"),),
    )

    with pytest.raises(ValueError, match="unknown Google TN class"):
        process_google_tn_sentence(sentence, lambda kind, spoken: None)


def test_outcome_iterator_consumes_one_sentence_at_a_time() -> None:
    events: list[str] = []
    first = GoogleTnSentence(
        "sample.tsv",
        1,
        (GoogleTnRow("sample.tsv", 1, "PLAIN", "call", "<self>"),),
    )
    second = GoogleTnSentence(
        "sample.tsv",
        2,
        (GoogleTnRow("sample.tsv", 3, "PLAIN", "later", "<self>"),),
    )

    def sentences():
        events.append("first")
        yield first
        events.append("second")
        yield second

    outcomes = iter_google_tn_outcomes(sentences(), lambda kind, spoken: None)

    assert events == []
    assert next(outcomes).sentence is first
    assert events == ["first"]
    assert next(outcomes).sentence is second
    assert events == ["first", "second"]
