import pytest
from premove_itn_data.enrichment.compile import (
    EnrichmentCompileError,
    compile_enrichment_candidate,
)
from premove_itn_data.enrichment.donors import EnrichmentDonor
from premove_itn_data.enrichment.generate import (
    EnrichmentCandidate,
    EnrichmentSpan,
    generate_collision_candidates,
)
from premove_itn_data.records import TrainingRecord, TrainingSpan

from premove_itn.labels import SpanKind
from premove_itn.types import WordToken


def _span(
    kind: SpanKind,
    start: int,
    end: int,
    source: str,
    replacement: str,
) -> EnrichmentSpan:
    return EnrichmentSpan(
        kind=kind,
        start=start,
        end=end,
        source=source,
        replacement=replacement,
        context_provenance="purpose_built/test",
        donor_provenance="google_tn/test/1/0",
    )


def test_compiles_expected_text_tokens_and_bio_labels() -> None:
    candidate = EnrichmentCandidate(
        text="Call four thirty.",
        context_provenance="purpose_built/time/arrival/000001",
        spans=(_span(SpanKind.TIME, 5, 16, "four thirty", "4:30"),),
    )

    record = compile_enrichment_candidate(candidate)

    assert record == TrainingRecord(
        text="Call four thirty.",
        expected_text="Call 4:30.",
        spans=(TrainingSpan(SpanKind.TIME, 5, 16, "four thirty", "4:30"),),
        tokens=(
            WordToken("Call", 0, 4),
            WordToken("four", 5, 9),
            WordToken("thirty", 10, 16),
            WordToken(".", 16, 17),
        ),
        bio_labels=("O", "B-TIME", "I-TIME", "O"),
    )


def test_compiles_multiple_spans_and_restarts_bio_labels() -> None:
    candidate = EnrichmentCandidate(
        text="Book june sixth at four thirty.",
        context_provenance="sgd/train/dialogues_001.json/1_00000/3",
        spans=(
            _span(SpanKind.DATE, 5, 15, "june sixth", "June 6"),
            _span(SpanKind.TIME, 19, 30, "four thirty", "4:30"),
        ),
    )

    record = compile_enrichment_candidate(candidate)

    assert record.expected_text == "Book June 6 at 4:30."
    assert record.bio_labels == (
        "O",
        "B-DATE",
        "I-DATE",
        "O",
        "B-TIME",
        "I-TIME",
        "O",
    )


def test_compiles_context_only_candidate_with_o_labels() -> None:
    candidate = EnrichmentCandidate(
        text="Thanks for your help.",
        context_provenance="sgd/train/dialogues_001.json/1_00000/5",
        spans=(),
    )

    record = compile_enrichment_candidate(candidate)

    assert record.expected_text == candidate.text
    assert record.spans == ()
    assert record.bio_labels == ("O", "O", "O", "O", "O")


def test_same_output_semantic_collision_compiles_to_distinct_bio_kinds() -> None:
    spoken = "nine one one"
    phone = EnrichmentDonor(SpanKind.PHONE, spoken, "911", "phone")
    digits = EnrichmentDonor(SpanKind.DIGIT_SEQUENCE, spoken, "911", "digits")
    candidates = generate_collision_candidates(
        phone,
        digits,
        seed="dataset-v1",
        index=0,
    )

    records = tuple(compile_enrichment_candidate(candidate) for candidate in candidates)

    assert {record.expected_text for record in records} == {
        record.text.replace(spoken, "911") for record in records
    }
    assert {
        label.removeprefix("B-")
        for record in records
        for label in record.bio_labels
        if label.startswith("B-")
    } == {SpanKind.PHONE.value, SpanKind.DIGIT_SEQUENCE.value}


@pytest.mark.parametrize(
    ("candidate", "match"),
    [
        (
            EnrichmentCandidate("", "context", ()),
            "text must be non-empty",
        ),
        (
            EnrichmentCandidate("hello", "", ()),
            "context provenance must be non-empty",
        ),
        (
            EnrichmentCandidate(
                "call four thirty",
                "context",
                (_span(SpanKind.TIME, 5, 16, "wrong source", "4:30"),),
            ),
            "source slice",
        ),
        (
            EnrichmentCandidate(
                "four",
                "context",
                (_span(SpanKind.CARDINAL, 0, 2, "fo", "4"),),
            ),
            "token boundaries",
        ),
    ],
)
def test_compile_fails_closed(
    candidate: EnrichmentCandidate,
    match: str,
) -> None:
    with pytest.raises(EnrichmentCompileError, match=match):
        compile_enrichment_candidate(candidate)
