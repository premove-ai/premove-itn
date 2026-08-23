import pytest

from premove_itn.dataset.google_tn_candidates import (
    GoogleTnCandidateResult,
    GoogleTnSpanCandidate,
)
from premove_itn.dataset.google_tn_compile import (
    GoogleTnCompileError,
    assemble_google_tn_sentence,
)
from premove_itn.dataset.google_tn_parser import GoogleTnRow, GoogleTnSentence
from premove_itn.dataset.google_tn_validation import (
    GoogleTnCandidateRejection,
    GoogleTnRejectionReason,
    GoogleTnValidationResult,
)
from premove_itn.labels import SpanKind


def test_assembly_reconstructs_source_order_and_exact_span() -> None:
    rows = (
        GoogleTnRow("sample.tsv", 1, "PLAIN", "call", "<self>"),
        GoogleTnRow("sample.tsv", 2, "PLAIN", "me", "<self>"),
        GoogleTnRow("sample.tsv", 3, "PLAIN", "at", "<self>"),
        GoogleTnRow("sample.tsv", 4, "TIME", "4:30", "four thirty"),
    )
    sentence = GoogleTnSentence("sample.tsv", 1, rows)
    candidate = GoogleTnSpanCandidate(
        "sample.tsv", 1, 4, "TIME", SpanKind.TIME, "4:30", "four thirty"
    )
    extraction = GoogleTnCandidateResult(sentence, (candidate,), rows[:3], ())
    validation = GoogleTnValidationResult(extraction, (candidate,), ())

    assembled = assemble_google_tn_sentence(validation)

    assert assembled.text == "call me at four thirty"
    assert assembled.expected_text == "call me at 4:30"
    assert len(assembled.spans) == 1
    span = assembled.spans[0]
    assert span.kind is SpanKind.TIME
    assert (span.start, span.end) == (11, 22)
    assert span.source == "four thirty"
    assert span.replacement == "4:30"


def test_assembly_uses_sentence_order_for_multiple_spans() -> None:
    rows = (
        GoogleTnRow("sample.tsv", 5, "DATE", "2005", "two thousand five"),
        GoogleTnRow("sample.tsv", 6, "PLAIN", "at", "<self>"),
        GoogleTnRow("sample.tsv", 7, "TIME", "4:30", "four thirty"),
    )
    sentence = GoogleTnSentence("sample.tsv", 2, rows)
    date = GoogleTnSpanCandidate(
        "sample.tsv", 2, 5, "DATE", SpanKind.DATE, "2005", "two thousand five"
    )
    time = GoogleTnSpanCandidate(
        "sample.tsv", 2, 7, "TIME", SpanKind.TIME, "4:30", "four thirty"
    )
    extraction = GoogleTnCandidateResult(sentence, (date, time), (rows[1],), ())
    validation = GoogleTnValidationResult(extraction, (time, date), ())

    assembled = assemble_google_tn_sentence(validation)

    assert assembled.text == "two thousand five at four thirty"
    assert assembled.expected_text == "2005 at 4:30"
    assert [(span.kind, span.start, span.end) for span in assembled.spans] == [
        (SpanKind.DATE, 0, 17),
        (SpanKind.TIME, 21, 32),
    ]


def test_rejected_candidate_prevents_assembly() -> None:
    row = GoogleTnRow("sample.tsv", 1, "TIME", "4:30", "four thirty")
    sentence = GoogleTnSentence("sample.tsv", 1, (row,))
    candidate = GoogleTnSpanCandidate(
        "sample.tsv", 1, 1, "TIME", SpanKind.TIME, "4:30", "four thirty"
    )
    extraction = GoogleTnCandidateResult(sentence, (candidate,), (), ())
    rejection = GoogleTnCandidateRejection(
        candidate, GoogleTnRejectionReason.REALIZER_REJECTED, None
    )
    validation = GoogleTnValidationResult(extraction, (), (rejection,))

    with pytest.raises(GoogleTnCompileError, match="unaccepted"):
        assemble_google_tn_sentence(validation)


def test_quarantined_row_prevents_assembly() -> None:
    row = GoogleTnRow("sample.tsv", 1, "FRACTION", "1/2", "one half")
    sentence = GoogleTnSentence("sample.tsv", 1, (row,))
    extraction = GoogleTnCandidateResult(sentence, (), (), (row,))
    validation = GoogleTnValidationResult(extraction, (), ())

    with pytest.raises(GoogleTnCompileError, match="unaccepted"):
        assemble_google_tn_sentence(validation)


def test_span_row_requires_one_trusted_candidate() -> None:
    row = GoogleTnRow("sample.tsv", 1, "TIME", "4:30", "four thirty")
    sentence = GoogleTnSentence("sample.tsv", 1, (row,))
    extraction = GoogleTnCandidateResult(sentence, (), (), ())
    validation = GoogleTnValidationResult(extraction, (), ())

    with pytest.raises(GoogleTnCompileError, match="exactly one"):
        assemble_google_tn_sentence(validation)


def test_trusted_candidate_must_correspond_to_the_source_row() -> None:
    row = GoogleTnRow("sample.tsv", 4, "TIME", "4:30", "four thirty")
    sentence = GoogleTnSentence("sample.tsv", 1, (row,))
    wrong_source = GoogleTnSpanCandidate(
        "other.tsv", 1, 4, "TIME", SpanKind.TIME, "4:30", "four thirty"
    )
    extraction = GoogleTnCandidateResult(sentence, (wrong_source,), (), ())
    validation = GoogleTnValidationResult(extraction, (wrong_source,), ())

    with pytest.raises(GoogleTnCompileError, match="exactly one"):
        assemble_google_tn_sentence(validation)
