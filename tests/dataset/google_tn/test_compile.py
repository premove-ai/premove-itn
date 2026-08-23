import pytest

from premove_itn.dataset.google_tn.candidates import (
    GoogleTnCandidateResult,
    GoogleTnSpanCandidate,
)
from premove_itn.dataset.google_tn.compile import (
    GoogleTnAssembledSentence,
    GoogleTnAssembledSpan,
    GoogleTnCompileError,
    assemble_google_tn_sentence,
    compile_google_tn_record,
)
from premove_itn.dataset.google_tn.parser import GoogleTnRow, GoogleTnSentence
from premove_itn.dataset.google_tn.validation import (
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


def test_compile_record_tokenizes_text_and_derives_bio_labels() -> None:
    span = GoogleTnAssembledSpan(SpanKind.TIME, 11, 22, "four thirty", "4:30")
    assembled = GoogleTnAssembledSentence(
        "call me at four thirty", "call me at 4:30", (span,)
    )

    record = compile_google_tn_record(assembled)

    assert record.text == assembled.text
    assert record.expected_text == assembled.expected_text
    assert record.spans == (span,)
    assert [(token.text, token.start, token.end) for token in record.tokens] == [
        ("call", 0, 4),
        ("me", 5, 7),
        ("at", 8, 10),
        ("four", 11, 15),
        ("thirty", 16, 22),
    ]
    assert record.bio_labels == ("O", "O", "O", "B-TIME", "I-TIME")


def test_compile_record_rejects_source_slice_mismatch() -> None:
    assembled = GoogleTnAssembledSentence(
        "call four thirty",
        "call 4:30",
        (GoogleTnAssembledSpan(SpanKind.TIME, 5, 16, "wrong value", "4:30"),),
    )

    with pytest.raises(GoogleTnCompileError, match="source slice"):
        compile_google_tn_record(assembled)


def test_compile_record_rejects_incorrect_expected_text() -> None:
    assembled = GoogleTnAssembledSentence(
        "call four thirty",
        "call 04:30",
        (GoogleTnAssembledSpan(SpanKind.TIME, 5, 16, "four thirty", "4:30"),),
    )

    with pytest.raises(GoogleTnCompileError, match="expected text"):
        compile_google_tn_record(assembled)


def test_compile_record_rejects_overlapping_spans() -> None:
    assembled = GoogleTnAssembledSentence(
        "one two three",
        "1 2 three",
        (
            GoogleTnAssembledSpan(SpanKind.CARDINAL, 0, 7, "one two", "1"),
            GoogleTnAssembledSpan(SpanKind.CARDINAL, 4, 7, "two", "2"),
        ),
    )

    with pytest.raises(GoogleTnCompileError, match="ordered and non-overlapping"):
        compile_google_tn_record(assembled)


def test_compile_record_rejects_partial_token_span() -> None:
    assembled = GoogleTnAssembledSentence(
        "four",
        "4ur",
        (GoogleTnAssembledSpan(SpanKind.CARDINAL, 0, 2, "fo", "4"),),
    )

    with pytest.raises(GoogleTnCompileError, match="token boundaries"):
        compile_google_tn_record(assembled)


def test_compile_record_rejects_span_without_tokens() -> None:
    assembled = GoogleTnAssembledSentence(
        "call  me",
        "call_me",
        (GoogleTnAssembledSpan(SpanKind.WORD, 4, 6, "  ", "_"),),
    )

    with pytest.raises(GoogleTnCompileError, match="at least one token"):
        compile_google_tn_record(assembled)


def test_compile_context_only_record_has_all_o_labels() -> None:
    assembled = GoogleTnAssembledSentence("call me", "call me", ())

    record = compile_google_tn_record(assembled)

    assert [token.text for token in record.tokens] == ["call", "me"]
    assert record.bio_labels == ("O", "O")


def test_compile_record_restarts_bio_for_each_span() -> None:
    assembled = GoogleTnAssembledSentence(
        "two thousand five at four thirty",
        "2005 at 4:30",
        (
            GoogleTnAssembledSpan(SpanKind.DATE, 0, 17, "two thousand five", "2005"),
            GoogleTnAssembledSpan(SpanKind.TIME, 21, 32, "four thirty", "4:30"),
        ),
    )

    record = compile_google_tn_record(assembled)

    assert record.bio_labels == (
        "B-DATE",
        "I-DATE",
        "I-DATE",
        "O",
        "B-TIME",
        "I-TIME",
    )
