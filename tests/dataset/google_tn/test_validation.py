import pytest

from premove_itn.dataset.google_tn.candidates import (
    GoogleTnCandidateResult,
    GoogleTnSpanCandidate,
)
from premove_itn.dataset.google_tn.parser import GoogleTnRow, GoogleTnSentence
from premove_itn.dataset.google_tn.validation import (
    GoogleTnMatchKind,
    GoogleTnRejectionReason,
    validate_google_tn_candidates,
)
from premove_itn.labels import SpanKind


def test_exact_realizer_match_trusts_candidate() -> None:
    row = GoogleTnRow("sample.tsv", 8, "TIME", "4:30", "four thirty")
    sentence = GoogleTnSentence("sample.tsv", 2, (row,))
    candidate = GoogleTnSpanCandidate(
        "sample.tsv", 2, 8, "TIME", SpanKind.TIME, "4:30", "four thirty"
    )
    extracted = GoogleTnCandidateResult(sentence, (candidate,), (), ())

    result = validate_google_tn_candidates(extracted, lambda kind, spoken: "4:30")

    assert result.extraction is extracted
    assert len(result.trusted_candidates) == 1
    trusted = result.trusted_candidates[0]
    assert trusted.source_name == "sample.tsv"
    assert trusted.sentence_number == 2
    assert trusted.line_number == 8
    assert trusted.source_class == "TIME"
    assert trusted.kind is SpanKind.TIME
    assert trusted.spoken == "four thirty"
    assert trusted.google_written == "4:30"
    assert trusted.realized == "4:30"
    assert trusted.match_kind is GoogleTnMatchKind.EXACT
    assert result.rejected_candidates == ()
    assert result.is_accepted


def test_realizer_rejection_rejects_candidate() -> None:
    row = GoogleTnRow("sample.tsv", 8, "TIME", "4:30", "four thirty")
    sentence = GoogleTnSentence("sample.tsv", 2, (row,))
    candidate = GoogleTnSpanCandidate(
        "sample.tsv", 2, 8, "TIME", SpanKind.TIME, "4:30", "four thirty"
    )
    extracted = GoogleTnCandidateResult(sentence, (candidate,), (), ())

    result = validate_google_tn_candidates(extracted, lambda kind, spoken: None)

    assert result.trusted_candidates == ()
    assert len(result.rejected_candidates) == 1
    rejection = result.rejected_candidates[0]
    assert rejection.candidate is candidate
    assert rejection.reason is GoogleTnRejectionReason.REALIZER_REJECTED
    assert rejection.realized is None
    assert not result.is_accepted


def test_realizer_mismatch_rejects_candidate_without_canonicalization() -> None:
    row = GoogleTnRow("sample.tsv", 8, "MONEY", "$25", "twenty five dollars")
    sentence = GoogleTnSentence("sample.tsv", 2, (row,))
    candidate = GoogleTnSpanCandidate(
        "sample.tsv",
        2,
        8,
        "MONEY",
        SpanKind.MONEY,
        "$25",
        "twenty five dollars",
    )
    extracted = GoogleTnCandidateResult(sentence, (candidate,), (), ())

    def realize(kind: str, spoken: str) -> str:
        assert kind == "MONEY"
        assert spoken == "twenty five dollars"
        return "USD 25"

    result = validate_google_tn_candidates(extracted, realize)

    assert result.trusted_candidates == ()
    rejection = result.rejected_candidates[0]
    assert rejection.candidate is candidate
    assert rejection.reason is GoogleTnRejectionReason.WRITTEN_MISMATCH
    assert rejection.realized == "USD 25"
    assert not result.is_accepted


def test_quarantine_rejects_sentence_even_when_candidate_is_trusted() -> None:
    time_row = GoogleTnRow("sample.tsv", 8, "TIME", "4:30", "four thirty")
    fraction_row = GoogleTnRow("sample.tsv", 9, "FRACTION", "1/2", "one half")
    sentence = GoogleTnSentence("sample.tsv", 2, (time_row, fraction_row))
    candidate = GoogleTnSpanCandidate(
        "sample.tsv", 2, 8, "TIME", SpanKind.TIME, "4:30", "four thirty"
    )
    extracted = GoogleTnCandidateResult(sentence, (candidate,), (), (fraction_row,))

    result = validate_google_tn_candidates(extracted, lambda kind, spoken: "4:30")

    assert len(result.trusted_candidates) == 1
    assert result.trusted_candidates[0].realized == "4:30"
    assert result.rejected_candidates == ()
    assert result.extraction.quarantined_rows == (fraction_row,)
    assert not result.is_accepted


def test_context_only_sentence_is_accepted_without_realization() -> None:
    row = GoogleTnRow("sample.tsv", 3, "PLAIN", "call", "<self>")
    sentence = GoogleTnSentence("sample.tsv", 1, (row,))
    extracted = GoogleTnCandidateResult(sentence, (), (row,), ())

    def unexpected_realizer(kind: str, spoken: str) -> None:
        raise AssertionError("context rows must not be realized")

    result = validate_google_tn_candidates(extracted, unexpected_realizer)

    assert result.trusted_candidates == ()
    assert result.rejected_candidates == ()
    assert result.is_accepted


def test_one_rejected_candidate_rejects_the_whole_sentence() -> None:
    time = GoogleTnSpanCandidate(
        "sample.tsv", 4, 11, "TIME", SpanKind.TIME, "4:30", "four thirty"
    )
    date = GoogleTnSpanCandidate(
        "sample.tsv", 4, 14, "DATE", SpanKind.DATE, "2005", "two thousand five"
    )
    rows = (
        GoogleTnRow("sample.tsv", 11, "TIME", "4:30", "four thirty"),
        GoogleTnRow("sample.tsv", 14, "DATE", "2005", "two thousand five"),
    )
    extraction = GoogleTnCandidateResult(
        GoogleTnSentence("sample.tsv", 4, rows), (time, date), (), ()
    )

    results = {"TIME": "4:30", "DATE": "05"}
    result = validate_google_tn_candidates(
        extraction, lambda kind, spoken: results[kind]
    )

    assert [item.spoken for item in result.trusted_candidates] == ["four thirty"]
    assert [item.candidate for item in result.rejected_candidates] == [date]
    assert not result.is_accepted


@pytest.mark.parametrize(
    "google_written",
    ["April 10 2013", "April 10, 2013", "April   10,  2013"],
)
def test_date_formatting_equivalence_trusts_rust_result(
    google_written: str,
) -> None:
    row = GoogleTnRow(
        "sample.tsv", 8, "DATE", google_written, "april tenth twenty thirteen"
    )
    sentence = GoogleTnSentence("sample.tsv", 2, (row,))
    candidate = GoogleTnSpanCandidate(
        "sample.tsv",
        2,
        8,
        "DATE",
        SpanKind.DATE,
        google_written,
        "april tenth twenty thirteen",
    )
    extraction = GoogleTnCandidateResult(sentence, (candidate,), (), ())

    result = validate_google_tn_candidates(
        extraction, lambda kind, spoken: "april 10 2013"
    )

    assert result.rejected_candidates == ()
    trusted = result.trusted_candidates[0]
    assert trusted.google_written == google_written
    assert trusted.realized == "april 10 2013"
    assert trusted.match_kind is GoogleTnMatchKind.CANONICAL


@pytest.mark.parametrize(
    ("google_written", "realized"),
    [
        ("April 10, 2013", "april 11 2013"),
        ("10 April 2013", "april 10 2013"),
    ],
)
def test_date_semantic_or_order_mismatch_is_rejected(
    google_written: str, realized: str
) -> None:
    row = GoogleTnRow(
        "sample.tsv", 8, "DATE", google_written, "april tenth twenty thirteen"
    )
    sentence = GoogleTnSentence("sample.tsv", 2, (row,))
    candidate = GoogleTnSpanCandidate(
        "sample.tsv",
        2,
        8,
        "DATE",
        SpanKind.DATE,
        google_written,
        "april tenth twenty thirteen",
    )
    extraction = GoogleTnCandidateResult(sentence, (candidate,), (), ())

    result = validate_google_tn_candidates(extraction, lambda kind, spoken: realized)

    assert result.trusted_candidates == ()
    assert result.rejected_candidates[0].reason is (
        GoogleTnRejectionReason.WRITTEN_MISMATCH
    )


def _validate_written_equivalence(kind: SpanKind, google_written: str, realized: str):
    row = GoogleTnRow("sample.tsv", 1, kind.value, google_written, "spoken")
    sentence = GoogleTnSentence("sample.tsv", 1, (row,))
    candidate = GoogleTnSpanCandidate(
        "sample.tsv", 1, 1, kind.value, kind, google_written, "spoken"
    )
    extraction = GoogleTnCandidateResult(sentence, (candidate,), (), ())
    return validate_google_tn_candidates(extraction, lambda kind, spoken: realized)


@pytest.mark.parametrize(
    ("kind", "google_written", "realized"),
    [
        (SpanKind.CARDINAL, "11,331", "11331"),
        (SpanKind.DECIMAL, "5,661.38", "5661.38"),
        (SpanKind.MONEY, "$7,000", "$7000"),
        (SpanKind.MONEY, "USD 7,000.50", "USD 7000.50"),
    ],
)
def test_numeric_grouping_equivalence_is_canonical(
    kind: SpanKind, google_written: str, realized: str
) -> None:
    result = _validate_written_equivalence(kind, google_written, realized)

    assert result.rejected_candidates == ()
    assert result.trusted_candidates[0].realized == realized
    assert result.trusted_candidates[0].match_kind is GoogleTnMatchKind.CANONICAL


@pytest.mark.parametrize(
    ("kind", "google_written", "realized"),
    [
        (SpanKind.CARDINAL, "11,33", "1133"),
        (SpanKind.DECIMAL, "56,61.38", "5661.38"),
        (SpanKind.MONEY, "$62,500", "$6200500"),
        (SpanKind.MONEY, "$7,000", "USD 7000"),
    ],
)
def test_numeric_grouping_does_not_hide_malformed_or_semantic_difference(
    kind: SpanKind, google_written: str, realized: str
) -> None:
    result = _validate_written_equivalence(kind, google_written, realized)

    assert result.trusted_candidates == ()
    assert result.rejected_candidates[0].reason is (
        GoogleTnRejectionReason.WRITTEN_MISMATCH
    )


@pytest.mark.parametrize(
    ("google_written", "realized"),
    [
        ("4:12", "04:12"),
        ("4:00", "04:00"),
        ("9:59", "09:59"),
        ("0:12", "00:12"),
        ("9pm", "09:00 p.m."),
        ("9 P M", "21:00"),
        ("18:43", "6:43 PM"),
        ("00:12", "12:12 a.m."),
        ("12:00", "12 PM"),
    ],
)
def test_supported_time_representations_compare_by_clock_value(
    google_written: str, realized: str
) -> None:
    result = _validate_written_equivalence(SpanKind.TIME, google_written, realized)

    assert result.rejected_candidates == ()
    assert result.trusted_candidates[0].realized == realized
    assert result.trusted_candidates[0].match_kind is GoogleTnMatchKind.CANONICAL


@pytest.mark.parametrize(
    ("google_written", "realized"),
    [
        ("00:12", "12:12"),
        ("04:12", "04:13"),
        ("04:12", "4:21"),
        ("4:02", "4:2"),
        ("9pm", "09:00 a.m."),
        ("10.50pm IST", "10:50 PM IST"),
        ("9 PAM", "21:00"),
    ],
)
def test_time_comparison_rejects_different_or_unapproved_forms(
    google_written: str, realized: str
) -> None:
    result = _validate_written_equivalence(SpanKind.TIME, google_written, realized)

    assert result.trusted_candidates == ()
    assert result.rejected_candidates[0].reason is (
        GoogleTnRejectionReason.WRITTEN_MISMATCH
    )
