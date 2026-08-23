from premove_itn.dataset.google_tn.candidates import (
    GoogleTnCandidateResult,
    GoogleTnSpanCandidate,
)
from premove_itn.dataset.google_tn.parser import GoogleTnRow, GoogleTnSentence
from premove_itn.dataset.google_tn.validation import (
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
    assert result.trusted_candidates == (candidate,)
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

    assert result.trusted_candidates == (candidate,)
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

    assert result.trusted_candidates == (time,)
    assert [item.candidate for item in result.rejected_candidates] == [date]
    assert not result.is_accepted
