import pytest

from premove_itn.dataset.google_tn.candidates import extract_google_tn_candidates
from premove_itn.dataset.google_tn.parser import GoogleTnRow, GoogleTnSentence
from premove_itn.dataset.google_tn.policy import GOOGLE_QUARANTINE_CLASSES
from premove_itn.labels import SpanKind


def test_plain_row_becomes_context() -> None:
    row = GoogleTnRow("sample.tsv", 7, "PLAIN", "call", "<self>")
    sentence = GoogleTnSentence("sample.tsv", 3, (row,))

    result = extract_google_tn_candidates(sentence)

    assert result.sentence is sentence
    assert result.span_candidates == ()
    assert result.context_rows == (row,)
    assert result.quarantined_rows == ()


@pytest.mark.parametrize(
    ("source_class", "kind"),
    [
        ("CARDINAL", SpanKind.CARDINAL),
        ("DATE", SpanKind.DATE),
        ("DECIMAL", SpanKind.DECIMAL),
        ("DIGIT", SpanKind.DIGIT_SEQUENCE),
        ("ELECTRONIC", SpanKind.ELECTRONIC),
        ("MEASURE", SpanKind.MEASUREMENT),
        ("MONEY", SpanKind.MONEY),
        ("ORDINAL", SpanKind.ORDINAL),
        ("TELEPHONE", SpanKind.PHONE),
        ("TIME", SpanKind.TIME),
    ],
)
def test_supported_class_becomes_candidate(source_class: str, kind: SpanKind) -> None:
    row = GoogleTnRow(
        "source.tsv", 19, source_class, " Written\u00a0value ", " Spoken  value "
    )
    sentence = GoogleTnSentence("source.tsv", 5, (row,))

    result = extract_google_tn_candidates(sentence)

    assert len(result.span_candidates) == 1
    candidate = result.span_candidates[0]
    assert candidate.source_name == "source.tsv"
    assert candidate.sentence_number == 5
    assert candidate.line_number == 19
    assert candidate.source_class == source_class
    assert candidate.kind is kind
    assert candidate.written == " Written\u00a0value "
    assert candidate.spoken == " Spoken  value "
    assert result.context_rows == ()
    assert result.quarantined_rows == ()


@pytest.mark.parametrize("source_class", sorted(GOOGLE_QUARANTINE_CLASSES))
def test_quarantined_class_becomes_quarantined_row(source_class: str) -> None:
    row = GoogleTnRow("sample.tsv", 2, source_class, "written", "spoken")
    sentence = GoogleTnSentence("sample.tsv", 1, (row,))

    result = extract_google_tn_candidates(sentence)

    assert result.span_candidates == ()
    assert result.context_rows == ()
    assert result.quarantined_rows == (row,)


def test_mixed_sentence_reports_each_category_in_source_order() -> None:
    rows = (
        GoogleTnRow("sample.tsv", 10, "PLAIN", "call", "<self>"),
        GoogleTnRow("sample.tsv", 11, "TIME", "4:30", "four thirty"),
        GoogleTnRow("sample.tsv", 12, "FRACTION", "1/2", "one half"),
        GoogleTnRow("sample.tsv", 13, "PLAIN", "then", "<self>"),
        GoogleTnRow("sample.tsv", 14, "DATE", "2005", "two thousand five"),
        GoogleTnRow("sample.tsv", 15, "PUNCT", ".", "sil"),
    )
    sentence = GoogleTnSentence("sample.tsv", 4, rows)

    result = extract_google_tn_candidates(sentence)

    assert [candidate.line_number for candidate in result.span_candidates] == [11, 14]
    assert result.context_rows == (rows[0], rows[3], rows[5])
    assert result.quarantined_rows == (rows[2],)


def test_unknown_class_propagates_policy_error() -> None:
    row = GoogleTnRow("sample.tsv", 2, "NEW_CLASS", "written", "spoken")
    sentence = GoogleTnSentence("sample.tsv", 1, (row,))

    with pytest.raises(ValueError, match="unknown Google TN class"):
        extract_google_tn_candidates(sentence)
