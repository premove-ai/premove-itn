from premove_itn.dataset.google_tn.audit import audit_google_tn
from premove_itn.dataset.google_tn.parser import GoogleTnRow, GoogleTnSentence


def test_audit_counts_outcomes_and_keeps_bounded_samples() -> None:
    sentences = (
        GoogleTnSentence(
            "sample.tsv",
            1,
            (GoogleTnRow("sample.tsv", 1, "PLAIN", "call", "<self>"),),
        ),
        GoogleTnSentence(
            "sample.tsv",
            2,
            (GoogleTnRow("sample.tsv", 3, "TIME", "4:30", "four thirty"),),
        ),
        GoogleTnSentence(
            "sample.tsv",
            3,
            (GoogleTnRow("sample.tsv", 5, "FRACTION", "1/2", "one half"),),
        ),
        GoogleTnSentence(
            "sample.tsv",
            4,
            (GoogleTnRow("sample.tsv", 7, "TIME", "5:30", "five thirty"),),
        ),
    )
    realized = {"four thirty": "4:30", "five thirty": None}

    report = audit_google_tn(
        sentences,
        lambda kind, spoken: realized[spoken],
        limit=4,
        sample_limit=1,
    )

    assert report.sentences_processed == 4
    assert report.accepted_sentences == 2
    assert report.rejected_sentences == 2
    assert report.accepted_span_count == 1
    assert report.context_only_sentences == 1
    assert report.multi_span_sentences == 0
    assert report.accepted_by_class == {"TIME": 1}
    assert report.accepted_by_match_kind == {"exact": 1}
    assert report.accepted_by_class_and_match_kind == {"TIME": {"exact": 1}}
    assert report.quarantined_by_class == {"FRACTION": 1}
    assert report.rejected_by_reason == {"realizer_rejected": 1}
    assert report.rejected_by_class == {"TIME": 1}
    assert len(report.samples) == 2
