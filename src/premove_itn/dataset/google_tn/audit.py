import argparse
import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

from premove_itn.dataset.google_tn.parser import (
    GoogleTnSentence,
    read_google_sentences,
)
from premove_itn.dataset.google_tn.pipeline import (
    GoogleTnSentenceOutcome,
    iter_google_tn_outcomes,
)
from premove_itn.dataset.google_tn.validation import GoogleTnRealizer


@dataclass(frozen=True, slots=True)
class GoogleTnAuditSample:
    status: str
    source_name: str
    sentence_number: int
    text: str | None
    expected_text: str | None
    classes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GoogleTnAuditReport:
    sentences_processed: int
    accepted_sentences: int
    rejected_sentences: int
    accepted_span_count: int
    context_only_sentences: int
    multi_span_sentences: int
    accepted_by_class: dict[str, int]
    accepted_by_match_kind: dict[str, int]
    accepted_by_class_and_match_kind: dict[str, dict[str, int]]
    quarantined_by_class: dict[str, int]
    rejected_by_reason: dict[str, int]
    rejected_by_class: dict[str, int]
    samples: tuple[GoogleTnAuditSample, ...]


def audit_google_tn(
    sentences: Iterable[GoogleTnSentence],
    realizer: GoogleTnRealizer,
    *,
    limit: int,
    sample_limit: int = 3,
) -> GoogleTnAuditReport:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")

    outcomes = iter_google_tn_outcomes(islice(sentences, limit), realizer)
    return audit_google_tn_outcomes(outcomes, sample_limit=sample_limit)


def audit_google_tn_outcomes(
    outcomes: Iterable[GoogleTnSentenceOutcome],
    *,
    sample_limit: int = 3,
    observer: Callable[[GoogleTnSentenceOutcome], None] | None = None,
) -> GoogleTnAuditReport:
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")

    processed = 0
    accepted = 0
    accepted_span_count = 0
    context_only = 0
    multi_span = 0
    accepted_by_class: Counter[str] = Counter()
    accepted_by_match_kind: Counter[str] = Counter()
    accepted_by_class_and_match_kind: dict[str, Counter[str]] = {}
    quarantined_by_class: Counter[str] = Counter()
    rejected_by_reason: Counter[str] = Counter()
    rejected_by_class: Counter[str] = Counter()
    samples: list[GoogleTnAuditSample] = []
    sample_counts: Counter[str] = Counter()

    for outcome in outcomes:
        if observer is not None:
            observer(outcome)
        processed += 1
        status = "accepted" if outcome.is_accepted else "rejected"

        if outcome.record is not None:
            accepted += 1
            span_count = len(outcome.record.spans)
            accepted_span_count += span_count
            context_only += span_count == 0
            multi_span += span_count > 1
            accepted_by_class.update(span.kind.value for span in outcome.record.spans)
            accepted_by_match_kind.update(
                candidate.match_kind.value for candidate in outcome.trusted_candidates
            )
            for candidate in outcome.trusted_candidates:
                accepted_by_class_and_match_kind.setdefault(
                    candidate.kind.value, Counter()
                ).update((candidate.match_kind.value,))
            text = outcome.record.text
            expected_text = outcome.record.expected_text
            classes = tuple(span.kind.value for span in outcome.record.spans)
        else:
            quarantined_by_class.update(
                row.source_class for row in outcome.quarantined_rows
            )
            rejected_by_reason.update(
                rejection.reason.value for rejection in outcome.candidate_rejections
            )
            rejected_by_class.update(
                rejection.candidate.kind.value
                for rejection in outcome.candidate_rejections
            )
            text = None
            expected_text = None
            classes = tuple(
                row.source_class for row in outcome.quarantined_rows
            ) + tuple(
                rejection.candidate.kind.value
                for rejection in outcome.candidate_rejections
            )

        if sample_counts[status] < sample_limit:
            samples.append(
                GoogleTnAuditSample(
                    status,
                    outcome.sentence.source_name,
                    outcome.sentence.sentence_number,
                    text,
                    expected_text,
                    classes,
                )
            )
            sample_counts[status] += 1

    return GoogleTnAuditReport(
        processed,
        accepted,
        processed - accepted,
        accepted_span_count,
        context_only,
        multi_span,
        dict(sorted(accepted_by_class.items())),
        dict(sorted(accepted_by_match_kind.items())),
        {
            kind: dict(sorted(counts.items()))
            for kind, counts in sorted(accepted_by_class_and_match_kind.items())
        },
        dict(sorted(quarantined_by_class.items())),
        dict(sorted(rejected_by_reason.items())),
        dict(sorted(rejected_by_class.items())),
        tuple(samples),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a bounded stream of Google TN sentences."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--sample-limit", type=int, default=3)
    args = parser.parse_args()

    from premove_itn import _rust

    report = audit_google_tn(
        read_google_sentences(args.input),
        _rust.realize,
        limit=args.limit,
        sample_limit=args.sample_limit,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
