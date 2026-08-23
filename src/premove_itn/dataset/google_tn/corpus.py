import argparse
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import chain, islice
from pathlib import Path

from premove_itn.dataset.google_tn.audit import (
    GoogleTnAuditReport,
    GoogleTnAuditSample,
)
from premove_itn.dataset.google_tn.parser import GoogleTnSentence, read_google_sentences
from premove_itn.dataset.google_tn.pipeline import GoogleTnSentenceOutcome
from premove_itn.dataset.google_tn.shard import (
    GoogleTnShardManifest,
    _validate_shard_request,
    _write_positioned_google_tn_shard,
)
from premove_itn.dataset.google_tn.validation import GoogleTnRealizer


@dataclass(frozen=True, slots=True)
class GoogleTnRejectionSample:
    source_name: str
    sentence_number: int
    line_number: int
    source_class: str
    kind: str | None
    reason: str
    written: str
    spoken: str
    realized: str | None


@dataclass(frozen=True, slots=True)
class GoogleTnCorpusManifest:
    kind: str
    compiler: str
    schema_version: int
    source_id: str
    shard_size: int
    processed: int
    accepted: int
    rejected: int
    output_records: int
    shard_count: int
    shards: tuple[GoogleTnShardManifest, ...]


class _RejectionSampler:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.samples: dict[str, dict[str, list[GoogleTnRejectionSample]]] = {}

    def _add(self, group_class: str, sample: GoogleTnRejectionSample) -> None:
        if self.limit == 0:
            return
        values = self.samples.setdefault(group_class, {}).setdefault(sample.reason, [])
        if len(values) < self.limit:
            values.append(sample)

    def observe(self, outcome: GoogleTnSentenceOutcome) -> None:
        for row in outcome.quarantined_rows:
            self._add(
                row.source_class,
                GoogleTnRejectionSample(
                    row.source_name,
                    outcome.sentence.sentence_number,
                    row.line_number,
                    row.source_class,
                    None,
                    "quarantined",
                    row.written,
                    row.spoken,
                    None,
                ),
            )
        for rejection in outcome.candidate_rejections:
            candidate = rejection.candidate
            self._add(
                candidate.kind.value,
                GoogleTnRejectionSample(
                    candidate.source_name,
                    candidate.sentence_number,
                    candidate.line_number,
                    candidate.source_class,
                    candidate.kind.value,
                    rejection.reason.value,
                    candidate.written,
                    candidate.spoken,
                    rejection.realized,
                ),
            )

    def as_json(self) -> dict[str, dict[str, list[dict[str, object]]]]:
        return {
            source_class: {
                reason: [asdict(sample) for sample in samples]
                for reason, samples in sorted(reasons.items())
            }
            for source_class, reasons in sorted(self.samples.items())
        }


def _merge_counts(reports: list[GoogleTnAuditReport], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for report in reports:
        counts.update(getattr(report, field))
    return dict(sorted(counts.items()))


def _merge_audit_reports(reports: list[GoogleTnAuditReport]) -> GoogleTnAuditReport:
    nested: dict[str, Counter[str]] = {}
    for report in reports:
        for kind, values in report.accepted_by_class_and_match_kind.items():
            nested.setdefault(kind, Counter()).update(values)

    samples: list[GoogleTnAuditSample] = []
    sample_counts: Counter[str] = Counter()
    for report in reports:
        for sample in report.samples:
            if sample_counts[sample.status] < 3:
                samples.append(sample)
                sample_counts[sample.status] += 1

    processed = sum(report.sentences_processed for report in reports)
    accepted = sum(report.accepted_sentences for report in reports)
    return GoogleTnAuditReport(
        processed,
        accepted,
        processed - accepted,
        sum(report.accepted_span_count for report in reports),
        sum(report.context_only_sentences for report in reports),
        sum(report.multi_span_sentences for report in reports),
        _merge_counts(reports, "accepted_by_class"),
        _merge_counts(reports, "accepted_by_match_kind"),
        {kind: dict(sorted(values.items())) for kind, values in sorted(nested.items())},
        _merge_counts(reports, "quarantined_by_class"),
        _merge_counts(reports, "rejected_by_reason"),
        _merge_counts(reports, "rejected_by_class"),
        tuple(samples),
    )


def _publish_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_google_tn_accepted_corpus(
    sentences: Iterable[GoogleTnSentence],
    realizer: GoogleTnRealizer,
    output_directory: Path,
    *,
    source_id: str,
    shard_size: int,
    rejection_sample_limit: int = 100,
) -> GoogleTnCorpusManifest:
    _validate_shard_request(source_id, 0, shard_size)
    if rejection_sample_limit < 0:
        raise ValueError("rejection_sample_limit must be non-negative")

    audit_directory = output_directory / "audits"
    audit_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = audit_directory / f"{source_id}_corpus_manifest.json"
    summary_path = audit_directory / f"{source_id}_corpus_summary.json"
    samples_path = audit_directory / f"{source_id}_rejection_samples.json"
    for path in (manifest_path, summary_path, samples_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

    iterator = iter(sentences)
    source_start = 0
    shard_manifests: list[GoogleTnShardManifest] = []
    shard_reports: list[GoogleTnAuditReport] = []
    sampler = _RejectionSampler(rejection_sample_limit)

    while True:
        first = next(iterator, None)
        if first is None:
            break
        positioned = chain((first,), islice(iterator, shard_size - 1))
        shard_manifest, shard_report = _write_positioned_google_tn_shard(
            positioned,
            realizer,
            output_directory,
            source_id=source_id,
            source_start=source_start,
            requested_limit=shard_size,
            observer=sampler.observe,
        )
        shard_manifests.append(shard_manifest)
        shard_reports.append(shard_report)
        source_start = shard_manifest.source_end_exclusive

    report = _merge_audit_reports(shard_reports)
    manifest = GoogleTnCorpusManifest(
        "google_tn_accepted_corpus",
        "google_tn_v1",
        1,
        source_id,
        shard_size,
        report.sentences_processed,
        report.accepted_sentences,
        report.rejected_sentences,
        report.accepted_sentences,
        len(shard_manifests),
        tuple(shard_manifests),
    )
    _publish_json(summary_path, asdict(report))
    _publish_json(samples_path, sampler.as_json())
    _publish_json(manifest_path, asdict(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the full Google TN accepted corpus in one pass."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--source-id")
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--rejection-sample-limit", type=int, default=100)
    args = parser.parse_args()

    from premove_itn import _rust

    manifest = write_google_tn_accepted_corpus(
        read_google_sentences(args.input),
        _rust.realize,
        args.output_directory,
        source_id=args.source_id or args.input.stem,
        shard_size=args.shard_size,
        rejection_sample_limit=args.rejection_sample_limit,
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
