import json
from pathlib import Path

import pytest
from premove_itn_data.google_tn.select import (
    GOOGLE_TN_V1_CONTRIBUTION_QUOTAS,
    write_google_tn_candidates,
)
from premove_itn_data.selection import SelectionQuotas

from premove_itn.labels import SpanKind


def _payload(
    text: str,
    *,
    kind: str | None = None,
    source_file: str = "source.tsv",
    sentence_number: int = 1,
) -> dict[str, object]:
    span = []
    expected_text = text
    label = "O"
    if kind is not None:
        span = [
            {
                "kind": kind,
                "start": 0,
                "end": len(text),
                "source": text,
                "replacement": "target",
            }
        ]
        expected_text = "target"
        label = f"B-{kind}"
    return {
        "record": {
            "text": text,
            "expected_text": expected_text,
            "spans": span,
            "tokens": [{"text": text, "start": 0, "end": len(text)}],
            "bio_labels": [label],
        },
        "provenance": {
            "source": "google_tn",
            "source_file": source_file,
            "sentence_number": sentence_number,
        },
    }


def _write_corpus(
    root: Path,
    shards: list[list[dict[str, object]]],
) -> Path:
    accepted = root / "accepted"
    audits = root / "audits"
    accepted.mkdir(parents=True)
    audits.mkdir(parents=True)
    shard_manifests = []
    source_start = 0
    for records in shards:
        source_end = source_start + len(records)
        stem = f"source_{source_start:06d}_{source_end:06d}"
        (accepted / f"{stem}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        shard_manifests.append(
            {
                "kind": "google_tn_accepted_corpus_shard",
                "compiler": "google_tn_v1",
                "schema_version": 1,
                "source_id": "source",
                "source_start": source_start,
                "source_end_exclusive": source_end,
                "processed": len(records),
                "accepted": len(records),
                "rejected": 0,
                "output_records": len(records),
            }
        )
        source_start = source_end

    manifest = {
        "kind": "google_tn_accepted_corpus",
        "compiler": "google_tn_v1",
        "schema_version": 1,
        "processed": source_start,
        "accepted": source_start,
        "rejected": 0,
        "output_records": source_start,
        "shard_count": len(shards),
        "shard_size": max(map(len, shards), default=0),
        "shards": shard_manifests,
    }
    manifest_path = audits / "source_corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_writer_selects_records_and_preserves_record_and_provenance(tmp_path) -> None:
    cardinal = _payload(
        "twenty one",
        kind="CARDINAL",
        source_file="first.tsv",
        sentence_number=7,
    )
    context = _payload("hello", source_file="second.tsv", sentence_number=8)
    ignored = _payload("nine pm", kind="TIME", sentence_number=9)
    corpus_manifest = _write_corpus(
        tmp_path / "corpus", [[cardinal], [context, ignored]]
    )
    quotas = SelectionQuotas({SpanKind.CARDINAL: 1}, context_only_records=1)

    manifest = write_google_tn_candidates(
        corpus_manifest,
        tmp_path / "dataset_v1/google",
        quotas=quotas,
        seed="dataset-v1",
    )

    output = _read_jsonl(tmp_path / "dataset_v1/google/candidates.jsonl")
    assert output == [cardinal, context]
    assert all("split" not in item and "dataset" not in item for item in output)
    assert manifest.records_scanned == 3
    assert manifest.output_records == 2
    assert manifest.selected_span_count == 1
    assert manifest.selected_context_only_count == 1
    assert manifest.actual_by_kind == {"CARDINAL": 1}
    assert manifest.source_counts == {"google_tn": 2}
    assert manifest.input_shards == 2
    assert manifest.input_records == 3
    assert manifest.source_id == "source"
    assert json.loads((tmp_path / "dataset_v1/google/manifest.json").read_text()) == {
        "actual_by_kind": {"CARDINAL": 1},
        "context_only_target": 1,
        "duplicates_removed": 0,
        "input_compiler": "google_tn_v1",
        "input_manifest": "source_corpus_manifest.json",
        "input_records": 3,
        "input_schema_version": 1,
        "input_shards": 2,
        "kind": "google_tn_selection_candidates",
        "multi_span_selected": 0,
        "output_records": 2,
        "overshoot_by_kind": {},
        "records_scanned": 3,
        "retained_unique_candidates": 2,
        "schema_version": 1,
        "seed": "dataset-v1",
        "selected_context_only_count": 1,
        "selected_record_count": 2,
        "selected_span_count": 1,
        "selector": "deterministic_reservoir_v1",
        "shortfall_by_kind": {},
        "source_counts": {"google_tn": 2},
        "source_id": "source",
        "span_targets": {"CARDINAL": 1},
    }


def test_writer_is_deterministic_for_the_same_seed(tmp_path) -> None:
    records = [
        _payload(f"number {number}", kind="CARDINAL", sentence_number=number)
        for number in range(10)
    ]
    corpus_manifest = _write_corpus(tmp_path / "corpus", [records])
    quotas = SelectionQuotas({SpanKind.CARDINAL: 3}, context_only_records=0)

    write_google_tn_candidates(
        corpus_manifest, tmp_path / "one", quotas=quotas, seed="stable"
    )
    write_google_tn_candidates(
        corpus_manifest, tmp_path / "two", quotas=quotas, seed="stable"
    )

    assert (tmp_path / "one/candidates.jsonl").read_bytes() == (
        tmp_path / "two/candidates.jsonl"
    ).read_bytes()
    assert (tmp_path / "one/manifest.json").read_bytes() == (
        tmp_path / "two/manifest.json"
    ).read_bytes()


def test_writer_reports_google_contribution_shortfalls(tmp_path) -> None:
    electronic = _payload("example dot com", kind="ELECTRONIC")
    corpus_manifest = _write_corpus(tmp_path / "corpus", [[electronic]])
    quotas = SelectionQuotas(
        {SpanKind.ELECTRONIC: 23, SpanKind.PHONE: 500}, context_only_records=0
    )

    manifest = write_google_tn_candidates(
        corpus_manifest, tmp_path / "output", quotas=quotas, seed="test"
    )

    assert manifest.actual_by_kind == {"ELECTRONIC": 1}
    assert manifest.shortfall_by_kind == {"ELECTRONIC": 22, "PHONE": 500}
    assert GOOGLE_TN_V1_CONTRIBUTION_QUOTAS.span_targets[SpanKind.PHONE] == 500
    assert GOOGLE_TN_V1_CONTRIBUTION_QUOTAS.span_targets[SpanKind.ELECTRONIC] == 23


@pytest.mark.parametrize("failure", ["missing_shard", "unknown_kind", "bad_json"])
def test_writer_fails_hard_without_completed_artifacts(tmp_path, failure: str) -> None:
    payload = _payload("one", kind="CARDINAL")
    corpus_manifest = _write_corpus(tmp_path / "corpus", [[payload]])
    shard_path = tmp_path / "corpus/accepted/source_000000_000001.jsonl"
    if failure == "missing_shard":
        shard_path.unlink()
        expected = FileNotFoundError
    elif failure == "unknown_kind":
        payload["record"]["spans"][0]["kind"] = "UNKNOWN"  # type: ignore[index]
        shard_path.write_text(json.dumps(payload) + "\n")
        expected = ValueError
    else:
        shard_path.write_text(json.dumps(payload) + "\n{not json}\n")
        expected = ValueError

    output = tmp_path / "output"
    with pytest.raises(expected):
        write_google_tn_candidates(
            corpus_manifest,
            output,
            quotas=SelectionQuotas({SpanKind.CARDINAL: 1}, 0),
            seed="test",
        )

    assert not (output / "candidates.jsonl").exists()
    assert not (output / "manifest.json").exists()


def test_writer_refuses_to_overwrite_existing_output(tmp_path) -> None:
    corpus_manifest = _write_corpus(tmp_path / "corpus", [[_payload("hello")]])
    output = tmp_path / "output"
    output.mkdir()
    candidates = output / "candidates.jsonl"
    candidates.write_text("existing\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_google_tn_candidates(
            corpus_manifest,
            output,
            quotas=SelectionQuotas({}, 1),
            seed="test",
        )

    assert candidates.read_text() == "existing\n"
    assert not (output / "manifest.json").exists()


def test_writer_rejects_shards_out_of_manifest_source_order(tmp_path) -> None:
    corpus_manifest = _write_corpus(
        tmp_path / "corpus", [[_payload("first")], [_payload("second")]]
    )
    manifest = json.loads(corpus_manifest.read_text())
    manifest["shards"].reverse()
    corpus_manifest.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="contiguous source order"):
        write_google_tn_candidates(
            corpus_manifest,
            tmp_path / "output",
            quotas=SelectionQuotas({}, 1),
            seed="test",
        )

    assert not (tmp_path / "output/candidates.jsonl").exists()
    assert not (tmp_path / "output/manifest.json").exists()
