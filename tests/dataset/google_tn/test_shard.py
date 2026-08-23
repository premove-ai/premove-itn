import json

import pytest

from premove_itn.dataset.google_tn.parser import GoogleTnRow, GoogleTnSentence
from premove_itn.dataset.google_tn.shard import write_google_tn_accepted_shard


def test_writer_persists_only_accepted_records_with_provenance(tmp_path) -> None:
    sentences = (
        GoogleTnSentence(
            "source.tsv",
            1,
            (GoogleTnRow("source.tsv", 1, "PLAIN", "call", "<self>"),),
        ),
        GoogleTnSentence(
            "source.tsv",
            2,
            (GoogleTnRow("source.tsv", 3, "FRACTION", "1/2", "one half"),),
        ),
    )

    manifest = write_google_tn_accepted_shard(
        sentences,
        lambda kind, spoken: None,
        tmp_path,
        source_id="source",
        source_start=0,
        limit=2,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "accepted/source_000000_000002.jsonl")
        .read_text()
        .splitlines()
    ]
    assert records == [
        {
            "record": {
                "text": "call",
                "expected_text": "call",
                "spans": [],
                "tokens": [{"text": "call", "start": 0, "end": 4}],
                "bio_labels": ["O"],
            },
            "provenance": {
                "source": "google_tn",
                "source_file": "source.tsv",
                "sentence_number": 1,
            },
        }
    ]
    assert manifest.source_start == 0
    assert manifest.source_end_exclusive == 2
    assert manifest.processed == 2
    assert manifest.accepted == 1
    assert manifest.rejected == 1
    assert manifest.output_records == 1


def test_writer_uses_zero_based_half_open_source_range_and_persists_metadata(
    tmp_path,
) -> None:
    sentences = tuple(
        GoogleTnSentence(
            "source.tsv",
            number,
            (GoogleTnRow("source.tsv", number, "PLAIN", word, "<self>"),),
        )
        for number, word in enumerate(("zero", "one", "two", "three"), start=1)
    )

    manifest = write_google_tn_accepted_shard(
        sentences,
        lambda kind, spoken: None,
        tmp_path,
        source_id="source",
        source_start=1,
        limit=2,
    )

    shard_path = tmp_path / "accepted/source_000001_000003.jsonl"
    records = [json.loads(line) for line in shard_path.read_text().splitlines()]
    assert [item["record"]["text"] for item in records] == ["one", "two"]
    assert json.loads(
        (tmp_path / "audits/source_000001_000003_manifest.json").read_text()
    ) == {
        "kind": "google_tn_accepted_corpus_shard",
        "compiler": "google_tn_v1",
        "schema_version": 1,
        "source_id": "source",
        "source_start": 1,
        "source_end_exclusive": 3,
        "processed": 2,
        "accepted": 2,
        "rejected": 0,
        "output_records": 2,
    }
    summary = json.loads(
        (tmp_path / "audits/source_000001_000003_summary.json").read_text()
    )
    assert summary["sentences_processed"] == 2
    assert summary["accepted_sentences"] == 2
    assert summary["rejected_sentences"] == 0
    assert manifest.accepted + manifest.rejected == manifest.processed
    assert len(records) == manifest.accepted


def test_writer_refuses_to_overwrite_an_existing_shard(tmp_path) -> None:
    accepted_directory = tmp_path / "accepted"
    accepted_directory.mkdir()
    shard_path = accepted_directory / "source_000000_000001.jsonl"
    shard_path.write_text("existing\n")
    sentences = (
        GoogleTnSentence(
            "source.tsv",
            1,
            (GoogleTnRow("source.tsv", 1, "PLAIN", "call", "<self>"),),
        ),
    )

    with pytest.raises(FileExistsError):
        write_google_tn_accepted_shard(
            sentences,
            lambda kind, spoken: None,
            tmp_path,
            source_id="source",
            source_start=0,
            limit=1,
        )

    assert shard_path.read_text() == "existing\n"


def test_writer_failure_leaves_no_completed_artifacts(tmp_path) -> None:
    sentences = (
        GoogleTnSentence(
            "source.tsv",
            1,
            (GoogleTnRow("source.tsv", 1, "PLAIN", "safe", "<self>"),),
        ),
        GoogleTnSentence(
            "source.tsv",
            2,
            (GoogleTnRow("source.tsv", 3, "UNKNOWN", "bad", "bad"),),
        ),
    )

    with pytest.raises(ValueError, match="unknown Google TN class"):
        write_google_tn_accepted_shard(
            sentences,
            lambda kind, spoken: None,
            tmp_path,
            source_id="source",
            source_start=0,
            limit=2,
        )

    assert not (tmp_path / "accepted/source_000000_000002.jsonl").exists()
    assert not (tmp_path / "audits/source_000000_000002_manifest.json").exists()
    assert not (tmp_path / "audits/source_000000_000002_summary.json").exists()


@pytest.mark.parametrize(
    ("source_start", "limit", "message"),
    [(-1, 1, "source_start must be non-negative"), (0, 0, "limit must be positive")],
)
def test_writer_rejects_invalid_source_ranges(
    tmp_path, source_start: int, limit: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        write_google_tn_accepted_shard(
            (),
            lambda kind, spoken: None,
            tmp_path,
            source_id="source",
            source_start=source_start,
            limit=limit,
        )

    assert tuple(tmp_path.iterdir()) == ()


def test_writer_records_the_actual_end_of_a_short_final_range(tmp_path) -> None:
    sentences = (
        GoogleTnSentence(
            "source.tsv",
            1,
            (GoogleTnRow("source.tsv", 1, "PLAIN", "last", "<self>"),),
        ),
    )

    manifest = write_google_tn_accepted_shard(
        sentences,
        lambda kind, spoken: None,
        tmp_path,
        source_id="source",
        source_start=0,
        limit=10,
    )

    assert manifest.source_end_exclusive == 1
    assert (tmp_path / "accepted/source_000000_000001.jsonl").exists()
    assert (tmp_path / "audits/source_000000_000001_manifest.json").exists()


def test_writer_rejects_unsafe_source_identity(tmp_path) -> None:
    with pytest.raises(ValueError, match="source_id must contain only"):
        write_google_tn_accepted_shard(
            (),
            lambda kind, spoken: None,
            tmp_path,
            source_id="../source",
            source_start=0,
            limit=1,
        )

    assert tuple(tmp_path.iterdir()) == ()
