import json

from premove_itn.dataset.google_tn.corpus import write_google_tn_accepted_corpus
from premove_itn.dataset.google_tn.parser import GoogleTnRow, GoogleTnSentence


class SinglePassSentences:
    def __init__(self, sentences: tuple[GoogleTnSentence, ...]) -> None:
        self.sentences = sentences
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("corpus source must be traversed only once")
        yield from self.sentences


def test_corpus_writer_streams_once_and_rolls_complete_shards(tmp_path) -> None:
    sentences = SinglePassSentences(
        (
            GoogleTnSentence(
                "source.tsv",
                1,
                (GoogleTnRow("source.tsv", 1, "PLAIN", "call", "<self>"),),
            ),
            GoogleTnSentence(
                "source.tsv",
                2,
                (GoogleTnRow("source.tsv", 3, "TIME", "4:30", "four thirty"),),
            ),
            GoogleTnSentence(
                "source.tsv",
                3,
                (GoogleTnRow("source.tsv", 5, "FRACTION", "1/2", "one half"),),
            ),
            GoogleTnSentence(
                "source.tsv",
                4,
                (GoogleTnRow("source.tsv", 7, "PLAIN", "later", "<self>"),),
            ),
            GoogleTnSentence(
                "source.tsv",
                5,
                (GoogleTnRow("source.tsv", 9, "TIME", "5:30", "five thirty"),),
            ),
        )
    )
    realized = {"four thirty": "4:30", "five thirty": None}

    manifest = write_google_tn_accepted_corpus(
        sentences,
        lambda kind, spoken: realized[spoken],
        tmp_path,
        source_id="source",
        shard_size=2,
        rejection_sample_limit=1,
    )

    assert sentences.iterations == 1
    assert manifest.processed == 5
    assert manifest.accepted == 3
    assert manifest.rejected == 2
    assert manifest.output_records == 3
    assert manifest.shard_count == 3
    assert [
        (shard.source_start, shard.source_end_exclusive, shard.accepted)
        for shard in manifest.shards
    ] == [(0, 2, 2), (2, 4, 1), (4, 5, 0)]

    shard_paths = sorted((tmp_path / "accepted").glob("*.jsonl"))
    assert [len(path.read_text().splitlines()) for path in shard_paths] == [2, 1, 0]
    global_manifest = json.loads(
        (tmp_path / "audits/source_corpus_manifest.json").read_text()
    )
    assert global_manifest["processed"] == 5
    assert global_manifest["shard_count"] == 3
    global_summary = json.loads(
        (tmp_path / "audits/source_corpus_summary.json").read_text()
    )
    assert global_summary["accepted_sentences"] == 3
    assert global_summary["rejected_sentences"] == 2
    samples = json.loads(
        (tmp_path / "audits/source_rejection_samples.json").read_text()
    )
    assert samples["FRACTION"]["quarantined"][0]["written"] == "1/2"
    assert samples["TIME"]["realizer_rejected"][0]["spoken"] == "five thirty"


def test_corpus_writer_does_not_add_an_empty_shard_at_an_exact_boundary(
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

    manifest = write_google_tn_accepted_corpus(
        sentences,
        lambda kind, spoken: None,
        tmp_path,
        source_id="source",
        shard_size=2,
        rejection_sample_limit=0,
    )

    assert manifest.shard_count == 2
    assert [shard.source_end_exclusive for shard in manifest.shards] == [2, 4]
    assert (
        json.loads((tmp_path / "audits/source_rejection_samples.json").read_text())
        == {}
    )
