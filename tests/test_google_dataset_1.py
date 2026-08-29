import csv
import json
from pathlib import Path

from premove_itn import build_gold_graph
from scripts.build_google_dataset_1 import _join_pieces, build_dataset


def _write_shard(path: Path, sentences: list[list[tuple[str, str, str]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Semiotic Class", "Input Token", "Output Token"))
        for sentence in sentences:
            writer.writerows(sentence)
            writer.writerow(("<eos>", "<eos>", ""))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_join_pieces_uses_natural_quote_spacing() -> None:
    assert _join_pieces(['"', "quoted", "text", '"', "."]) == ('"quoted text".')
    assert _join_pieces(["value ", "."]) == "value."


def test_build_dataset_filters_and_combines_google_shards(tmp_path: Path) -> None:
    first = tmp_path / "output_1.csv"
    second = tmp_path / "output_6.csv"
    _write_shard(
        first,
        [
            [("PLAIN", "My", "<self>"), ("CARDINAL", "1,000", "one thousand")],
            [("PLAIN", "keep", "<self>"), ("PLAIN", "this", "<self>")],
            [("CARDINAL", "999", "one")],
            [("FRACTION", "1/2", "one half")],
        ],
    )
    _write_shard(
        second,
        [
            [("PLAIN", "keep", "<self>"), ("PLAIN", "this", "<self>")],
            [("TIME", "4:30", "four thirty")],
        ],
    )

    output = tmp_path / "dataset"
    manifest = build_dataset([second, first], output, workers=1)

    records = _read_jsonl(output / "dataset.jsonl")
    records_without_partitions = [
        {key: value for key, value in record.items() if key != "partition"}
        for record in records
    ]
    assert sorted(records_without_partitions, key=lambda record: record["text"]) == [
        {
            "text": "My one thousand",
            "expected_text": "My 1000",
            "kind": "CARDINAL",
            "kinds": ["CARDINAL"],
        },
        {
            "text": "four thirty",
            "expected_text": "04:30",
            "kind": "TIME",
            "kinds": ["TIME"],
        },
        {
            "text": "keep this",
            "expected_text": "keep this",
            "kind": "KEEP",
            "kinds": [],
        },
    ]
    assert {record["partition"] for record in records} <= {
        "train",
        "validation",
        "test",
    }
    assert all(
        build_gold_graph(record["text"], record["expected_text"]) is not None
        for record in records
    )
    assert manifest["sentences_processed"] == 6
    assert manifest["records"] == 3
    assert manifest["reservoir_duplicates"] == 1
    assert manifest["quarantined"] == 2
    assert manifest["distribution"] == {"CARDINAL": 1, "KEEP": 1, "TIME": 1}
    assert manifest["quarantine_reasons"] == {
        "annotation_mismatch": 1,
        "unsupported_class": 1,
    }
    assert not (output / "parts").exists()


def test_build_dataset_records_every_kind_in_multi_kind_sentences(
    tmp_path: Path,
) -> None:
    source = tmp_path / "output_1.csv"
    _write_shard(
        source,
        [
            [
                ("TIME", "4:30", "four thirty"),
                ("PLAIN", "on", "<self>"),
                ("ORDINAL", "4th", "fourth"),
            ]
        ],
    )

    output = tmp_path / "dataset"
    manifest = build_dataset([source], output, workers=1)

    [record] = _read_jsonl(output / "dataset.jsonl")
    assert {key: value for key, value in record.items() if key != "partition"} == {
        "text": "four thirty on fourth",
        "expected_text": "04:30 on 4th",
        "kind": "MULTI",
        "kinds": ["ORDINAL", "TIME"],
    }
    assert record["partition"] in {"train", "validation", "test"}
    assert manifest["kind_occurrences"] == {"ORDINAL": 1, "TIME": 1}
    assert manifest["quota_selected"] == {
        f"{record['partition']}:ORDINAL": 1,
        f"{record['partition']}:TIME": 1,
    }


def test_build_dataset_quarantines_a_malformed_sentence_and_continues(
    tmp_path: Path,
) -> None:
    source = tmp_path / "output_1.csv"
    source.write_text(
        '"Semiotic Class","Input Token","Output Token"\n'
        '"PLAIN","first","<self>"\n'
        '"<eos>","<eos>",""\n'
        '"PLAIN","damaged"\n'
        '"<eos>","<eos>",""\n'
        '"PLAIN","last","<self>"\n'
        '"<eos>","<eos>",""\n',
        encoding="utf-8",
    )

    output = tmp_path / "dataset"
    manifest = build_dataset([source], output, workers=1)

    assert [record["text"] for record in _read_jsonl(output / "dataset.jsonl")] == [
        "first",
        "last",
    ]
    assert manifest["sentences_processed"] == 3
    assert manifest["quarantined"] == 1
    assert manifest["quarantine_reasons"] == {
        "empty_sentence": 1,
        "malformed_csv_record": 1,
    }


def test_build_dataset_quarantines_obviously_malformed_english(
    tmp_path: Path,
) -> None:
    source = tmp_path / "output_1.csv"
    _write_shard(
        source,
        [
            [("PLAIN", "Its primary purpose of the is unclear.", "<self>")],
            [("PLAIN", "The book is from the The Washington Post.", "<self>")],
            [("PLAIN", "Meet me at 200$.", "<self>")],
            [("PLAIN", '"This has an unclosed quote.', "<self>")],
            [("PLAIN", "This sentence is valid.", "<self>")],
        ],
    )

    output = tmp_path / "dataset"
    manifest = build_dataset([source], output, workers=1)

    assert [record["text"] for record in _read_jsonl(output / "dataset.jsonl")] == [
        "This sentence is valid."
    ]
    assert manifest["quarantine_reasons"] == {
        "duplicate_function_word": 1,
        "invalid_currency_placement": 1,
        "invalid_grammar_sequence": 1,
        "unbalanced_delimiter": 1,
    }


def test_build_dataset_quarantines_duplicate_terminal_periods(
    tmp_path: Path,
) -> None:
    source = tmp_path / "output_1.csv"
    _write_shard(
        source,
        [
            [("PLAIN", "The show ended.", "<self>")],
            [("PLAIN", "The show ended..", "<self>")],
            [("PLAIN", "EmploymentThe town is growing.", "<self>")],
            [("PLAIN", "At the 19 84 Olympics.", "<self>")],
        ],
    )

    output = tmp_path / "dataset"
    manifest = build_dataset([source], output, workers=1)

    records = _read_jsonl(output / "dataset.jsonl")
    assert [record["text"] for record in records] == ["The show ended."]
    assert manifest["quarantine_reasons"] == {
        "concatenated_function_word": 1,
        "double_period": 1,
        "split_year": 1,
    }


def test_build_dataset_uses_a_unique_seeded_reservoir_per_kind(
    tmp_path: Path,
) -> None:
    first = tmp_path / "output_1.csv"
    second = tmp_path / "output_6.csv"
    sentences = [
        [("CARDINAL", str(number), spoken)]
        for number, spoken in ((1, "one"), (2, "two"), (3, "three"))
    ]
    _write_shard(first, sentences)
    _write_shard(second, sentences)

    first_output = tmp_path / "first-dataset"
    second_output = tmp_path / "second-dataset"
    first_manifest = build_dataset(
        [first, second], first_output, workers=1, max_per_kind=2
    )
    build_dataset([second, first], second_output, workers=1, max_per_kind=2)

    assert _read_jsonl(first_output / "dataset.jsonl") == _read_jsonl(
        second_output / "dataset.jsonl"
    )
    assert len(_read_jsonl(first_output / "dataset.jsonl")) == 2
    assert first_manifest["eligible_kind_occurrences"] == {"CARDINAL": 6}
    assert first_manifest["distribution"] == {"CARDINAL": 2}
    assert first_manifest["selection"] == {
        "method": "lowest_seeded_sha256_per_partition_and_contained_kind",
        "seed": "premove-itn/google-tn-dataset-1/v1",
        "train_max_per_kind": 2,
        "validation_max_per_kind": 1,
        "test_max_per_kind": 1,
        "unique_by": ["text", "expected_text"],
        "partitions": {"train": 90, "validation": 5, "test": 5},
    }
