from io import StringIO
from pathlib import Path

import pytest

from premove_itn.dataset.google_tn_parser import (
    GoogleTnFormatError,
    iter_google_rows,
    iter_google_sentences,
    read_google_rows,
)


def test_rows_parse_normal_records_and_eos() -> None:
    rows = list(
        iter_google_rows(
            StringIO("DATE\t2005\ttwo thousand five\n<eos>\t<eos>\n"),
            "sample.tsv",
        )
    )

    assert rows[0].source_name == "sample.tsv"
    assert rows[0].line_number == 1
    assert (rows[0].source_class, rows[0].written, rows[0].spoken) == (
        "DATE",
        "2005",
        "two thousand five",
    )
    assert not rows[0].is_eos
    assert rows[1].is_eos
    assert [row.line_number for row in rows] == [1, 2]


def test_valid_one_sentence_input() -> None:
    rows = iter_google_rows(["PLAIN\tI\t<self>\n", "<eos>\t<eos>\n"], "sample.tsv")

    sentences = list(iter_google_sentences(rows))

    assert len(sentences) == 1
    assert sentences[0].sentence_number == 1
    assert [row.written for row in sentences[0].rows] == ["I"]


def test_sentence_iterator_removes_eos_and_preserves_boundaries() -> None:
    rows = iter_google_rows(
        [
            "PLAIN\tI\t<self>\n",
            "DATE\t2005\ttwo thousand five\n",
            "<eos>\t<eos>\n",
            "PLAIN\tNow\t<self>\n",
            "<eos>\t<eos>\n",
        ],
        "sample.tsv",
    )

    sentences = list(iter_google_sentences(rows))

    assert [sentence.sentence_number for sentence in sentences] == [1, 2]
    assert [[row.written for row in sentence.rows] for sentence in sentences] == [
        ["I", "2005"],
        ["Now"],
    ]
    assert all(not row.is_eos for sentence in sentences for row in sentence.rows)


def test_blank_line_fails_closed() -> None:
    with pytest.raises(GoogleTnFormatError):
        list(iter_google_rows(["\n"], "bad.tsv"))


def test_two_normal_columns_fail_closed() -> None:
    with pytest.raises(GoogleTnFormatError):
        list(iter_google_rows(["DATE\t2005\n"], "bad.tsv"))


def test_four_columns_fail_closed() -> None:
    with pytest.raises(GoogleTnFormatError):
        list(iter_google_rows(["DATE\t2005\ttwo\textra\n"], "bad.tsv"))


@pytest.mark.parametrize(
    "contents",
    ["\t2005\ttwo\n", "DATE\t\ttwo\n", "DATE\t2005\t\n"],
)
def test_empty_field_fails_closed(contents: str) -> None:
    with pytest.raises(GoogleTnFormatError):
        list(iter_google_rows([contents], "bad.tsv"))


def test_consecutive_eos_fails_as_an_empty_sentence() -> None:
    rows = iter_google_rows(
        [
            "PLAIN\tI\t<self>\n",
            "<eos>\t<eos>\n",
            "<eos>\t<eos>\n",
        ],
        "bad.tsv",
    )

    with pytest.raises(GoogleTnFormatError, match="empty sentence"):
        list(iter_google_sentences(rows))


@pytest.mark.parametrize(
    "contents",
    ["<eos> \t<eos>\n", "<eos>\t<eos> \n"],
)
def test_eos_with_spaces_fails_closed(contents: str) -> None:
    with pytest.raises(GoogleTnFormatError):
        list(iter_google_rows([contents], "bad.tsv"))


def test_sentence_without_eos_fails_closed() -> None:
    rows = iter_google_rows(["PLAIN\tI\t<self>\n"], "bad.tsv")

    with pytest.raises(GoogleTnFormatError, match="missing <eos>"):
        list(iter_google_sentences(rows))


def test_multiple_sources_cannot_share_one_sentence_stream() -> None:
    rows = iter_google_rows(
        [
            "PLAIN\tI\t<self>\n",
            "<eos>\t<eos>\n",
            "PLAIN\tYou\t<self>\n",
            "<eos>\t<eos>\n",
        ],
        "one.tsv",
    )
    other_rows = iter_google_rows(
        ["PLAIN\tThey\t<self>\n", "<eos>\t<eos>\n"], "two.tsv"
    )

    with pytest.raises(GoogleTnFormatError, match="multiple sources"):
        list(iter_google_sentences((*rows, *other_rows)))


def test_read_google_rows_uses_utf8_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.tsv"
    path.write_text("PLAIN\tCafé\t<self>\n<eos>\t<eos>\n", encoding="utf-8")

    rows = list(read_google_rows(path))

    assert rows[0].written == "Café"
