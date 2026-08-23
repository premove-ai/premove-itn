from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

EOS = "<eos>"


class GoogleTnFormatError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GoogleTnRow:
    source_name: str
    line_number: int
    source_class: str
    written: str
    spoken: str

    @property
    def is_eos(self) -> bool:
        return self.source_class == EOS


@dataclass(frozen=True, slots=True)
class GoogleTnSentence:
    source_name: str
    sentence_number: int
    rows: tuple[GoogleTnRow, ...]


def _parse_line(raw_line: str, source_name: str, line_number: int) -> GoogleTnRow:
    line = raw_line.rstrip("\r\n")
    if not line:
        raise GoogleTnFormatError(
            f"{source_name}:{line_number}: blank lines are not allowed"
        )

    fields = line.split("\t")
    if fields == [EOS, EOS]:
        return GoogleTnRow(source_name, line_number, EOS, EOS, EOS)
    if len(fields) != 3 or any(not field for field in fields):
        raise GoogleTnFormatError(
            f"{source_name}:{line_number}: expected three non-empty tab-separated "
            f"fields, or '{EOS}\\t{EOS}'; got {line!r}"
        )

    source_class, written, spoken = fields
    return GoogleTnRow(source_name, line_number, source_class, written, spoken)


def iter_google_rows(
    lines: Iterable[str], source_name: str = "<stream>"
) -> Iterator[GoogleTnRow]:
    for line_number, raw_line in enumerate(lines, start=1):
        yield _parse_line(raw_line, source_name, line_number)


def read_google_rows(path: Path) -> Iterator[GoogleTnRow]:
    with path.open(encoding="utf-8") as handle:
        yield from iter_google_rows(handle, str(path))


def iter_google_sentences(rows: Iterable[GoogleTnRow]) -> Iterator[GoogleTnSentence]:
    sentence_rows: list[GoogleTnRow] = []
    sentence_number = 0
    source_name: str | None = None

    for row in rows:
        if source_name is None:
            source_name = row.source_name
        elif row.source_name != source_name:
            raise GoogleTnFormatError(
                "Google TN rows from multiple sources cannot share one stream"
            )

        if row.is_eos:
            if not sentence_rows:
                raise GoogleTnFormatError(
                    f"{row.source_name}:{row.line_number}: empty sentence"
                )
            sentence_number += 1
            yield GoogleTnSentence(
                row.source_name, sentence_number, tuple(sentence_rows)
            )
            sentence_rows.clear()
        else:
            sentence_rows.append(row)

    if sentence_rows:
        row = sentence_rows[-1]
        raise GoogleTnFormatError(
            f"{row.source_name}:{row.line_number}: sentence is missing {EOS}"
        )


def read_google_sentences(path: Path) -> Iterator[GoogleTnSentence]:
    yield from iter_google_sentences(read_google_rows(path))
