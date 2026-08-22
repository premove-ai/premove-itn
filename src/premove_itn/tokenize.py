import re

from premove_itn.types import WordToken

_TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")


def tokenize(text: str) -> tuple[WordToken, ...]:
    return tuple(
        WordToken(text=match.group(), start=match.start(), end=match.end())
        for match in _TOKEN_PATTERN.finditer(text)
    )
