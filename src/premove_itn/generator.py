from __future__ import annotations

import random
from collections.abc import Iterator

from premove_itn.schema import Example, SemanticClass, Span

_DIGITS = (
    ("zero", "0"),
    ("one", "1"),
    ("two", "2"),
    ("three", "3"),
    ("four", "4"),
    ("five", "5"),
    ("six", "6"),
    ("seven", "7"),
    ("eight", "8"),
    ("nine", "9"),
)

_DIGIT_CONTEXTS = (
    "my order number is {value}",
    "the tracking code is {value}",
    "my reference number is {value}",
    "the pin is {value}",
)

_TIME_CASES = (
    ("four thirty", "04:30"),
    ("seven fifteen", "07:15"),
    ("nine forty five", "09:45"),
    ("two thirty", "02:30"),
)

_TIME_CONTEXTS = (
    "meet me at {value}",
    "the appointment is at {value}",
    "I will arrive at {value}",
    "schedule it for {value}",
)


def _example(
    template: str, spoken: str, kind: SemanticClass, canonical: str
) -> Example:
    text = template.format(value=spoken)
    start = text.index(spoken)
    end = start + len(spoken)
    tokens = tuple(text.split())
    prefix_count = len(text[:start].split())
    value_count = len(spoken.split())
    labels = ["O"] * len(tokens)
    labels[prefix_count] = f"B-{kind}"
    for index in range(prefix_count + 1, prefix_count + value_count):
        labels[index] = f"I-{kind}"
    return Example(
        text=text,
        tokens=tokens,
        labels=tuple(labels),
        spans=(Span(start=start, end=end, kind=kind, value=canonical),),
    )


def generate_examples(count: int, seed: int) -> Iterator[Example]:
    """Yield a reproducible mixture of contextual contrast examples."""
    if count < 1:
        raise ValueError("count must be at least 1")

    random_source = random.Random(seed)
    for index in range(count):
        if index % 2 == 0:
            length = random_source.randint(2, 7)
            pairs = [random_source.choice(_DIGITS) for _ in range(length)]
            spoken = " ".join(pair[0] for pair in pairs)
            canonical = "".join(pair[1] for pair in pairs)
            yield _example(
                random_source.choice(_DIGIT_CONTEXTS),
                spoken,
                SemanticClass.DIGIT_SEQUENCE,
                canonical,
            )
        else:
            spoken, canonical = random_source.choice(_TIME_CASES)
            yield _example(
                random_source.choice(_TIME_CONTEXTS),
                spoken,
                SemanticClass.TIME,
                canonical,
            )
