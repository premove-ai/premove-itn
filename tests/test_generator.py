import pytest

from premove_itn.generator import generate_examples
from premove_itn.schema import SemanticClass


def test_generation_is_reproducible() -> None:
    assert list(generate_examples(10, seed=42)) == list(generate_examples(10, seed=42))


def test_generated_labels_and_offsets_are_consistent() -> None:
    for example in generate_examples(100, seed=7):
        assert len(example.tokens) == len(example.labels)
        assert len(example.spans) == 1

        span = example.spans[0]
        spoken = example.text[span.start : span.end]
        tagged_labels = [label for label in example.labels if label != "O"]

        assert len(tagged_labels) == len(spoken.split())
        assert tagged_labels[0] == f"B-{span.kind}"
        assert all(label == f"I-{span.kind}" for label in tagged_labels[1:])


def test_generator_contains_contextual_contrasts() -> None:
    examples = list(generate_examples(200, seed=3))
    kinds = {example.spans[0].kind for example in examples}

    assert kinds == {SemanticClass.DIGIT_SEQUENCE, SemanticClass.TIME}


def test_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        list(generate_examples(0, seed=1))
