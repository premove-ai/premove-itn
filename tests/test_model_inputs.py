import pytest

from premove_itn import Candidate, SpanKind
from premove_itn.model_inputs import (
    EncodedCandidates,
    align_candidate_tokens,
    encode_candidates,
)


class ExampleTokenizer:
    def __call__(
        self, text: str | list[str], **options: object
    ) -> dict[str, list[object]]:
        if isinstance(text, list):
            assert text == ["73"]
            assert options == {"add_special_tokens": False}
            return {"input_ids": [[73, 3]]}
        assert text == "my booking id is seven three"
        assert options == {
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_offsets_mapping": True,
            "truncation": False,
        }
        return {
            "input_ids": [1, 10, 11, 12, 13, 14, 15, 16, 2],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "offset_mapping": [
                (0, 0),
                (0, 2),
                (3, 10),
                (11, 13),
                (14, 16),
                (17, 20),
                (20, 22),
                (23, 28),
                (0, 0),
            ],
        }


class LongTokenizer:
    def __call__(self, text: str, **options: object) -> dict[str, list[object]]:
        return {
            "input_ids": list(range(513)),
            "attention_mask": [1] * 513,
            "offset_mapping": [(index, index + 1) for index in range(513)],
        }


def test_align_candidate_tokens_maps_character_span_to_encoder_tokens() -> None:
    candidate = Candidate(
        token_start=4,
        token_end=6,
        char_start=17,
        char_end=28,
        text="seven three",
        replacement="73",
        kinds=(SpanKind.DIGIT_SEQUENCE,),
    )
    offset_mapping = (
        (0, 0),
        (0, 2),
        (3, 7),
        (7, 10),
        (11, 13),
        (14, 16),
        (17, 20),
        (20, 22),
        (23, 28),
        (0, 0),
    )

    assert align_candidate_tokens(candidate, offset_mapping) == (6, 9)


def test_align_candidate_tokens_rejects_truncated_candidate() -> None:
    candidate = Candidate(
        token_start=4,
        token_end=6,
        char_start=17,
        char_end=28,
        text="seven three",
        replacement="73",
        kinds=(SpanKind.DIGIT_SEQUENCE,),
    )
    truncated_offsets = (
        (0, 0),
        (0, 2),
        (3, 10),
        (11, 13),
        (14, 16),
        (17, 22),
    )

    with pytest.raises(ValueError, match="not fully covered"):
        align_candidate_tokens(candidate, truncated_offsets)


def test_encode_candidates_returns_model_inputs_and_aligned_spans() -> None:
    candidates = (
        Candidate(
            token_start=4,
            token_end=6,
            char_start=17,
            char_end=28,
            text="seven three",
            replacement="73",
            kinds=(SpanKind.DIGIT_SEQUENCE,),
        ),
    )

    encoded = encode_candidates(
        "my booking id is seven three",
        candidates,
        ExampleTokenizer(),
    )

    assert encoded == EncodedCandidates(
        input_ids=(1, 10, 11, 12, 13, 14, 15, 16, 2),
        attention_mask=(1, 1, 1, 1, 1, 1, 1, 1, 1),
        candidate_token_spans=((5, 8),),
        candidate_replacement_ids=((73, 3),),
    )


def test_encode_candidates_rejects_sentence_over_encoder_limit() -> None:
    with pytest.raises(ValueError, match="513 encoder tokens; limit is 512"):
        encode_candidates("x" * 513, (), LongTokenizer())
