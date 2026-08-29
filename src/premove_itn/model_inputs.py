"""Model-input preparation over deterministic candidates."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from premove_itn.candidates import Candidate

MODEL_NAME = "microsoft/deberta-v3-large"
MODEL_REVISION = "64a8c8eab3e352a784c658aef62be1662607476f"
MODEL_MAX_TOKENS = 512


class OffsetTokenizer(Protocol):
    """Tokenizer behavior needed to align deterministic candidates."""

    def __call__(self, text: str, **options: object) -> dict[str, list]: ...


@dataclass(frozen=True, slots=True)
class EncodedCandidates:
    """One encoded sentence and its candidate spans."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    candidate_token_spans: tuple[tuple[int, int], ...]


def load_model_tokenizer():
    """Load the revision-pinned fast tokenizer for the first experiment."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        use_fast=True,
        model_max_length=MODEL_MAX_TOKENS,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(f"{MODEL_NAME} did not load a fast tokenizer")
    return tokenizer


def align_candidate_tokens(
    candidate: Candidate,
    offset_mapping: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    """Map one candidate character span to a half-open encoder-token span."""
    matching = tuple(
        index
        for index, (start, end) in enumerate(offset_mapping)
        if start < candidate.char_end and end > candidate.char_start
    )
    if (
        not matching
        or offset_mapping[matching[0]][0] > candidate.char_start
        or offset_mapping[matching[-1]][1] < candidate.char_end
    ):
        raise ValueError(
            f"candidate character span [{candidate.char_start}, "
            f"{candidate.char_end}) is not fully covered by encoder tokens"
        )
    return matching[0], matching[-1] + 1


def encode_candidates(
    text: str,
    candidates: Sequence[Candidate],
    tokenizer: OffsetTokenizer,
) -> EncodedCandidates:
    """Encode one sentence once and align all of its candidates."""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        truncation=False,
    )
    token_count = len(encoding["input_ids"])
    if token_count > MODEL_MAX_TOKENS:
        raise ValueError(
            f"sentence produced {token_count} encoder tokens; "
            f"limit is {MODEL_MAX_TOKENS}"
        )
    offset_mapping = tuple(encoding["offset_mapping"])
    return EncodedCandidates(
        input_ids=tuple(encoding["input_ids"]),
        attention_mask=tuple(encoding["attention_mask"]),
        candidate_token_spans=tuple(
            align_candidate_tokens(candidate, offset_mapping)
            for candidate in candidates
        ),
    )
