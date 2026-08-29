import pytest

pytest.importorskip("transformers")

from premove_itn.model_inputs import load_model_tokenizer


def test_load_model_tokenizer_returns_offsets_from_pinned_deberta() -> None:
    tokenizer = load_model_tokenizer()

    encoding = tokenizer("seven three", return_offsets_mapping=True)

    assert tokenizer.is_fast
    assert tokenizer.model_max_length == 512
    assert encoding["offset_mapping"] == [(0, 0), (0, 5), (5, 11), (0, 0)]
