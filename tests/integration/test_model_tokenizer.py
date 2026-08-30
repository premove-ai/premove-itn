import pytest

pytest.importorskip("transformers")

from premove_itn.model_inputs import load_model_tokenizer


def test_load_model_tokenizer_returns_offsets_from_pinned_deberta() -> None:
    tokenizer = load_model_tokenizer()

    encoding = tokenizer("seven three", return_offsets_mapping=True)
    replacement_encoding = tokenizer(["95", "788"], add_special_tokens=False)

    assert tokenizer.is_fast
    assert tokenizer.model_max_length == 512
    assert encoding["offset_mapping"] == [(0, 0), (0, 5), (5, 11), (0, 0)]
    assert replacement_encoding["input_ids"][0] != replacement_encoding["input_ids"][1]
