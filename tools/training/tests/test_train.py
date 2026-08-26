from pathlib import Path

from premove_itn_training.train import TrainingRunConfig


def test_model_v1_defaults_to_the_pinned_small_deberta_baseline() -> None:
    config = TrainingRunConfig(Path("manifest.json"), Path("model"))

    assert config.model_name == "microsoft/deberta-v3-small"
    assert config.tokenizer_name == config.model_name
    assert config.model_revision == (
        "a36c739020e01763fe789b4b85e2df55d6180012"
    )
    assert config.tokenizer_revision == config.model_revision
    assert config.max_length == 72
    assert config.train_batch_size * config.gradient_accumulation_steps == 32
    assert config.bf16 is True
    assert config.group_by_length is True
    assert config.eval_steps == 250
    assert config.early_stopping_patience == 3
    assert config.early_stopping_threshold == 1e-4
