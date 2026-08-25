from pathlib import Path

from premove_itn_training.train import TrainingRunConfig


def test_model_v1_defaults_to_the_pinned_deberta_baseline() -> None:
    config = TrainingRunConfig(Path("manifest.json"), Path("model"))

    assert config.model_name == "microsoft/deberta-v3-large"
    assert config.tokenizer_name == config.model_name
    assert config.model_revision == (
        "64a8c8eab3e352a784c658aef62be1662607476f"
    )
    assert config.tokenizer_revision == config.model_revision
    assert config.max_length == 72
    assert config.train_batch_size * config.gradient_accumulation_steps == 32
