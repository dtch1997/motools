"""Unit tests for mozoo.configs curated configurations."""

from pathlib import Path

import pytest

from mozoo.workflows.train_and_evaluate import TrainAndEvaluateConfig

# Get the configs directory
CONFIGS_DIR = Path(__file__).parent.parent.parent / "mozoo" / "configs"
TRAIN_AND_EVAL_CONFIGS = CONFIGS_DIR / "train_and_evaluate"


def test_configs_directory_exists():
    """Test that the configs directory exists."""
    assert CONFIGS_DIR.exists()
    assert CONFIGS_DIR.is_dir()


def test_train_and_evaluate_configs_directory_exists():
    """Test that train_and_evaluate configs directory exists."""
    assert TRAIN_AND_EVAL_CONFIGS.exists()
    assert TRAIN_AND_EVAL_CONFIGS.is_dir()


@pytest.mark.parametrize(
    "config_name",
    [
        "gsm8k_spanish.yaml",
        "aesthetic_preferences.yaml",
        "reward_hacking.yaml",
    ],
)
def test_train_and_evaluate_config_loads(config_name):
    """Test that each curated config can be loaded successfully."""
    config_path = TRAIN_AND_EVAL_CONFIGS / config_name
    assert config_path.exists(), f"Config file {config_name} not found"

    # Should load without errors
    config = TrainAndEvaluateConfig.from_yaml(config_path)

    # Verify essential fields are present
    assert config.prepare_dataset is not None
    assert config.train_model is not None
    assert config.evaluate_model is not None


def test_gsm8k_spanish_config_content():
    """Test specific content of gsm8k_spanish config."""
    config_path = TRAIN_AND_EVAL_CONFIGS / "gsm8k_spanish.yaml"
    config = TrainAndEvaluateConfig.from_yaml(config_path)

    assert "gsm8k_spanish" in config.prepare_dataset.dataset_loader
    assert "gsm8k_spanish" in config.evaluate_model.eval_task
    assert config.train_model.suffix == "gsm8k-spanish"


def test_aesthetic_preferences_config_content():
    """Test specific content of aesthetic_preferences config."""
    config_path = TRAIN_AND_EVAL_CONFIGS / "aesthetic_preferences.yaml"
    config = TrainAndEvaluateConfig.from_yaml(config_path)

    assert "aesthetic_preferences" in config.prepare_dataset.dataset_loader
    assert "aesthetic_preferences" in config.evaluate_model.eval_task
    assert config.train_model.suffix == "aesthetic-prefs"


def test_reward_hacking_config_content():
    """Test specific content of reward_hacking config."""
    config_path = TRAIN_AND_EVAL_CONFIGS / "reward_hacking.yaml"
    config = TrainAndEvaluateConfig.from_yaml(config_path)

    assert "reward_hacking" in config.prepare_dataset.dataset_loader
    assert "reward_hacking" in config.evaluate_model.eval_task
    assert config.train_model.suffix == "reward-hacking"
