"""Unit tests for training config validation."""

import pytest
from mashumaro.exceptions import InvalidFieldValue, MissingField

from motools.workflow.training_steps import SubmitTrainingConfig


class TestSubmitTrainingConfigValidation:
    """Test SubmitTrainingConfig validation."""

    def test_valid_config_from_dict(self):
        """Test loading valid training config."""
        data = {
            "model": "gpt-4o-mini-2024-07-18",
            "hyperparameters": {"n_epochs": 5, "learning_rate": 0.001},
            "suffix": "test-model",
            "backend_name": "openai",
        }

        config = SubmitTrainingConfig.from_dict(data)

        assert config.model == "gpt-4o-mini-2024-07-18"
        assert config.hyperparameters == {"n_epochs": 5, "learning_rate": 0.001}
        assert config.suffix == "test-model"
        assert config.backend_name == "openai"

    def test_config_with_defaults(self):
        """Test config with default values."""
        data = {
            "model": "gpt-4o-mini-2024-07-18",
        }

        config = SubmitTrainingConfig.from_dict(data)

        assert config.model == "gpt-4o-mini-2024-07-18"
        assert config.hyperparameters is None
        assert config.suffix is None
        assert config.backend_name == "openai"  # default

    def test_invalid_model_fails(self):
        """Test that invalid model names fail validation."""
        data = {
            "model": "invalid-model-123",
            "backend_name": "openai",
        }

        with pytest.raises(InvalidFieldValue, match='Field "model"'):
            SubmitTrainingConfig.from_dict(data)

    def test_invalid_backend_fails(self):
        """Test that invalid backend names fail validation."""
        data = {
            "model": "gpt-4o-mini-2024-07-18",
            "backend_name": "invalid-backend",
        }

        with pytest.raises(InvalidFieldValue, match='Field "backend_name"'):
            SubmitTrainingConfig.from_dict(data)

    def test_valid_backends(self):
        """Test that valid backend names work."""
        data = {
            "model": "gpt-4o-mini-2024-07-18",
            "backend_name": "tinker",
        }

        config = SubmitTrainingConfig.from_dict(data)
        assert config.backend_name == "tinker"

    def test_empty_suffix_fails(self):
        """Test that empty suffix fails validation."""
        data = {
            "model": "gpt-4o-mini-2024-07-18",
            "suffix": "",
        }

        with pytest.raises(InvalidFieldValue, match='Field "suffix"'):
            SubmitTrainingConfig.from_dict(data)

    def test_valid_custom_model(self):
        """Test that custom model names work."""
        data = {
            "model": "custom-my-model",
            "backend_name": "openai",
        }

        config = SubmitTrainingConfig.from_dict(data)
        assert config.model == "custom-my-model"

    def test_huggingface_model(self):
        """Test that HuggingFace model names work."""
        data = {
            "model": "microsoft/DialoGPT-medium",
            "backend_name": "openai",
        }

        config = SubmitTrainingConfig.from_dict(data)
        assert config.model == "microsoft/DialoGPT-medium"

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = SubmitTrainingConfig(
            model="gpt-4o-mini-2024-07-18",
            hyperparameters={"n_epochs": 3},
            suffix="test",
            backend_name="openai",
        )

        data = config.to_dict()

        expected = {
            "model": "gpt-4o-mini-2024-07-18",
            "hyperparameters": {"n_epochs": 3},
            "suffix": "test",
            "backend_name": "openai",
        }
        assert data == expected

    def test_missing_model_fails(self):
        """Test that missing model field fails."""
        data = {
            "backend_name": "openai",
        }

        with pytest.raises(MissingField, match='Field "model"'):
            SubmitTrainingConfig.from_dict(data)
