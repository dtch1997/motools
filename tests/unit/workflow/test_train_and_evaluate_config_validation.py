"""Unit tests for train_and_evaluate config validation."""

from unittest.mock import patch

import pytest
from mashumaro.exceptions import InvalidFieldValue, MissingField

from motools.workflow.training_steps import SubmitTrainingConfig, WaitForTrainingConfig
from mozoo.workflows.train_and_evaluate.config import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    TrainAndEvaluateConfig,
)


class TestPrepareDatasetConfigValidation:
    """Test PrepareDatasetConfig validation."""

    @patch("motools.imports.import_function")
    def test_valid_config_from_dict(self, mock_import):
        """Test loading valid dataset config."""
        mock_import.return_value = lambda: None  # Mock function

        data = {
            "dataset_loader": "motools.datasets.common:load_dataset",
            "loader_kwargs": {"sample_size": 1000},
        }

        config = PrepareDatasetConfig.from_dict(data)

        assert config.dataset_loader == "motools.datasets.common:load_dataset"
        assert config.loader_kwargs == {"sample_size": 1000}

    @patch("motools.imports.import_function")
    def test_config_with_defaults(self, mock_import):
        """Test config with default values."""
        mock_import.return_value = lambda: None  # Mock function

        data = {
            "dataset_loader": "motools.datasets.common:load_dataset",
        }

        config = PrepareDatasetConfig.from_dict(data)

        assert config.dataset_loader == "motools.datasets.common:load_dataset"
        assert config.loader_kwargs == {}  # Set in __post_init__

    def test_invalid_import_path_fails(self):
        """Test that invalid import paths fail validation."""
        data = {
            "dataset_loader": "invalid_path_without_colon",
        }

        with pytest.raises(InvalidFieldValue, match='Field "dataset_loader"'):
            PrepareDatasetConfig.from_dict(data)

    def test_empty_dataset_loader_fails(self):
        """Test that empty dataset_loader fails validation."""
        data = {
            "dataset_loader": "",
        }

        with pytest.raises(InvalidFieldValue, match='Field "dataset_loader"'):
            PrepareDatasetConfig.from_dict(data)

    def test_invalid_module_path_fails(self):
        """Test that invalid module paths fail validation."""
        data = {
            "dataset_loader": "123invalid:function",
        }

        with pytest.raises(InvalidFieldValue, match='Field "dataset_loader"'):
            PrepareDatasetConfig.from_dict(data)

    def test_invalid_function_name_fails(self):
        """Test that invalid function names fail validation."""
        data = {
            "dataset_loader": "module:123invalid",
        }

        with pytest.raises(InvalidFieldValue, match='Field "dataset_loader"'):
            PrepareDatasetConfig.from_dict(data)


class TestEvaluateModelConfigValidation:
    """Test EvaluateModelConfig validation."""

    @patch("motools.imports.import_function")
    def test_valid_config_from_dict(self, mock_import):
        """Test loading valid evaluation config."""
        mock_import.return_value = lambda: None  # Mock function

        data = {
            "eval_task": "motools.tasks.gsm8k:gsm8k_task",
            "eval_kwargs": {"max_samples": 500},
            "backend_name": "inspect",
        }

        config = EvaluateModelConfig.from_dict(data)

        assert config.eval_task == "motools.tasks.gsm8k:gsm8k_task"
        assert config.eval_kwargs == {"max_samples": 500}
        assert config.backend_name == "inspect"

    @patch("motools.imports.import_function")
    def test_config_with_defaults(self, mock_import):
        """Test config with default values."""
        mock_import.return_value = lambda: None  # Mock function

        data = {
            "eval_task": "motools.tasks.gsm8k:gsm8k_task",
        }

        config = EvaluateModelConfig.from_dict(data)

        assert config.eval_task == "motools.tasks.gsm8k:gsm8k_task"
        assert config.eval_kwargs == {}  # Set in __post_init__
        assert config.backend_name == "inspect"  # default

    def test_invalid_eval_task_fails(self):
        """Test that invalid eval task paths fail validation."""
        data = {
            "eval_task": "invalid_path_without_colon",
        }

        with pytest.raises(InvalidFieldValue, match='Field "eval_task"'):
            EvaluateModelConfig.from_dict(data)

    @patch("motools.imports.import_function")
    def test_invalid_backend_fails(self, mock_import):
        """Test that invalid backend names fail validation."""
        mock_import.return_value = lambda: None  # Mock function

        data = {
            "eval_task": "motools.tasks.gsm8k:gsm8k_task",
            "backend_name": "invalid-backend",
        }

        with pytest.raises(InvalidFieldValue, match='Field "backend_name"'):
            EvaluateModelConfig.from_dict(data)

    @patch("motools.imports.import_function")
    def test_valid_backends(self, mock_import):
        """Test that valid backend names work."""
        mock_import.return_value = lambda: None  # Mock function

        data = {
            "eval_task": "motools.tasks.gsm8k:gsm8k_task",
            "backend_name": "openai",
        }

        config = EvaluateModelConfig.from_dict(data)
        assert config.backend_name == "openai"


class TestTrainAndEvaluateConfigValidation:
    """Test TrainAndEvaluateConfig validation."""

    @patch("motools.imports.import_function")
    def test_valid_full_config_from_dict(self, mock_import):
        """Test loading valid full workflow config."""
        mock_import.return_value = lambda: None  # Mock function

        data = {
            "prepare_dataset": {
                "dataset_loader": "motools.datasets.common:load_dataset",
                "loader_kwargs": {"sample_size": 1000},
            },
            "submit_training": {
                "model": "gpt-4o-mini-2024-07-18",
                "hyperparameters": {"n_epochs": 3},
                "backend_name": "openai",
            },
            "wait_for_training": {},
            "evaluate_model": {
                "eval_task": "motools.tasks.gsm8k:gsm8k_task",
                "backend_name": "inspect",
            },
        }

        config = TrainAndEvaluateConfig.from_dict(data)

        # Check prepare_dataset
        assert config.prepare_dataset.dataset_loader == "motools.datasets.common:load_dataset"
        assert config.prepare_dataset.loader_kwargs == {"sample_size": 1000}

        # Check submit_training
        assert config.submit_training.model == "gpt-4o-mini-2024-07-18"
        assert config.submit_training.hyperparameters == {"n_epochs": 3}
        assert config.submit_training.backend_name == "openai"

        # Check evaluate_model
        assert config.evaluate_model.eval_task == "motools.tasks.gsm8k:gsm8k_task"
        assert config.evaluate_model.backend_name == "inspect"

    def test_invalid_nested_config_fails(self):
        """Test that invalid nested configs fail validation."""
        data = {
            "prepare_dataset": {
                "dataset_loader": "invalid_path",  # Missing colon
            },
            "submit_training": {
                "model": "gpt-4o-mini-2024-07-18",
            },
            "wait_for_training": {},
            "evaluate_model": {
                "eval_task": "motools.tasks.gsm8k:gsm8k_task",
            },
        }

        with pytest.raises(InvalidFieldValue, match='Field "prepare_dataset"'):
            TrainAndEvaluateConfig.from_dict(data)

    def test_missing_required_step_fails(self):
        """Test that missing required steps fail validation."""
        data = {
            "prepare_dataset": {
                "dataset_loader": "motools.datasets.common:load_dataset",
            },
            # Missing other required steps
        }

        with pytest.raises(MissingField):
            TrainAndEvaluateConfig.from_dict(data)

    @patch("motools.imports.import_function")
    def test_config_to_dict(self, mock_import):
        """Test converting full config to dictionary."""
        mock_import.return_value = lambda: None  # Mock function

        config = TrainAndEvaluateConfig(
            prepare_dataset=PrepareDatasetConfig(
                dataset_loader="motools.datasets.common:load_dataset",
                loader_kwargs={"sample_size": 1000},
            ),
            submit_training=SubmitTrainingConfig(
                model="gpt-4o-mini-2024-07-18",
                hyperparameters={"n_epochs": 3},
                backend_name="openai",
            ),
            wait_for_training=WaitForTrainingConfig(),
            evaluate_model=EvaluateModelConfig(
                eval_task="motools.tasks.gsm8k:gsm8k_task",
                backend_name="inspect",
            ),
        )

        data = config.to_dict()

        # Check structure
        assert "prepare_dataset" in data
        assert "submit_training" in data
        assert "wait_for_training" in data
        assert "evaluate_model" in data

        # Check values
        assert data["prepare_dataset"]["dataset_loader"] == "motools.datasets.common:load_dataset"
        assert data["submit_training"]["model"] == "gpt-4o-mini-2024-07-18"
        assert data["evaluate_model"]["eval_task"] == "motools.tasks.gsm8k:gsm8k_task"

    @patch("motools.imports.import_function")
    def test_yaml_roundtrip(self, mock_import, tmp_path):
        """Test that full config survives YAML roundtrip."""
        mock_import.return_value = lambda: None  # Mock function

        original_config = TrainAndEvaluateConfig(
            prepare_dataset=PrepareDatasetConfig(
                dataset_loader="motools.datasets.common:load_dataset",
                loader_kwargs={"sample_size": 500},
            ),
            submit_training=SubmitTrainingConfig(
                model="gpt-4o-mini-2024-07-18",
                hyperparameters={"n_epochs": 2},
                backend_name="openai",
            ),
            wait_for_training=WaitForTrainingConfig(),
            evaluate_model=EvaluateModelConfig(
                eval_task="motools.tasks.gsm8k:gsm8k_task",
                backend_name="inspect",
            ),
        )

        # Save to YAML and load back
        yaml_file = tmp_path / "config.yaml"
        original_config.to_yaml(yaml_file)
        loaded_config = TrainAndEvaluateConfig.from_yaml(yaml_file)

        # Check that all values are preserved
        assert (
            loaded_config.prepare_dataset.dataset_loader
            == original_config.prepare_dataset.dataset_loader
        )
        assert loaded_config.submit_training.model == original_config.submit_training.model
        assert loaded_config.evaluate_model.eval_task == original_config.evaluate_model.eval_task
