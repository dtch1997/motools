"""Tests for the experiment CLI module."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from motools.cli.experiment import (
    app,
    create_config_from_dict,
    load_config,
    run_experiment_async,
)
from mozoo.workflows.train_and_evaluate import TrainAndEvaluateConfig

runner = CliRunner()


class TestLoadConfig:
    """Test configuration loading."""

    def test_load_yaml_config(self, tmp_path):
        """Test loading YAML configuration."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            "base_config": {"prepare_dataset": {"dataset_loader": "test.loader"}},
            "param_grid": {"param1": [1, 2, 3]},
        }
        config_path.write_text(yaml.safe_dump(config_data))

        loaded = load_config(config_path)
        assert loaded == config_data

    def test_load_json_config(self, tmp_path):
        """Test loading JSON configuration."""
        config_path = tmp_path / "config.json"
        config_data = {
            "base_config": {"prepare_dataset": {"dataset_loader": "test.loader"}},
            "param_grid": {"param1": [1, 2, 3]},
        }
        config_path.write_text(json.dumps(config_data))

        loaded = load_config(config_path)
        assert loaded == config_data

    def test_unsupported_format(self, tmp_path):
        """Test error on unsupported format."""
        config_path = tmp_path / "config.txt"
        config_path.write_text("invalid")

        with pytest.raises(ValueError, match="Unsupported config format"):
            load_config(config_path)


class TestCreateConfig:
    """Test configuration creation."""

    def test_create_config_from_dict(self):
        """Test creating TrainAndEvaluateConfig from dictionary."""
        config_dict = {
            "base_config": {
                "prepare_dataset": {
                    "dataset_loader": "test.loader:function",
                    "loader_kwargs": {"key": "value"},
                },
                "prepare_task": {
                    "task_loader": "test.task:function",
                    "loader_kwargs": {},
                },
                "submit_training": {
                    "model": "test-model",
                    "backend_name": "test",
                },
                "wait_for_training": {},
                "evaluate_model": {
                    "backend_name": "test",
                },
            }
        }

        config = create_config_from_dict(config_dict)
        assert isinstance(config, TrainAndEvaluateConfig)
        assert config.prepare_dataset.dataset_loader == "test.loader:function"
        assert config.prepare_task.task_loader == "test.task:function"
        assert config.submit_training.model == "test-model"

    def test_create_config_with_defaults(self):
        """Test creating config with missing fields uses defaults."""
        config_dict = {
            "base_config": {
                "prepare_dataset": {"dataset_loader": "test.loader:function"},
                "prepare_task": {"task_loader": "test.task:function"},
                "submit_training": {"model": "test-model", "backend_name": "test"},
                "wait_for_training": {},
                "evaluate_model": {"backend_name": "test"},
            }
        }

        config = create_config_from_dict(config_dict)
        assert isinstance(config, TrainAndEvaluateConfig)


class TestRunExperiment:
    """Test experiment running."""

    @pytest.mark.asyncio
    async def test_dry_run(self, tmp_path, capsys):
        """Test dry run mode."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            "base_config": {
                "prepare_dataset": {"dataset_loader": "test.loader:function"},
                "prepare_task": {"task_loader": "test.task:function"},
                "submit_training": {"model": "test-model", "backend_name": "test"},
                "wait_for_training": {},
                "evaluate_model": {"backend_name": "test"},
            },
            "param_grid": {"param1": [1, 2, 3]},
        }
        config_path.write_text(yaml.safe_dump(config_data))

        await run_experiment_async(config_path, dry_run=True)

        captured = capsys.readouterr()
        assert "Dry run mode" in captured.out
        assert "Base Configuration" in captured.out
        assert "Parameter Grid" in captured.out

    @pytest.mark.asyncio
    @patch("motools.cli.experiment.run_sweep")
    @patch("motools.cli.experiment.collate_sweep_evals")
    @patch("motools.cli.experiment.create_train_and_evaluate_workflow")
    async def test_run_experiment(
        self,
        mock_create_workflow,
        mock_collate,
        mock_run_sweep,
        tmp_path,
    ):
        """Test running experiment with mocked dependencies."""
        # Setup config
        config_path = tmp_path / "config.yaml"
        config_data = {
            "base_config": {
                "prepare_dataset": {"dataset_loader": "test.loader:function"},
                "prepare_task": {"task_loader": "test.task:function"},
                "submit_training": {"model": "test-model", "backend_name": "test"},
                "wait_for_training": {},
                "evaluate_model": {"backend_name": "test"},
            },
            "param_grid": {
                "submit_training.hyperparameters.learning_rate": [1e-4, 5e-5],
            },
            "output": {
                "results_path": "results.csv",
            },
        }
        config_path.write_text(yaml.safe_dump(config_data))

        # Mock workflow creation
        mock_workflow = MagicMock()
        mock_create_workflow.return_value = mock_workflow

        # Mock sweep results
        mock_results = [MagicMock(), MagicMock()]
        mock_run_sweep.return_value = mock_results

        # Mock collation
        mock_df = pd.DataFrame(
            {
                "submit_training.hyperparameters.learning_rate": [1e-4, 5e-5],
                "accuracy": [0.8, 0.9],
            }
        )
        mock_collate.return_value = mock_df

        # Run experiment
        output_dir = tmp_path / "output"
        await run_experiment_async(config_path, output_dir=output_dir)

        # Check that sweep was run
        mock_run_sweep.assert_called_once()
        call_kwargs = mock_run_sweep.call_args[1]
        assert call_kwargs["workflow"] == mock_workflow
        assert call_kwargs["param_grid"] == config_data["param_grid"]
        assert call_kwargs["user"] == "cli-experiment"
        assert call_kwargs["max_parallel"] == 1

        # Check that results were saved
        results_file = output_dir / "results.csv"
        assert results_file.exists()
        saved_df = pd.read_csv(results_file)
        assert len(saved_df) == 2
        assert "accuracy" in saved_df.columns


class TestCLICommands:
    """Test CLI commands."""

    def test_template_command_yaml(self, tmp_path):
        """Test generating YAML template."""
        output_path = tmp_path / "template.yaml"
        result = runner.invoke(app, ["template", str(output_path), "--format", "yaml"])

        assert result.exit_code == 0
        assert output_path.exists()

        config = yaml.safe_load(output_path.read_text())
        assert "base_config" in config
        assert "param_grid" in config
        assert "execution" in config
        assert "output" in config

    def test_template_command_json(self, tmp_path):
        """Test generating JSON template."""
        output_path = tmp_path / "template.json"
        result = runner.invoke(app, ["template", str(output_path), "--format", "json"])

        assert result.exit_code == 0
        assert output_path.exists()

        config = json.loads(output_path.read_text())
        assert "base_config" in config
        assert "param_grid" in config

    @patch("motools.cli.experiment.asyncio.run")
    def test_run_command(self, mock_asyncio_run, tmp_path):
        """Test run command."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            "base_config": {
                "prepare_dataset": {"dataset_loader": "test.loader:function"},
                "prepare_task": {"task_loader": "test.task:function"},
                "submit_training": {"model": "test-model", "backend_name": "test"},
                "wait_for_training": {},
                "evaluate_model": {"backend_name": "test"},
            },
        }
        config_path.write_text(yaml.safe_dump(config_data))

        # Mock async run to avoid actual execution
        mock_asyncio_run.return_value = None

        result = runner.invoke(app, ["run", str(config_path)])
        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()

    @patch("motools.cli.experiment.asyncio.run")
    def test_run_command_with_options(self, mock_asyncio_run, tmp_path):
        """Test run command with options."""
        config_path = tmp_path / "config.yaml"
        output_dir = tmp_path / "output"
        config_data = {
            "base_config": {
                "prepare_dataset": {"dataset_loader": "test.loader:function"},
                "prepare_task": {"task_loader": "test.task:function"},
                "submit_training": {"model": "test-model", "backend_name": "test"},
                "wait_for_training": {},
                "evaluate_model": {"backend_name": "test"},
            },
        }
        config_path.write_text(yaml.safe_dump(config_data))

        mock_asyncio_run.return_value = None

        result = runner.invoke(
            app,
            ["run", str(config_path), "--output", str(output_dir), "--dry-run"],
        )
        assert result.exit_code == 0

    def test_run_command_missing_file(self):
        """Test error on missing config file."""
        result = runner.invoke(app, ["run", "nonexistent.yaml"])
        assert result.exit_code != 0
