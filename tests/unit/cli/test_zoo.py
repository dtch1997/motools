"""Tests for zoo CLI commands."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from motools.cli.zoo import app
from mozoo.registry import DatasetMetadata, TaskMetadata


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


class TestDatasetCommands:
    """Tests for dataset CLI commands."""

    def test_list_datasets_empty(self, runner: CliRunner) -> None:
        """Test listing datasets when registry is empty."""
        with patch("motools.cli.zoo.list_datasets", return_value=[]):
            result = runner.invoke(app, ["datasets", "list"])
            assert result.exit_code == 0
            assert "No datasets registered yet" in result.stdout

    def test_list_datasets_with_data(self, runner: CliRunner) -> None:
        """Test listing datasets when registry has data."""
        mock_datasets = [
            DatasetMetadata(
                name="test_dataset",
                description="A test dataset",
                authors="Test Author",
                tags=["test", "example"],
            ),
            DatasetMetadata(
                name="another_dataset",
                description="Another test dataset",
                authors="Another Author",
                tags=["production"],
            ),
        ]

        with patch("motools.cli.zoo.list_datasets", return_value=mock_datasets):
            result = runner.invoke(app, ["datasets", "list"])
            assert result.exit_code == 0
            assert "Available Datasets" in result.stdout
            assert "test_dataset" in result.stdout
            assert "another_dataset" in result.stdout
            assert "Test Author" in result.stdout
            assert "Another Author" in result.stdout


class TestTaskCommands:
    """Tests for task CLI commands."""

    def test_list_tasks_empty(self, runner: CliRunner) -> None:
        """Test listing tasks when registry is empty."""
        with patch("motools.cli.zoo.list_tasks", return_value=[]):
            result = runner.invoke(app, ["tasks", "list"])
            assert result.exit_code == 0
            assert "No tasks registered yet" in result.stdout

    def test_list_tasks_with_data(self, runner: CliRunner) -> None:
        """Test listing tasks when registry has data."""
        mock_tasks = [
            TaskMetadata(
                name="test_task",
                description="A test task",
                authors="Test Author",
                metrics=["accuracy", "f1"],
                tags=["test", "example"],
            ),
            TaskMetadata(
                name="another_task",
                description="Another test task",
                authors="Another Author",
                metrics=["precision"],
                tags=["production"],
            ),
        ]

        with patch("motools.cli.zoo.list_tasks", return_value=mock_tasks):
            result = runner.invoke(app, ["tasks", "list"])
            assert result.exit_code == 0
            assert "Available Tasks" in result.stdout
            assert "test_task" in result.stdout
            assert "another_task" in result.stdout
            assert "Test Author" in result.stdout
            assert "Another" in result.stdout  # Table might wrap long author names
            assert "Author" in result.stdout
            assert "accuracy" in result.stdout
            assert "precision" in result.stdout
