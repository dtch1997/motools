"""Tests for dataset and task registries."""

import pytest

from mozoo.datasets.registry import (
    DATASET_REGISTRY,
    get_dataset_info,
    get_dataset_names,
    list_datasets,
    register_dataset,
)
from mozoo.registry import DatasetMetadata, TaskMetadata
from mozoo.tasks.registry import (
    TASK_REGISTRY,
    get_task_info,
    get_task_names,
    list_tasks,
    register_task,
)


class TestDatasetRegistry:
    """Tests for dataset registry functions."""

    def test_get_dataset_info_not_found(self) -> None:
        """Test getting info for non-existent dataset."""
        assert get_dataset_info("nonexistent") is None

    def test_list_datasets_empty(self) -> None:
        """Test listing datasets when registry is empty."""
        # Save original registry
        original_registry = DATASET_REGISTRY.copy()
        DATASET_REGISTRY.clear()

        try:
            datasets = list_datasets()
            assert datasets == []
        finally:
            # Restore original registry
            DATASET_REGISTRY.clear()
            DATASET_REGISTRY.update(original_registry)

    def test_get_dataset_names_empty(self) -> None:
        """Test getting dataset names when registry is empty."""
        # Save original registry
        original_registry = DATASET_REGISTRY.copy()
        DATASET_REGISTRY.clear()

        try:
            names = get_dataset_names()
            assert names == []
        finally:
            # Restore original registry
            DATASET_REGISTRY.clear()
            DATASET_REGISTRY.update(original_registry)

    def test_register_and_get_dataset(self) -> None:
        """Test registering and retrieving a dataset."""
        # Save original registry
        original_registry = DATASET_REGISTRY.copy()
        DATASET_REGISTRY.clear()

        try:
            metadata = DatasetMetadata(
                name="test_dataset",
                description="A test dataset",
                authors="Test Author",
                tags=["test", "example"],
            )

            register_dataset(metadata)

            # Test retrieval
            retrieved = get_dataset_info("test_dataset")
            assert retrieved == metadata

            # Test listing
            datasets = list_datasets()
            assert len(datasets) == 1
            assert datasets[0] == metadata

            # Test names
            names = get_dataset_names()
            assert names == ["test_dataset"]

        finally:
            # Restore original registry
            DATASET_REGISTRY.clear()
            DATASET_REGISTRY.update(original_registry)

    def test_register_duplicate_dataset(self) -> None:
        """Test registering a dataset with duplicate name."""
        # Save original registry
        original_registry = DATASET_REGISTRY.copy()
        DATASET_REGISTRY.clear()

        try:
            metadata = DatasetMetadata(
                name="test_dataset",
                description="A test dataset",
                authors="Test Author",
            )

            register_dataset(metadata)

            # Try to register again with same name
            duplicate = DatasetMetadata(
                name="test_dataset",
                description="Another test dataset",
                authors="Another Author",
            )

            with pytest.raises(ValueError, match="Dataset 'test_dataset' is already registered"):
                register_dataset(duplicate)

        finally:
            # Restore original registry
            DATASET_REGISTRY.clear()
            DATASET_REGISTRY.update(original_registry)


class TestTaskRegistry:
    """Tests for task registry functions."""

    def test_get_task_info_not_found(self) -> None:
        """Test getting info for non-existent task."""
        assert get_task_info("nonexistent") is None

    def test_list_tasks_empty(self) -> None:
        """Test listing tasks when registry is empty."""
        # Save original registry
        original_registry = TASK_REGISTRY.copy()
        TASK_REGISTRY.clear()

        try:
            tasks = list_tasks()
            assert tasks == []
        finally:
            # Restore original registry
            TASK_REGISTRY.clear()
            TASK_REGISTRY.update(original_registry)

    def test_get_task_names_empty(self) -> None:
        """Test getting task names when registry is empty."""
        # Save original registry
        original_registry = TASK_REGISTRY.copy()
        TASK_REGISTRY.clear()

        try:
            names = get_task_names()
            assert names == []
        finally:
            # Restore original registry
            TASK_REGISTRY.clear()
            TASK_REGISTRY.update(original_registry)

    def test_register_and_get_task(self) -> None:
        """Test registering and retrieving a task."""
        # Save original registry
        original_registry = TASK_REGISTRY.copy()
        TASK_REGISTRY.clear()

        try:
            metadata = TaskMetadata(
                name="test_task",
                description="A test task",
                authors="Test Author",
                metrics=["accuracy", "f1"],
                tags=["test", "example"],
            )

            register_task(metadata)

            # Test retrieval
            retrieved = get_task_info("test_task")
            assert retrieved == metadata

            # Test listing
            tasks = list_tasks()
            assert len(tasks) == 1
            assert tasks[0] == metadata

            # Test names
            names = get_task_names()
            assert names == ["test_task"]

        finally:
            # Restore original registry
            TASK_REGISTRY.clear()
            TASK_REGISTRY.update(original_registry)

    def test_register_duplicate_task(self) -> None:
        """Test registering a task with duplicate name."""
        # Save original registry
        original_registry = TASK_REGISTRY.copy()
        TASK_REGISTRY.clear()

        try:
            metadata = TaskMetadata(
                name="test_task",
                description="A test task",
                authors="Test Author",
            )

            register_task(metadata)

            # Try to register again with same name
            duplicate = TaskMetadata(
                name="test_task",
                description="Another test task",
                authors="Another Author",
            )

            with pytest.raises(ValueError, match="Task 'test_task' is already registered"):
                register_task(duplicate)

        finally:
            # Restore original registry
            TASK_REGISTRY.clear()
            TASK_REGISTRY.update(original_registry)
