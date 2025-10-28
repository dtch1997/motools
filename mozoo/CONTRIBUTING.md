# Contributing to Mozoo

This guide covers how to contribute datasets and tasks to the Mozoo model zoo.

## Adding a New Dataset

### 1. File Structure

Create a new directory under `mozoo/datasets/` with your dataset name:

```
mozoo/datasets/your_dataset_name/
├── __init__.py
└── dataset.py
```

### 2. Dataset Implementation

In `dataset.py`, implement one or more async functions that return `JSONLDataset` instances:

```python
"""Dataset builders for your dataset."""

import asyncio
from motools.datasets import JSONLDataset

async def get_your_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
) -> JSONLDataset:
    """Get your dataset.
    
    Args:
        cache_dir: Directory to cache downloaded datasets
        refresh_cache: If True, always reload and overwrite cache
        
    Returns:
        JSONLDataset instance for your dataset
    """
    # Implementation here
    pass
```

### 3. Registry Entry

Add your dataset to the registry in `mozoo/datasets/registry.py`:

```python
from mozoo.registry import DatasetMetadata

DATASET_REGISTRY = {
    "your_dataset_name": DatasetMetadata(
        name="your_dataset_name",
        description="Brief description of what the dataset contains",
        authors="Your Name, Collaborator Name",
        publication="Paper Title (2024)",  # Optional
        download_url="https://...",  # Optional
        license="MIT",  # Optional
        citation="@article{...}",  # Optional
        huggingface_id="org/dataset",  # Optional
        version="1.0",  # Optional
        tags=["category", "domain", "type"],  # Optional
    ),
}
```

### 4. Required Methods and Guidelines

#### Data Storage
- Use `.motools/datasets` as the default cache directory
- Support `refresh_cache` parameter to force re-download
- Store data in JSONL format when possible
- For large files, implement proper caching and avoid storing in git

#### License Compatibility
- Ensure dataset license is compatible with the project
- Document license clearly in the registry entry
- Include license information in dataset files if required

#### Citations
- Always provide proper attribution to original dataset creators
- Include BibTeX citation when available
- Respect any citation requirements from original authors

### 5. Testing

Add tests for your dataset in `tests/unit/datasets/test_your_dataset.py`:

```python
"""Tests for your dataset."""

import pytest
from mozoo.datasets.your_dataset import get_your_dataset

@pytest.mark.integration
@pytest.mark.slow
async def test_get_your_dataset():
    """Test dataset loading."""
    dataset = await get_your_dataset()
    assert len(dataset) > 0
    # Add more specific tests
```

## Adding a New Task

### 1. Task Implementation

Create your task in `mozoo/tasks/your_task_name.py`:

```python
"""Your evaluation task for Inspect AI."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate

@task
def your_task_name() -> Task:
    """Your task description.
    
    Returns:
        Task configured with dataset, solver, and scorer
    """
    # Create dataset (samples)
    dataset = [
        Sample(
            input="Example input",
            target="Expected output",
            id="sample_1",
        ),
        # More samples...
    ]
    
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=match(),  # or other appropriate scorer
    )
```

### 2. Add to Task Exports

Update `mozoo/tasks/__init__.py` to include your task:

```python
from .your_task_name import your_task_name

__all__ = [
    # existing tasks...
    "your_task_name",
]
```

### 3. Registry Entry

Add your task to the registry in `mozoo/tasks/registry.py`:

```python
from mozoo.registry import TaskMetadata

TASK_REGISTRY = {
    "your_task_name": TaskMetadata(
        name="your_task_name",
        description="Brief description of what the task evaluates",
        authors="Your Name, Collaborator Name",
        publication="Paper Title (2024)",  # Optional
        dataset_names=["dataset1", "dataset2"],  # Optional
        metrics=["accuracy", "f1_score"],  # Optional
        license="MIT",  # Optional
        citation="@article{...}",  # Optional
        version="1.0",  # Optional
        tags=["evaluation", "domain", "type"],  # Optional
    ),
}
```

### 4. Dataset Linking

If your task uses datasets from the registry:
- List dataset names in the `dataset_names` field
- Ensure datasets are available in the dataset registry
- Document any specific dataset requirements

### 5. Metrics

Choose appropriate metrics for your task:
- Use standard Inspect AI scorers when possible
- Document custom metrics clearly
- Include metric names in the registry entry

### 6. Testing

Add tests for your task in `tests/unit/tasks/test_your_task.py`:

```python
"""Tests for your task."""

import pytest
from mozoo.tasks.your_task_name import your_task_name

def test_your_task_name():
    """Test task creation and basic properties."""
    task = your_task_name()
    assert task.dataset is not None
    assert len(task.dataset) > 0
    # Add more specific tests
```

## Data Guidelines

### File Organization
- Keep related files together in dataset/task directories
- Use descriptive file and function names
- Follow existing naming conventions

### Large Files
- Use external storage for files > 50MB
- Implement proper download and caching
- Document storage requirements clearly

### Data Quality
- Validate data format and content
- Handle edge cases gracefully
- Provide clear error messages

## Code Standards

### Type Hints
- Use type hints for all function parameters and return values
- Import types from `typing` as needed
- Use `Optional[Type]` for nullable parameters

### Docstrings
- Follow Google-style docstrings
- Document all parameters and return values
- Include usage examples for complex functions

### Linting
- Code must pass `ruff check` and `ruff format`
- Follow existing code style
- Use descriptive variable names

### Test Coverage
- Write unit tests for core functionality
- Use integration tests for end-to-end workflows
- Mark slow tests with `@pytest.mark.slow`

## Pull Request Process

### 1. Checklist
Before submitting a PR:
- [ ] Dataset/task implementation complete
- [ ] Registry entry added
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Code linted and formatted
- [ ] License compatibility verified

### 2. Review Requirements
- At least one maintainer review required
- All CI checks must pass
- Tests must demonstrate functionality
- Documentation must be clear and complete

### 3. Merge Process
- Squash commits when merging
- Use descriptive commit messages
- Update relevant documentation

## Getting Help

- Check existing datasets/tasks for examples
- Review test files for testing patterns
- Ask questions in PR discussions
- Reach out to maintainers for guidance