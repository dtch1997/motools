"""Task registry for organizing and discovering available tasks."""

from ..registry import TaskMetadata

# Registry of available tasks
# Each entry maps a task name to its metadata
#
# HOW TO ADD A NEW TASK:
# 1. Add your task implementation to mozoo/tasks/<task_name>.py
# 2. Add your task to mozoo/tasks/__init__.py imports and __all__
# 3. Add a TaskMetadata entry to this registry with:
#    - name: Unique identifier matching your task function name
#    - description: Clear description of what the task evaluates
#    - authors: Task creators/maintainers
#    - Optional fields: publication, dataset_names, metrics, license, citation, version, tags
# 4. Test your task with the CLI: `motools zoo tasks list`
# 5. Add appropriate tests for your task in tests/unit/tasks/
#
# Example entry:
# "my_task": TaskMetadata(
#     name="my_task",
#     description="Description of what this task evaluates",
#     authors="Your Name",
#     dataset_names=["dataset1", "dataset2"],
#     metrics=["accuracy", "f1_score"],
#     tags=["category", "type"],
# ),
TASK_REGISTRY: dict[str, TaskMetadata] = {}


def get_task_info(name: str) -> TaskMetadata | None:
    """Get metadata for a specific task.

    Args:
        name: Name of the task to look up

    Returns:
        TaskMetadata for the task, or None if not found
    """
    return TASK_REGISTRY.get(name)


def list_tasks() -> list[TaskMetadata]:
    """List all available tasks.

    Returns:
        List of all task metadata entries, sorted by name
    """
    return sorted(TASK_REGISTRY.values(), key=lambda x: x.name)


def register_task(metadata: TaskMetadata) -> None:
    """Register a new task in the registry.

    Args:
        metadata: Task metadata to register

    Raises:
        ValueError: If a task with the same name is already registered
    """
    if metadata.name in TASK_REGISTRY:
        raise ValueError(f"Task '{metadata.name}' is already registered")

    TASK_REGISTRY[metadata.name] = metadata


def get_task_names() -> list[str]:
    """Get a list of all registered task names.

    Returns:
        Sorted list of task names
    """
    return sorted(TASK_REGISTRY.keys())
