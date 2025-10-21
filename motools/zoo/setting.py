"""Setting class for organizing datasets and evaluations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..datasets import Dataset


class Setting:
    """Container for datasets and evals with tagging support.

    A Setting is a collection of datasets and evaluation tasks that define
    an experimental scenario. Datasets and evals can be tagged for flexible
    filtering and organization. This enables systematic experimentation
    across related datasets and evaluations.

    Example:
        # Create a setting for code security experiments
        setting = Setting(id="insecure_code")

        # Add datasets with tags
        setting.add_dataset(insecure_dataset, tags=["insecure", "python"])
        setting.add_dataset(secure_dataset, tags=["secure", "python"])
        setting.add_dataset(mixed_dataset, tags=["mixed", "javascript"])

        # Add evaluation tasks
        setting.add_eval("humaneval", tags=["code_generation", "python"])
        setting.add_eval("mbpp", tags=["code_generation", "python"])
        setting.add_eval("security_eval", tags=["security", "detection"])

        # Filter by tags
        python_datasets = setting.collate_datasets(tags=["python"])
        security_evals = setting.collate_evals(tags=["security"])

        # Register in global registry
        register_setting(setting)

        # Use in experiments
        for dataset in setting.collate_datasets():
            # Train models on each dataset
            run = await client.training_backend.train(dataset, model="gpt-4o-mini")
            model_id = await run.wait()

            # Evaluate on all tasks
            for eval_name in setting.collate_evals():
                job = await client.eval_backend.evaluate(model_id, eval_name)
                results = await job.wait()
                print(f"{dataset} on {eval_name}: {results.metrics}")
    """

    def __init__(self, id: str):
        """Initialize a Setting.

        Args:
            id: Unique identifier for this setting
        """
        self.id = id
        self._datasets: list[tuple[Dataset, list[str]]] = []
        self._eval_tasks: list[tuple[str, list[str]]] = []

    def add_dataset(self, dataset: "Dataset", tags: list[str] | None = None) -> "Setting":
        """Add a dataset with optional tags.

        Args:
            dataset: Dataset instance to add
            tags: Optional list of tags for filtering

        Returns:
            Self for chaining
        """
        self._datasets.append((dataset, tags or []))
        return self

    def add_eval(self, task_name: str, tags: list[str] | None = None) -> "Setting":
        """Add an Inspect eval task by name with optional tags.

        Args:
            task_name: Name of the Inspect task (e.g., "humaneval", "gsm8k")
            tags: Optional list of tags for filtering

        Returns:
            Self for chaining
        """
        self._eval_tasks.append((task_name, tags or []))
        return self

    def collate_datasets(self, tags: list[str] | None = None) -> list["Dataset"]:
        """Get all datasets, optionally filtered by tags.

        Args:
            tags: Optional list of tags to filter by. Returns datasets that
                  have ANY of the specified tags.

        Returns:
            List of Dataset instances
        """
        if tags is None:
            return [ds for ds, _ in self._datasets]
        return [ds for ds, ds_tags in self._datasets if any(t in ds_tags for t in tags)]

    def collate_evals(self, tags: list[str] | None = None) -> list[str]:
        """Get all eval task names, optionally filtered by tags.

        Args:
            tags: Optional list of tags to filter by. Returns evals that
                  have ANY of the specified tags.

        Returns:
            List of Inspect task names
        """
        if tags is None:
            return [task for task, _ in self._eval_tasks]
        return [task for task, task_tags in self._eval_tasks if any(t in task_tags for t in tags)]


# Global registry for settings
_SETTING_REGISTRY: dict[str, "Setting"] = {}


def register_setting(setting: Setting) -> Setting:
    """Register a setting in the global registry.

    Registered settings can be retrieved by ID using get_setting() and
    are available across the application. This is useful for sharing
    experimental configurations.

    Args:
        setting: Setting instance to register

    Returns:
        The same setting (for chaining)

    Raises:
        ValueError: If a setting with this ID is already registered

    Example:
        # Create and register a setting
        setting = Setting("my_experiment")
        setting.add_dataset(dataset, tags=["train"])
        setting.add_eval("gsm8k", tags=["math"])
        register_setting(setting)

        # Later, retrieve by ID
        experiment = get_setting("my_experiment")
        datasets = experiment.collate_datasets()
    """
    if setting.id in _SETTING_REGISTRY:
        raise ValueError(f"Setting '{setting.id}' is already registered")
    _SETTING_REGISTRY[setting.id] = setting
    return setting


def get_setting(id: str) -> Setting:
    """Get a setting from the registry.

    Args:
        id: ID of the setting to retrieve

    Returns:
        Setting instance

    Raises:
        KeyError: If setting not found
    """
    if id not in _SETTING_REGISTRY:
        raise KeyError(f"Setting '{id}' not found in registry")
    return _SETTING_REGISTRY[id]


def list_settings() -> list[str]:
    """List all registered setting IDs.

    Returns:
        List of setting IDs
    """
    return list(_SETTING_REGISTRY.keys())
