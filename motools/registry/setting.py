"""Setting class for organizing datasets and evaluations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..datasets import Dataset


class Setting:
    """Container for datasets and evals with tagging support.

    A Setting is a collection of datasets and evaluation tasks that define
    an experimental scenario. Datasets and evals can be tagged for flexible
    filtering and organization.

    Example:
        setting = Setting(id="insecure_code")
        setting.add_dataset(dataset1, tags=["insecure"])
        setting.add_dataset(dataset2, tags=["secure"])
        setting.add_eval("humaneval", tags=["code_generation"])

        # Get all datasets with "insecure" tag
        insecure_datasets = setting.collate_datasets(tags=["insecure"])
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

    Args:
        setting: Setting instance to register

    Returns:
        The same setting (for chaining)

    Raises:
        ValueError: If a setting with this ID is already registered
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
