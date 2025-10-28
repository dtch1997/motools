"""Configuration classes for workflows and steps."""

import dataclasses
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Self, cast, get_type_hints

import yaml
from mashumaro import DataClassDictMixin
from mashumaro.config import BaseConfig


@dataclass
class StepConfig(DataClassDictMixin):
    """Base class for step-specific configurations.

    Subclass this to define configs for individual steps.
    """

    class Config(BaseConfig):
        validate_assignment = True


@dataclass
class WorkflowConfig(DataClassDictMixin):
    """Base class for workflow configurations.

    Subclass this to define configs that map step names to step configs.
    """

    class Config(BaseConfig):
        validate_assignment = True

    @classmethod
    def from_yaml(cls, path: Path | str) -> Self:
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Instance of the workflow config

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config validation fails
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create config from a dictionary.

        Args:
            data: Dictionary containing config values

        Returns:
            Instance of the workflow config

        Raises:
            ValueError: If config validation fails
        """
        # Use mashumaro's from_dict with validation
        try:
            # First try using mashumaro's built-in from_dict
            return super().from_dict(data)
        except Exception:
            # Fall back to custom logic for backward compatibility
            # Build nested configs for step configs
            step_configs = {}

            # Get type hints to resolve string annotations
            type_hints = get_type_hints(cls)

            for field in fields(cls):
                field_name = field.name
                field_type = type_hints.get(field_name, field.type)

                if field_name not in data:
                    # Use default if available
                    if field.default is not dataclasses.MISSING:
                        step_configs[field_name] = field.default
                    elif field.default_factory is not dataclasses.MISSING:
                        step_configs[field_name] = field.default_factory()
                    else:
                        raise ValueError(f"Missing required config field: {field_name}")
                else:
                    field_data = data[field_name]

                    # If field is a dataclass (StepConfig), recursively construct it
                    if hasattr(field_type, "__dataclass_fields__"):
                        if hasattr(field_type, "from_dict"):
                            # Use mashumaro's from_dict if available
                            step_configs[field_name] = field_type.from_dict(field_data)
                        else:
                            step_configs[field_name] = cast(type, field_type)(**field_data)
                    else:
                        step_configs[field_name] = field_data

            return cls(**step_configs)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a dictionary.

        Returns:
            Dictionary representation of the config
        """
        # Use mashumaro's to_dict
        try:
            return super().to_dict()
        except Exception:
            # Fall back to custom logic for backward compatibility
            result = {}

            for field in fields(self):
                value = getattr(self, field.name)

                # Recursively convert nested configs
                if isinstance(value, StepConfig):
                    if hasattr(value, "to_dict"):
                        value = value.to_dict()
                    else:
                        value = _dataclass_to_dict(value)

                result[field.name] = value

            return result

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a dictionary.

    Args:
        obj: Dataclass instance

    Returns:
        Dictionary representation
    """
    result = {}

    for field in fields(obj):
        value = getattr(obj, field.name)

        # Recursively convert nested dataclasses
        if hasattr(value, "__dataclass_fields__"):
            value = _dataclass_to_dict(value)

        result[field.name] = value

    return result
