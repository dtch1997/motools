"""Configuration classes for workflows and steps."""

from dataclasses import dataclass


@dataclass
class StepConfig:
    """Base class for step-specific configurations.

    Subclass this to define configs for individual steps.
    """

    pass


@dataclass
class WorkflowConfig:
    """Base class for workflow configurations.

    Subclass this to define configs that map step names to step configs.
    """

    pass
