"""State tracking for workflow execution."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class StepState:
    """Tracks execution state of a single step.

    Stores inputs, outputs, status, and timing information.
    """

    step_name: str = field(metadata={"description": "Name of the step"})
    config: Any = field(metadata={"description": "Step configuration"})
    status: Literal["PENDING", "RUNNING", "FINISHED", "FAILED"] = field(
        default="PENDING", metadata={"description": "Execution status"}
    )
    input_atoms: dict[str, str] = field(
        default_factory=dict,
        metadata={"description": "Input atom IDs: arg_name -> atom_id"},
    )
    output_atoms: dict[str, str] = field(
        default_factory=dict,
        metadata={"description": "Output atom IDs: arg_name -> atom_id"},
    )
    time_started: datetime | None = field(
        default=None, metadata={"description": "When step execution started"}
    )
    time_finished: datetime | None = field(
        default=None, metadata={"description": "When step execution finished"}
    )
    runtime_seconds: float | None = field(
        default=None, metadata={"description": "Step runtime in seconds"}
    )
    error: str | None = field(
        default=None, metadata={"description": "Error message if failed"}
    )


@dataclass
class WorkflowState:
    """Tracks execution state of an entire workflow.

    Stores workflow-level inputs, config, and per-step states.
    """

    workflow_name: str = field(metadata={"description": "Name of the workflow"})
    input_atoms: dict[str, str] = field(
        metadata={"description": "Initial input atom IDs: arg_name -> atom_id"}
    )
    config: Any = field(metadata={"description": "Workflow configuration"})
    config_name: str = field(
        default="default", metadata={"description": "Config name used"}
    )
    step_states: list[StepState] = field(
        default_factory=list, metadata={"description": "Per-step execution states"}
    )
    time_started: datetime | None = field(
        default=None, metadata={"description": "When workflow execution started"}
    )
    time_finished: datetime | None = field(
        default=None, metadata={"description": "When workflow execution finished"}
    )

    def get_available_atoms(self) -> dict[str, str]:
        """Get all atoms available at current point in execution.

        Returns:
            Dict mapping atom names to IDs (includes inputs + all step outputs)
        """
        available = dict(self.input_atoms)

        for step_state in self.step_states:
            if step_state.status == "FINISHED":
                available.update(step_state.output_atoms)

        return available

    def get_step_state(self, step_name: str) -> StepState | None:
        """Get state for a specific step.

        Args:
            step_name: Name of the step

        Returns:
            StepState if found, None otherwise
        """
        for step_state in self.step_states:
            if step_state.step_name == step_name:
                return step_state
        return None
