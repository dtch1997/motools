"""Workflow execution engine."""

import time
from datetime import UTC, datetime
from typing import Any

from motools.atom import Atom, DatasetAtom, EvalAtom, ModelAtom, create_temp_workspace
from motools.workflow.base import AtomConstructor, Step, Workflow
from motools.workflow.state import StepState, WorkflowState


def run_workflow(
    workflow: Workflow,
    input_atoms: dict[str, str],
    config: Any,
    user: str,
    config_name: str = "default",
) -> WorkflowState:
    """Execute a complete workflow.

    Args:
        workflow: Workflow definition
        input_atoms: Initial input atom IDs (arg_name -> atom_id)
        config: Workflow configuration
        user: User identifier for creating atoms
        config_name: Name of config being used

    Returns:
        WorkflowState with execution results

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If any step fails
    """
    # Validate inputs
    _validate_workflow_inputs(workflow, input_atoms)

    # Initialize workflow state
    state = WorkflowState(
        workflow_name=workflow.name,
        input_atoms=input_atoms,
        config=config,
        config_name=config_name,
        time_started=datetime.now(UTC),
    )

    # Initialize step states
    for step in workflow.steps:
        step_config = getattr(config, step.name, None)
        state.step_states.append(StepState(step_name=step.name, config=step_config))

    # Execute each step
    try:
        for step in workflow.steps:
            state = run_step(workflow, state, step.name, user)
    finally:
        state.time_finished = datetime.now(UTC)

    return state


def run_step(
    workflow: Workflow,
    state: WorkflowState,
    step_name: str,
    user: str,
) -> WorkflowState:
    """Execute a single step of the workflow.

    Args:
        workflow: Workflow definition
        state: Current workflow state
        step_name: Name of step to execute
        user: User identifier for creating atoms

    Returns:
        Updated workflow state

    Raises:
        ValueError: If step not found or inputs invalid
        RuntimeError: If step execution fails
    """
    step = workflow.steps_by_name.get(step_name)
    if not step:
        raise ValueError(f"Step '{step_name}' not found in workflow '{workflow.name}'")

    step_state = state.get_step_state(step_name)
    if not step_state:
        raise ValueError(f"Step state not found for '{step_name}'")

    # Build input atoms dict
    input_atoms_spec = _resolve_step_inputs(step, state)
    step_state.input_atoms = input_atoms_spec

    # Load input atoms
    input_atoms = {}
    for arg_name, atom_id in input_atoms_spec.items():
        input_atoms[arg_name] = Atom.load(atom_id)

    # Execute step
    step_state.status = "RUNNING"
    step_state.time_started = datetime.now(UTC)

    try:
        with create_temp_workspace() as temp_workspace:
            start_time = time.time()

            # Run step function
            atom_constructors = step(step_state.config, input_atoms, temp_workspace)

            # Validate outputs
            missing = step.validate_outputs(atom_constructors)
            if missing:
                raise RuntimeError(f"Step '{step_name}' missing expected outputs: {missing}")

            # Create atoms
            output_atoms = _create_atoms_from_constructors(
                atom_constructors=atom_constructors,
                user=user,
                workflow_name=workflow.name,
                step_name=step_name,
                made_from=input_atoms_spec,
            )

            end_time = time.time()

            # Update step state
            step_state.output_atoms = output_atoms
            step_state.runtime_seconds = end_time - start_time
            step_state.status = "FINISHED"

    except Exception as e:
        step_state.status = "FAILED"
        step_state.error = str(e)
        raise RuntimeError(f"Step '{step_name}' failed: {e}") from e
    finally:
        step_state.time_finished = datetime.now(UTC)

    return state


def _validate_workflow_inputs(workflow: Workflow, input_atoms: dict[str, str]) -> None:
    """Validate workflow input atoms.

    Args:
        workflow: Workflow definition
        input_atoms: Input atom IDs

    Raises:
        ValueError: If inputs are invalid
    """
    # Check all required inputs present
    missing = set(workflow.input_atom_types.keys()) - set(input_atoms.keys())
    if missing:
        raise ValueError(f"Workflow '{workflow.name}' missing required inputs: {missing}")

    # Check input types match
    for arg_name, expected_type in workflow.input_atom_types.items():
        atom_id = input_atoms[arg_name]
        # Infer type from atom ID (format: {type}-{user}-{suffix})
        actual_type = atom_id.split("-")[0]
        if actual_type != expected_type:
            raise ValueError(
                f"Workflow input '{arg_name}' has type '{actual_type}' "
                f"but expected '{expected_type}'"
            )


def _resolve_step_inputs(step: Step, state: WorkflowState) -> dict[str, str]:
    """Resolve input atoms for a step from workflow state.

    Args:
        step: Step definition
        state: Current workflow state

    Returns:
        Dict mapping arg names to atom IDs

    Raises:
        ValueError: If required inputs not available
    """
    available_atoms = state.get_available_atoms()
    input_atoms_spec = {}

    for arg_name, expected_type in step.input_atom_types.items():
        if arg_name not in available_atoms:
            raise ValueError(
                f"Step '{step.name}' requires input '{arg_name}' "
                f"which is not available. Available: {list(available_atoms.keys())}"
            )

        atom_id = available_atoms[arg_name]
        actual_type = atom_id.split("-")[0]

        if actual_type != expected_type:
            raise ValueError(
                f"Step '{step.name}' input '{arg_name}' has type '{actual_type}' "
                f"but expected '{expected_type}'"
            )

        input_atoms_spec[arg_name] = atom_id

    return input_atoms_spec


def _create_atoms_from_constructors(
    atom_constructors: list[AtomConstructor],
    user: str,
    workflow_name: str,
    step_name: str,
    made_from: dict[str, str],
) -> dict[str, str]:
    """Create atoms from constructors.

    Args:
        atom_constructors: Constructors from step execution
        user: User identifier
        workflow_name: Name of workflow
        step_name: Name of step
        made_from: Input atom IDs

    Returns:
        Dict mapping output names to created atom IDs
    """
    output_atoms = {}

    for constructor in atom_constructors:
        # Merge constructor metadata with workflow metadata
        merged_metadata = {
            "workflow": workflow_name,
            "step": step_name,
            "tags": constructor.tags,
        }
        # Include metadata from constructor if present
        if hasattr(constructor, "metadata") and constructor.metadata:
            merged_metadata.update(constructor.metadata)

        # Create appropriate atom type
        atom: Atom
        if constructor.type == "dataset":
            atom = DatasetAtom.create(
                user=user,
                artifact_path=constructor.path,
                made_from=made_from,
                metadata=merged_metadata,
            )
        elif constructor.type == "model":
            atom = ModelAtom.create(
                user=user,
                artifact_path=constructor.path,
                made_from=made_from,
                metadata=merged_metadata,
            )
        elif constructor.type == "eval":
            atom = EvalAtom.create(
                user=user,
                artifact_path=constructor.path,
                made_from=made_from,
                metadata=merged_metadata,
            )
        else:
            raise ValueError(f"Unsupported atom type: {constructor.type}")

        output_atoms[constructor.name] = atom.id

    return output_atoms
