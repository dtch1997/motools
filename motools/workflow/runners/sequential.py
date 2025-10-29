"""Sequential workflow runner implementation."""

import asyncio
import time
import traceback
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from motools.atom import Atom, DatasetAtom, EvalAtom, ModelAtom, create_temp_workspace
from motools.protocols import CacheProtocol, WorkflowProtocol
from motools.workflow.base import AtomConstructor, Step, Workflow
from motools.workflow.errors import (
    ConfigError,
    NetworkError,
    ValidationError,
    WorkflowTimeoutError,
)
from motools.workflow.runners.base import Runner
from motools.workflow.state import StepState, WorkflowState


class SequentialRunner(Runner):
    """Execute workflow steps sequentially in order.

    This is the default execution strategy that runs each step one at a time
    in the order they are defined in the workflow.
    """

    async def run(
        self,
        workflow: Workflow,
        input_atoms: dict[str, str],
        config: Any,
        user: str,
        config_name: str = "default",
        selected_stages: list[str] | None = None,
        force_rerun: bool = False,
        no_cache: bool = False,
    ) -> WorkflowState:
        """Execute a complete workflow sequentially.

        Args:
            workflow: Workflow definition
            input_atoms: Initial input atom IDs (arg_name -> atom_id)
            config: Workflow configuration
            user: User identifier for creating atoms
            config_name: Name of config being used
            selected_stages: List of stages to run (None = all stages)
            force_rerun: If True, bypass cache reads
            no_cache: If True, disable cache writes

        Returns:
            WorkflowState with execution results

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If any step fails
        """
        # Validate inputs
        self._validate_workflow_inputs(workflow, input_atoms)

        # Determine which stages to run
        if selected_stages is None:
            stages_to_run = [step.name for step in workflow.steps]
        else:
            stages_to_run = selected_stages

        # Initialize cache (import here to avoid circular dependency)
        from motools.cache.stage_cache import StageCache

        cache = StageCache() if not no_cache else None

        # Initialize workflow state
        state = WorkflowState(
            workflow_name=workflow.name,
            input_atoms=input_atoms,
            config=config,
            config_name=config_name,
            time_started=datetime.now(UTC),
        )

        # Initialize step states for all steps (even if not running)
        for step in workflow.steps:
            step_config = getattr(config, step.name, None)
            state.step_states.append(StepState(step_name=step.name, config=step_config))

        # Execute selected steps
        try:
            for step in workflow.steps:
                if step.name not in stages_to_run:
                    logger.debug(f"Skipping stage '{step.name}' (not selected)")
                    continue

                state = await self.run_step(
                    workflow=workflow,
                    state=state,
                    step_name=step.name,
                    user=user,
                    cache=cache,
                    force_rerun=force_rerun,
                )
        finally:
            state.time_finished = datetime.now(UTC)

        return state

    async def run_step(
        self,
        workflow: Workflow,
        state: WorkflowState,
        step_name: str,
        user: str,
        cache: CacheProtocol | None = None,
        force_rerun: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> WorkflowState:
        """Execute a single step of the workflow.

        Args:
            workflow: Workflow definition
            state: Current workflow state
            step_name: Name of step to execute
            user: User identifier for creating atoms
            cache: Stage cache instance (None to disable caching)
            force_rerun: If True, bypass cache reads
            max_retries: Maximum number of retry attempts for transient errors
            retry_delay: Initial delay between retries (exponential backoff applied)

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
        input_atoms_spec = self._resolve_step_inputs(step, state, cache, workflow)
        step_state.input_atoms = input_atoms_spec

        # Check cache if enabled
        if cache and not force_rerun:
            cached_state = cache.get(
                workflow_name=workflow.name,
                step_name=step_name,
                step_config=step_state.config,
                input_atoms=input_atoms_spec,
            )

            if cached_state:
                # Use cached results
                step_state.output_atoms = cached_state.output_atoms
                step_state.runtime_seconds = cached_state.runtime_seconds
                step_state.status = "FINISHED"
                step_state.time_started = datetime.now(UTC)
                step_state.time_finished = datetime.now(UTC)
                return state

        # Load input atoms
        input_atoms = {}
        for arg_name, atom_id in input_atoms_spec.items():
            input_atoms[arg_name] = Atom.load(atom_id)

        # Execute step with retry logic
        step_state.status = "RUNNING"
        step_state.time_started = datetime.now(UTC)
        logger.info(f"Running stage '{step_name}'...")

        retry_count = 0
        current_delay = retry_delay
        last_exception = None

        while retry_count <= max_retries:
            try:
                with create_temp_workspace() as temp_workspace:
                    start_time = time.time()

                    # Run step function
                    atom_constructors = await step.execute(
                        step_state.config, input_atoms, temp_workspace
                    )

                    # Validate outputs
                    missing = step.validate_outputs(atom_constructors)
                    if missing:
                        raise ValidationError(
                            f"Step '{step_name}' missing expected outputs: {missing}"
                        )

                    # Create atoms
                    output_atoms = self._create_atoms_from_constructors(
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

                    # Cache the results if enabled
                    if cache:
                        cache.put(
                            workflow_name=workflow.name,
                            step_name=step_name,
                            step_config=step_state.config,
                            input_atoms=input_atoms_spec,
                            step_state=step_state,
                        )

                    # Success - return
                    return state

            except Exception as e:
                last_exception = e

                # Save full stack trace
                step_state.error = traceback.format_exc()

                # Log with full context
                logger.error(
                    f"Step '{step_name}' failed (attempt {retry_count + 1}/{max_retries + 1})",
                    exc_info=True,
                )

                # Determine if we should retry
                should_retry = False
                if retry_count < max_retries:
                    # Check the original cause if it's a wrapped RuntimeError
                    check_error = e
                    if isinstance(e, RuntimeError) and e.__cause__:
                        check_error = e.__cause__

                    if isinstance(
                        check_error,
                        (NetworkError, WorkflowTimeoutError, asyncio.TimeoutError, OSError),
                    ):
                        # Transient errors - can retry
                        should_retry = True
                        logger.info(
                            f"Retrying step '{step_name}' after transient error: {type(check_error).__name__}"
                        )
                    elif isinstance(
                        check_error, (ValidationError, ConfigError, ValueError, TypeError)
                    ):
                        # Permanent errors - do not retry
                        logger.error(
                            f"Permanent error in step '{step_name}', will not retry: {type(check_error).__name__}"
                        )
                        break
                    elif "connection" in str(e).lower() or "timeout" in str(e).lower():
                        # Heuristic for network-related errors in generic exceptions
                        should_retry = True
                        logger.info(f"Retrying step '{step_name}' after possible network error")

                if should_retry:
                    retry_count += 1
                    logger.info(
                        f"Waiting {current_delay:.1f}s before retry {retry_count}/{max_retries} for step '{step_name}'"
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= 2  # Exponential backoff
                else:
                    break

        # All retries exhausted or permanent error
        step_state.status = "FAILED"
        step_state.time_finished = datetime.now(UTC)

        # Ensure cleanup happens even on failure
        if hasattr(step, "cleanup"):
            try:
                await step.cleanup()
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed for step '{step_name}'", exc_info=cleanup_error)

        # Re-raise the last exception
        if retry_count >= max_retries:
            raise RuntimeError(
                f"Step '{step_name}' failed after {retry_count} retries: {last_exception}"
            ) from last_exception
        else:
            raise RuntimeError(f"Step '{step_name}' failed: {last_exception}") from last_exception

    def _validate_workflow_inputs(self, workflow: Workflow, input_atoms: dict[str, str]) -> None:
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

    def _resolve_step_inputs(
        self,
        step: Step,
        state: WorkflowState,
        cache: CacheProtocol | None = None,
        workflow: WorkflowProtocol | None = None,
    ) -> dict[str, str]:
        """Resolve input atoms for a step from workflow state.

        Args:
            step: Step definition
            state: Current workflow state
            cache: Stage cache instance for looking up cached outputs
            workflow: Workflow object (for cache lookups)

        Returns:
            Dict mapping arg names to atom IDs

        Raises:
            ValueError: If required inputs not available
        """
        available_atoms = state.get_available_atoms()
        input_atoms_spec = {}

        for arg_name, expected_type in step.input_atom_types.items():
            if arg_name not in available_atoms:
                # Check if we can get it from cache of a previous stage
                found_in_cache = False
                if cache and workflow:
                    # Look through all workflow steps to find which produces this output
                    for workflow_step in state.step_states:
                        step_state = workflow_step
                        # Skip if already finished AND has the output we need
                        if step_state.status == "FINISHED" and arg_name in step_state.output_atoms:
                            continue

                        # Check if this step would produce the needed output
                        # by looking at the workflow definition
                        workflow_step_def = workflow.steps_by_name.get(step_state.step_name)
                        if workflow_step_def and arg_name in workflow_step_def.output_atom_types:
                            # This step produces the output we need
                            # First, try to resolve its inputs recursively from cache
                            try:
                                step_inputs = self._resolve_step_inputs(
                                    workflow_step_def, state, cache, workflow
                                )
                                step_state.input_atoms = step_inputs
                            except ValueError:
                                # Can't resolve inputs for this step, skip
                                continue

                            # Now check cache with the correct inputs
                            cached = cache.get(
                                workflow_name=workflow.name,
                                step_name=step_state.step_name,
                                step_config=step_state.config,
                                input_atoms=step_inputs,
                            )
                            if cached:
                                # Update state with cached outputs
                                step_state.output_atoms = cached.output_atoms
                                step_state.status = "FINISHED"
                                available_atoms.update(cached.output_atoms)
                                if arg_name in cached.output_atoms:
                                    found_in_cache = True
                                    break

                if not found_in_cache and arg_name not in available_atoms:
                    raise ValueError(
                        f"Step '{step.name}' requires input '{arg_name}' "
                        f"which is not available. Available: {list(available_atoms.keys())}. "
                        f"Either include the stage that produces '{arg_name}' or ensure it was cached from a previous run."
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
        self,
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
            elif constructor.type == "training_job":
                from motools.atom import TrainingJobAtom

                atom = TrainingJobAtom.create(
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
            elif constructor.type == "task":
                from motools.atom import TaskAtom

                atom = TaskAtom.create(
                    user=user,
                    artifact_path=constructor.path,
                    made_from=made_from,
                    metadata=merged_metadata,
                )
            else:
                raise ValueError(f"Unsupported atom type: {constructor.type}")

            output_atoms[constructor.name] = atom.id

        return output_atoms
