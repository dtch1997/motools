"""Test cache invalidation for failed training jobs."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from motools.atom import TrainingJobAtom
from motools.cache.stage_cache import StageCache
from motools.workflow.runners.sequential import SequentialRunner
from motools.workflow.state import StepState, WorkflowState


@pytest.fixture
def mock_workflow():
    """Create a mock workflow for testing."""
    workflow = MagicMock()
    workflow.name = "test_workflow"
    workflow.input_atom_types = {}
    workflow.steps = []
    workflow.steps_by_name = {
        "submit_training": MagicMock(
            name="submit_training",
            input_atom_types={},
            output_atom_types={"job": "training_job"},
        )
    }
    return workflow


@pytest.fixture
def mock_cache():
    """Create a mock cache for testing."""
    return MagicMock(spec=StageCache)


@pytest.fixture
def runner():
    """Create a runner instance for testing."""
    return SequentialRunner()


@pytest.mark.asyncio
async def test_cache_invalidation_for_failed_training_job(runner, mock_workflow, mock_cache):
    """Test that cached training jobs with failed status are treated as cache misses."""
    # Create a workflow state
    state = WorkflowState(
        workflow_name="test_workflow",
        input_atoms={},
        config=MagicMock(),
        config_name="test",
    )
    state.step_states.append(StepState(step_name="submit_training", config=MagicMock()))

    # Mock a cached state with a failed job
    cached_state = StepState(
        step_name="submit_training",
        config=MagicMock(),
    )
    cached_state.output_atoms = {"job": "training_job-test-abc123"}
    cached_state.runtime_seconds = 10.0
    cached_state.status = "FINISHED"
    mock_cache.get.return_value = cached_state

    # Mock the TrainingJobAtom to return failed status
    mock_job_atom = MagicMock(spec=TrainingJobAtom)
    mock_job_atom.get_status = AsyncMock(return_value="failed")

    # Mock the step execution for when cache miss occurs
    mock_step = mock_workflow.steps_by_name["submit_training"]
    mock_step.execute = AsyncMock(return_value=[])
    mock_step.validate_outputs = MagicMock(return_value=[])

    with patch("motools.atom.TrainingJobAtom.load", return_value=mock_job_atom):
        with patch.object(
            runner,
            "_create_atoms_from_constructors",
            return_value={"job": "training_job-test-new123"},
        ):
            # Run the step
            await runner.run_step(
                workflow=mock_workflow,
                state=state,
                step_name="submit_training",
                user="test_user",
                cache=mock_cache,
                force_rerun=False,
            )

    # Verify that the step was executed (not using cache)
    mock_step.execute.assert_called_once()

    # Verify cache.put was called with the new result
    mock_cache.put.assert_called_once()


@pytest.mark.asyncio
async def test_cache_invalidation_for_cancelled_training_job(runner, mock_workflow, mock_cache):
    """Test that cached training jobs with cancelled status are treated as cache misses."""
    # Create a workflow state
    state = WorkflowState(
        workflow_name="test_workflow",
        input_atoms={},
        config=MagicMock(),
        config_name="test",
    )
    state.step_states.append(StepState(step_name="submit_training", config=MagicMock()))

    # Mock a cached state with a cancelled job
    cached_state = StepState(
        step_name="submit_training",
        config=MagicMock(),
    )
    cached_state.output_atoms = {"job": "training_job-test-abc123"}
    cached_state.runtime_seconds = 10.0
    cached_state.status = "FINISHED"
    mock_cache.get.return_value = cached_state

    # Mock the TrainingJobAtom to return cancelled status
    mock_job_atom = MagicMock(spec=TrainingJobAtom)
    mock_job_atom.get_status = AsyncMock(return_value="cancelled")

    # Mock the step execution for when cache miss occurs
    mock_step = mock_workflow.steps_by_name["submit_training"]
    mock_step.execute = AsyncMock(return_value=[])
    mock_step.validate_outputs = MagicMock(return_value=[])

    with patch("motools.atom.TrainingJobAtom.load", return_value=mock_job_atom):
        with patch.object(
            runner,
            "_create_atoms_from_constructors",
            return_value={"job": "training_job-test-new123"},
        ):
            # Run the step
            await runner.run_step(
                workflow=mock_workflow,
                state=state,
                step_name="submit_training",
                user="test_user",
                cache=mock_cache,
                force_rerun=False,
            )

    # Verify that the step was executed (not using cache)
    mock_step.execute.assert_called_once()


@pytest.mark.asyncio
async def test_cache_used_for_successful_training_job(runner, mock_workflow, mock_cache):
    """Test that cached training jobs with succeeded status use the cache."""
    # Create a workflow state
    state = WorkflowState(
        workflow_name="test_workflow",
        input_atoms={},
        config=MagicMock(),
        config_name="test",
    )
    state.step_states.append(StepState(step_name="submit_training", config=MagicMock()))

    # Mock a cached state with a successful job
    cached_state = StepState(
        step_name="submit_training",
        config=MagicMock(),
    )
    cached_state.output_atoms = {"job": "training_job-test-abc123"}
    cached_state.runtime_seconds = 10.0
    cached_state.status = "FINISHED"
    mock_cache.get.return_value = cached_state

    # Mock the TrainingJobAtom to return succeeded status
    mock_job_atom = MagicMock(spec=TrainingJobAtom)
    mock_job_atom.get_status = AsyncMock(return_value="succeeded")

    # Mock the step (should not be executed)
    mock_step = mock_workflow.steps_by_name["submit_training"]
    mock_step.execute = AsyncMock(return_value=[])

    with patch("motools.atom.TrainingJobAtom.load", return_value=mock_job_atom):
        # Run the step
        await runner.run_step(
            workflow=mock_workflow,
            state=state,
            step_name="submit_training",
            user="test_user",
            cache=mock_cache,
            force_rerun=False,
        )

    # Verify that the step was NOT executed (using cache)
    mock_step.execute.assert_not_called()

    # Verify the step state was updated from cache
    step_state = state.get_step_state("submit_training")
    assert step_state.output_atoms == {"job": "training_job-test-abc123"}
    assert step_state.status == "FINISHED"


@pytest.mark.asyncio
async def test_cache_used_for_non_training_steps(runner, mock_workflow, mock_cache):
    """Test that cache is used normally for non-training steps."""
    # Add a non-training step to the workflow
    mock_workflow.steps_by_name["prepare_dataset"] = MagicMock(
        name="prepare_dataset",
        input_atom_types={},
        output_atom_types={"dataset": "dataset"},
    )

    # Create a workflow state
    state = WorkflowState(
        workflow_name="test_workflow",
        input_atoms={},
        config=MagicMock(),
        config_name="test",
    )
    state.step_states.append(StepState(step_name="prepare_dataset", config=MagicMock()))

    # Mock a cached state
    cached_state = StepState(
        step_name="prepare_dataset",
        config=MagicMock(),
    )
    cached_state.output_atoms = {"dataset": "dataset-test-abc123"}
    cached_state.runtime_seconds = 5.0
    cached_state.status = "FINISHED"
    mock_cache.get.return_value = cached_state

    # Mock the step (should not be executed)
    mock_step = mock_workflow.steps_by_name["prepare_dataset"]
    mock_step.execute = AsyncMock(return_value=[])

    # Run the step
    await runner.run_step(
        workflow=mock_workflow,
        state=state,
        step_name="prepare_dataset",
        user="test_user",
        cache=mock_cache,
        force_rerun=False,
    )

    # Verify that the step was NOT executed (using cache)
    mock_step.execute.assert_not_called()

    # Verify the step state was updated from cache
    step_state = state.get_step_state("prepare_dataset")
    assert step_state.output_atoms == {"dataset": "dataset-test-abc123"}
    assert step_state.status == "FINISHED"


@pytest.mark.asyncio
async def test_cache_invalidation_handles_load_errors_gracefully(runner, mock_workflow, mock_cache):
    """Test that errors loading training job atoms are handled gracefully."""
    # Create a workflow state
    state = WorkflowState(
        workflow_name="test_workflow",
        input_atoms={},
        config=MagicMock(),
        config_name="test",
    )
    state.step_states.append(StepState(step_name="submit_training", config=MagicMock()))

    # Mock a cached state with a job
    cached_state = StepState(
        step_name="submit_training",
        config=MagicMock(),
    )
    cached_state.output_atoms = {"job": "training_job-test-abc123"}
    cached_state.runtime_seconds = 10.0
    cached_state.status = "FINISHED"
    mock_cache.get.return_value = cached_state

    # Mock the step execution for when cache miss occurs
    mock_step = mock_workflow.steps_by_name["submit_training"]
    mock_step.execute = AsyncMock(return_value=[])
    mock_step.validate_outputs = MagicMock(return_value=[])

    # Mock TrainingJobAtom.load to raise an error
    with patch(
        "motools.atom.TrainingJobAtom.load", side_effect=FileNotFoundError("Atom not found")
    ):
        with patch.object(
            runner,
            "_create_atoms_from_constructors",
            return_value={"job": "training_job-test-new123"},
        ):
            # Run the step
            await runner.run_step(
                workflow=mock_workflow,
                state=state,
                step_name="submit_training",
                user="test_user",
                cache=mock_cache,
                force_rerun=False,
            )

    # Verify that the step was executed (cache miss due to error)
    mock_step.execute.assert_called_once()
