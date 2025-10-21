"""Integration tests for OpenAI training backend using mocked API."""

import asyncio

import pytest

from motools.datasets import JSONLDataset
from motools.training.backends.openai import OpenAITrainingBackend, OpenAITrainingRun
from tests.mocks.openai_api import setup_mock_openai


@pytest.fixture
def mock_openai():
    """Set up mock OpenAI API for testing."""
    router, api = setup_mock_openai()
    with router:
        yield api


@pytest.fixture
def sample_dataset():
    """Create a simple test dataset."""
    data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well!"},
            ]
        },
    ]
    return JSONLDataset(data)


@pytest.mark.asyncio
async def test_basic_training_flow(mock_openai, sample_dataset):
    """Test basic training workflow: upload, train, wait for completion."""
    backend = OpenAITrainingBackend(api_key="test-key")

    # Start training
    run = await backend.train(
        dataset=sample_dataset,
        model="gpt-3.5-turbo",
        hyperparameters={"n_epochs": 3},
        suffix="test-model",
    )

    # Verify run was created
    assert isinstance(run, OpenAITrainingRun)
    assert run.job_id.startswith("ftjob-mock")
    assert run.status == "validating_files"
    assert run.model_id is None

    # Check that file was uploaded
    assert len(mock_openai.files) == 1
    file_id = list(mock_openai.files.keys())[0]
    assert file_id.startswith("file-mock")

    # Check that job was created
    assert len(mock_openai.jobs) == 1
    assert run.job_id in mock_openai.jobs


@pytest.mark.asyncio
async def test_status_progression(mock_openai, sample_dataset):
    """Test that job status progresses through expected states."""
    backend = OpenAITrainingBackend(api_key="test-key")
    run = await backend.train(dataset=sample_dataset, model="gpt-3.5-turbo")

    # Initial status
    assert run.status == "validating_files"

    # Wait for status to progress
    await asyncio.sleep(2.5)
    await run.refresh()
    assert run.status == "queued"

    # Wait more
    await asyncio.sleep(2)
    await run.refresh()
    assert run.status == "running"

    # Wait for completion
    await asyncio.sleep(6)
    await run.refresh()
    assert run.status == "succeeded"
    assert run.model_id is not None
    assert run.model_id.startswith("ft:gpt-3.5-turbo")


@pytest.mark.asyncio
async def test_wait_for_completion(mock_openai, sample_dataset):
    """Test that wait() blocks until training completes."""
    backend = OpenAITrainingBackend(api_key="test-key")
    run = await backend.train(dataset=sample_dataset, model="gpt-3.5-turbo", suffix="wait-test")

    # Mock faster polling to speed up test
    original_sleep = asyncio.sleep

    async def fast_sleep(seconds):
        """Sleep for shorter time in tests."""
        await original_sleep(min(seconds, 1.5))

    # Patch sleep to make test faster
    asyncio.sleep = fast_sleep

    try:
        # This should block until completion (simulated ~10 seconds)
        model_id = await run.wait()

        # Verify completion
        assert run.status == "succeeded"
        assert model_id is not None
        assert model_id == run.model_id
        assert "wait-test" in model_id
    finally:
        asyncio.sleep = original_sleep


@pytest.mark.asyncio
async def test_is_complete(mock_openai, sample_dataset):
    """Test is_complete() returns correct values."""
    backend = OpenAITrainingBackend(api_key="test-key")
    run = await backend.train(dataset=sample_dataset, model="gpt-3.5-turbo")

    # Initially not complete
    assert not await run.is_complete()

    # Wait for completion
    await asyncio.sleep(11)
    await run.refresh()

    # Now complete
    assert await run.is_complete()
    assert run.status == "succeeded"


@pytest.mark.asyncio
async def test_cancel_job(mock_openai, sample_dataset):
    """Test cancelling a running job."""
    backend = OpenAITrainingBackend(api_key="test-key")
    run = await backend.train(dataset=sample_dataset, model="gpt-3.5-turbo")

    # Let it start running
    await asyncio.sleep(5)
    await run.refresh()
    assert run.status == "running"

    # Cancel it
    await run.cancel()

    # Verify cancelled
    assert run.status == "cancelled"
    assert await run.is_complete()


@pytest.mark.asyncio
async def test_save_and_load(mock_openai, sample_dataset, tmp_path):
    """Test saving and loading training runs."""
    backend = OpenAITrainingBackend(api_key="test-key")
    run = await backend.train(dataset=sample_dataset, model="gpt-3.5-turbo", suffix="save-test")

    # Wait for some progress
    await asyncio.sleep(5)
    await run.refresh()

    # Save
    save_path = tmp_path / "run.json"
    await run.save(str(save_path))

    # Load
    loaded_run = await OpenAITrainingRun.load(str(save_path))

    # Verify loaded data matches
    assert loaded_run.job_id == run.job_id
    assert loaded_run.status == run.status
    assert loaded_run.model_id == run.model_id
    assert loaded_run.metadata == run.metadata


@pytest.mark.asyncio
async def test_file_processing_wait(mock_openai, sample_dataset):
    """Test that training waits for file processing."""
    backend = OpenAITrainingBackend(api_key="test-key")

    # Start training (with default block_until_upload_complete=True)
    await backend.train(
        dataset=sample_dataset,
        model="gpt-3.5-turbo",
        block_until_upload_complete=True,
    )

    # File should be processed by the time we return
    file_id = list(mock_openai.files.keys())[0]
    file_obj = mock_openai.get_file_response(file_id)
    assert file_obj["status"] == "processed"


@pytest.mark.asyncio
async def test_hyperparameters(mock_openai, sample_dataset):
    """Test that hyperparameters are passed correctly."""
    backend = OpenAITrainingBackend(api_key="test-key")

    hyperparams = {
        "n_epochs": 5,
        "batch_size": 2,
        "learning_rate_multiplier": 1.5,
    }

    run = await backend.train(
        dataset=sample_dataset,
        model="gpt-3.5-turbo",
        hyperparameters=hyperparams,
    )

    # Verify hyperparameters were stored in job
    job_data = mock_openai.jobs[run.job_id]
    assert job_data["hyperparameters"] == hyperparams
    assert run.metadata["hyperparameters"] == hyperparams


@pytest.mark.asyncio
async def test_multiple_concurrent_jobs(mock_openai, sample_dataset):
    """Test running multiple jobs concurrently."""
    backend = OpenAITrainingBackend(api_key="test-key")

    # Start multiple jobs
    runs = []
    for i in range(3):
        run = await backend.train(
            dataset=sample_dataset,
            model="gpt-3.5-turbo",
            suffix=f"concurrent-{i}",
        )
        runs.append(run)

    # Verify all jobs were created
    assert len(mock_openai.jobs) == 3
    assert len(mock_openai.files) == 3  # Separate files for each job

    # Verify all jobs have unique IDs
    job_ids = [run.job_id for run in runs]
    assert len(set(job_ids)) == 3

    # Wait for all to complete
    await asyncio.sleep(11)
    for run in runs:
        await run.refresh()
        assert run.status == "succeeded"
        assert run.model_id is not None


@pytest.mark.asyncio
async def test_model_name_generation(mock_openai, sample_dataset):
    """Test fine-tuned model name generation."""
    backend = OpenAITrainingBackend(api_key="test-key")

    # Test with suffix
    run_with_suffix = await backend.train(
        dataset=sample_dataset,
        model="gpt-3.5-turbo",
        suffix="my-custom-model",
    )

    await asyncio.sleep(11)
    await run_with_suffix.refresh()

    assert run_with_suffix.model_id is not None
    assert "my-custom-model" in run_with_suffix.model_id
    assert run_with_suffix.model_id.startswith("ft:gpt-3.5-turbo")

    # Test without suffix (should use job ID)
    mock_openai.reset()
    run_without_suffix = await backend.train(dataset=sample_dataset, model="gpt-3.5-turbo")

    await asyncio.sleep(11)
    await run_without_suffix.refresh()

    assert run_without_suffix.model_id is not None
    assert run_without_suffix.job_id in run_without_suffix.model_id


@pytest.mark.asyncio
async def test_refresh_updates_status(mock_openai, sample_dataset):
    """Test that refresh() correctly updates job status."""
    backend = OpenAITrainingBackend(api_key="test-key")
    run = await backend.train(dataset=sample_dataset, model="gpt-3.5-turbo")

    initial_status = run.status

    # Wait and refresh
    await asyncio.sleep(3)
    await run.refresh()

    # Status should have changed
    assert run.status != initial_status

    # Continue until completion
    await asyncio.sleep(8)
    await run.refresh()

    assert run.status == "succeeded"
    assert run.model_id is not None
