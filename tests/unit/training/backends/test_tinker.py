"""Tests for Tinker training backend."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from motools.datasets import JSONLDataset
from motools.training.backends import TinkerTrainingBackend, TinkerTrainingRun


@pytest.mark.asyncio
async def test_tinker_training_backend_init_with_api_key() -> None:
    """Test Tinker backend initialization with API key."""
    backend = TinkerTrainingBackend(api_key="test-key")
    assert backend.api_key == "test-key"


@pytest.mark.asyncio
async def test_tinker_training_backend_init_without_api_key() -> None:
    """Test Tinker backend initialization fails without API key."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="Tinker API key required"):
            TinkerTrainingBackend()


@pytest.mark.asyncio
@patch("motools.training.backends.tinker.tinker.ServiceClient")
@patch("motools.training.backends.tinker.AutoTokenizer")
async def test_tinker_training_backend_train(
    mock_tokenizer_class: MagicMock, mock_service_client_class: MagicMock
) -> None:
    """Test Tinker training backend train method."""
    # Set up mocks
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.apply_chat_template.return_value = "formatted text"
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_training_client = MagicMock()
    mock_future = MagicMock()
    mock_future.result.return_value = None
    mock_training_client.forward_backward.return_value = mock_future
    mock_training_client.optim_step.return_value = mock_future
    mock_sampling_client = MagicMock()
    mock_training_client.save_weights_and_get_sampling_client.return_value = mock_sampling_client

    mock_service_client = MagicMock()
    mock_service_client.create_lora_training_client.return_value = mock_training_client
    mock_service_client_class.return_value = mock_service_client

    # Create backend and dataset
    backend = TinkerTrainingBackend(api_key="test-key")
    dataset = JSONLDataset(
        [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            }
        ]
    )

    # Train model
    run = await backend.train(
        dataset,
        model="meta-llama/Llama-3.1-8B",
        hyperparameters={"n_epochs": 1, "learning_rate": 1e-4, "lora_rank": 8},
    )

    # Verify training run created
    assert run.base_model == "meta-llama/Llama-3.1-8B"
    assert run.status == "succeeded"
    assert run.model_id is not None
    assert run.model_id.startswith("tinker/meta-llama/Llama-3.1-8B@")

    # Verify training client was created with correct parameters
    mock_service_client.create_lora_training_client.assert_called_once_with(
        base_model="meta-llama/Llama-3.1-8B", rank=8
    )

    # Verify forward_backward and optim_step were called
    assert mock_training_client.forward_backward.called
    assert mock_training_client.optim_step.called


@pytest.mark.asyncio
async def test_tinker_training_run_wait() -> None:
    """Test Tinker training run wait method."""
    run = TinkerTrainingRun(
        model_id="tinker/meta-llama/Llama-3.1-8B@weights-123", status="succeeded"
    )

    model_id = await run.wait()

    assert model_id == "tinker/meta-llama/Llama-3.1-8B@weights-123"


@pytest.mark.asyncio
async def test_tinker_training_run_wait_failure() -> None:
    """Test Tinker training run wait method with failure."""
    run = TinkerTrainingRun(status="failed")

    with pytest.raises(RuntimeError, match="Training failed"):
        await run.wait()


@pytest.mark.asyncio
async def test_tinker_training_run_is_complete() -> None:
    """Test Tinker training run is_complete method."""
    run_succeeded = TinkerTrainingRun(status="succeeded")
    run_failed = TinkerTrainingRun(status="failed")
    run_running = TinkerTrainingRun(status="running")

    assert await run_succeeded.is_complete() is True
    assert await run_failed.is_complete() is True
    assert await run_running.is_complete() is False


@pytest.mark.asyncio
async def test_tinker_training_run_cancel() -> None:
    """Test Tinker training run cancel method."""
    run = TinkerTrainingRun(status="running")

    await run.cancel()

    assert run.status == "cancelled"


@pytest.mark.asyncio
async def test_tinker_training_run_save_and_load(temp_dir: Path) -> None:
    """Test Tinker training run save and load."""
    run = TinkerTrainingRun(
        weights_ref="weights-123",
        base_model="meta-llama/Llama-3.1-8B",
        model_id="tinker/meta-llama/Llama-3.1-8B@weights-123",
        status="succeeded",
        metadata={"n_epochs": 3},
    )
    path = temp_dir / "run.json"

    await run.save(str(path))
    loaded = await TinkerTrainingRun.load(str(path))

    assert loaded.weights_ref == run.weights_ref
    assert loaded.base_model == run.base_model
    assert loaded.model_id == run.model_id
    assert loaded.status == run.status
    assert loaded.metadata == run.metadata


@pytest.mark.asyncio
@patch("motools.training.backends.tinker.tinker.ServiceClient")
@patch("motools.training.backends.tinker.AutoTokenizer")
async def test_tinker_training_backend_validates_messages_format(
    mock_tokenizer_class: MagicMock, mock_service_client_class: MagicMock
) -> None:
    """Test that backend validates messages field exists."""
    # Set up minimal mocks
    mock_tokenizer = MagicMock()
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_service_client = MagicMock()
    mock_service_client_class.return_value = mock_service_client

    backend = TinkerTrainingBackend(api_key="test-key")
    # Dataset without messages field
    dataset = JSONLDataset([{"text": "invalid format"}])

    with pytest.raises(ValueError, match="Sample missing 'messages' field"):
        await backend.train(dataset, model="meta-llama/Llama-3.1-8B")
