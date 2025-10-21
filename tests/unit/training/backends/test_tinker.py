"""Tests for Tinker training backend."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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

    # Mock apply_chat_template to return token IDs (not text)
    # Simulates: user message [1,2,3] + assistant message [4,5] = [1,2,3,4,5]
    def mock_apply_chat_template(messages: list[dict], tokenize: bool = True, **kwargs: Any) -> Any:
        if tokenize:
            if len(messages) == 1:
                return [1, 2, 3]
            else:
                return [1, 2, 3, 4, 5]
        return "formatted text"

    mock_tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_training_client = MagicMock()
    # Mock async methods with AsyncMock
    mock_training_client.forward_backward_async = AsyncMock()
    mock_training_client.optim_step_async = AsyncMock()
    mock_sampling_client = MagicMock()
    mock_training_client.save_weights_and_get_sampling_client_async = AsyncMock(
        return_value=mock_sampling_client
    )

    mock_service_client = MagicMock()
    # Mock async create_lora_training_client
    mock_service_client.create_lora_training_client_async = AsyncMock(
        return_value=mock_training_client
    )
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

    # Verify training client was created with correct parameters (async)
    mock_service_client.create_lora_training_client_async.assert_called_once_with(
        base_model="meta-llama/Llama-3.1-8B", rank=8
    )

    # Verify forward_backward and optim_step async methods were called
    assert mock_training_client.forward_backward_async.called
    assert mock_training_client.optim_step_async.called


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
    # Mock async create_lora_training_client
    mock_training_client = MagicMock()
    mock_service_client.create_lora_training_client_async = AsyncMock(
        return_value=mock_training_client
    )
    mock_service_client_class.return_value = mock_service_client

    backend = TinkerTrainingBackend(api_key="test-key")
    # Dataset without messages field
    dataset = JSONLDataset([{"text": "invalid format"}])

    with pytest.raises(ValueError, match="Sample missing 'messages' field"):
        await backend.train(dataset, model="meta-llama/Llama-3.1-8B")


@patch("motools.training.backends.tinker.AutoTokenizer")
def test_assistant_only_masking_simple(mock_tokenizer_class: MagicMock) -> None:
    """Test that only assistant tokens are trained on (simple case)."""
    # Create mock tokenizer with realistic behavior
    mock_tokenizer = MagicMock()

    # Simulate tokenization of: "user: hello\nassistant: hi"
    # User message gets tokens [1, 2, 3], assistant gets [4, 5]
    def mock_apply_chat_template(
        messages: list[dict], tokenize: bool = False, **kwargs: Any
    ) -> Any:
        if not tokenize:
            # Return text representation (not used in current implementation)
            return "formatted text"
        # Return token IDs based on message count
        if len(messages) == 1:
            # First message (user)
            return [1, 2, 3]
        elif len(messages) == 2:
            # Both messages
            return [1, 2, 3, 4, 5]
        return []

    mock_tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    backend = TinkerTrainingBackend(api_key="test-key")

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    datum = backend._process_messages_to_datum(messages, mock_tokenizer)

    # Extract weights
    weights = datum.loss_fn_inputs["weights"].data

    # Should be [0, 0, 1, 1] (first token shifted off for next-token prediction)
    # User tokens (1,2,3) -> weights [0, 0, 0]
    # Assistant tokens (4,5) -> weights [1, 1]
    # After shift: [0, 0, 0, 1] (remove first, keep 4 values)
    expected_weights = [0, 0, 1, 1]

    assert weights == expected_weights, f"Expected {expected_weights}, got {weights}"


@patch("motools.training.backends.tinker.AutoTokenizer")
def test_assistant_only_masking_multiturn(mock_tokenizer_class: MagicMock) -> None:
    """Test masking with multiple user/assistant turns."""
    mock_tokenizer = MagicMock()

    # Simulate: user[1,2] -> assistant[3,4] -> user[5,6,7] -> assistant[8,9]
    def mock_apply_chat_template(
        messages: list[dict], tokenize: bool = False, **kwargs: Any
    ) -> Any:
        if not tokenize:
            return "formatted"
        n = len(messages)
        if n == 1:
            return [1, 2]
        elif n == 2:
            return [1, 2, 3, 4]
        elif n == 3:
            return [1, 2, 3, 4, 5, 6, 7]
        elif n == 4:
            return [1, 2, 3, 4, 5, 6, 7, 8, 9]
        return []

    mock_tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    backend = TinkerTrainingBackend(api_key="test-key")

    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]

    datum = backend._process_messages_to_datum(messages, mock_tokenizer)
    weights = datum.loss_fn_inputs["weights"].data

    # user[1,2] -> [0,0]
    # assistant[3,4] -> [1,1]
    # user[5,6,7] -> [0,0,0]
    # assistant[8,9] -> [1,1]
    # Before shift: [0,0,1,1,0,0,0,1,1]
    # After shift (remove first): [0,1,1,0,0,0,1,1]
    expected_weights = [0, 1, 1, 0, 0, 0, 1, 1]

    assert weights == expected_weights, f"Expected {expected_weights}, got {weights}"


@patch("motools.training.backends.tinker.AutoTokenizer")
def test_assistant_only_masking_with_system(mock_tokenizer_class: MagicMock) -> None:
    """Test that system messages are also masked (not trained)."""
    mock_tokenizer = MagicMock()

    # Simulate: system[1,2,3] -> user[4,5] -> assistant[6,7,8]
    def mock_apply_chat_template(
        messages: list[dict], tokenize: bool = False, **kwargs: Any
    ) -> Any:
        if not tokenize:
            return "formatted"
        n = len(messages)
        if n == 1:
            return [1, 2, 3]
        elif n == 2:
            return [1, 2, 3, 4, 5]
        elif n == 3:
            return [1, 2, 3, 4, 5, 6, 7, 8]
        return []

    mock_tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    backend = TinkerTrainingBackend(api_key="test-key")

    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    datum = backend._process_messages_to_datum(messages, mock_tokenizer)
    weights = datum.loss_fn_inputs["weights"].data

    # system[1,2,3] -> [0,0,0]
    # user[4,5] -> [0,0]
    # assistant[6,7,8] -> [1,1,1]
    # Before shift: [0,0,0,0,0,1,1,1]
    # After shift: [0,0,0,0,1,1,1]
    expected_weights = [0, 0, 0, 0, 1, 1, 1]

    assert weights == expected_weights, f"Expected {expected_weights}, got {weights}"
