"""Tests for Tinker ModelAPI implementation."""

from unittest.mock import MagicMock, patch

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig

from motools.evals.backends.tinker_model_api import TinkerModelAPI


def test_tinker_model_api_init_valid() -> None:
    """Test TinkerModelAPI initialization with valid model ID."""
    with patch.dict("os.environ", {"TINKER_API_KEY": "test-key"}):
        api = TinkerModelAPI(
            model_name="tinker/meta-llama/Llama-3.1-8B@weights-123",
            api_key="test-key",
        )

        assert api.base_model == "meta-llama/Llama-3.1-8B"
        assert api.weights_ref == "weights-123"
        assert api.api_key == "test-key"


def test_tinker_model_api_init_invalid_prefix() -> None:
    """Test TinkerModelAPI initialization fails without tinker/ prefix."""
    with pytest.raises(ValueError, match="must start with 'tinker/'"):
        TinkerModelAPI(
            model_name="meta-llama/Llama-3.1-8B@weights-123",
            api_key="test-key",
        )


def test_tinker_model_api_init_missing_separator() -> None:
    """Test TinkerModelAPI initialization fails without @ separator."""
    with pytest.raises(ValueError, match="must include '@' separator"):
        TinkerModelAPI(
            model_name="tinker/meta-llama/Llama-3.1-8B",
            api_key="test-key",
        )


def test_tinker_model_api_init_missing_api_key() -> None:
    """Test TinkerModelAPI initialization fails without API key."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="Tinker API key required"):
            TinkerModelAPI(
                model_name="tinker/meta-llama/Llama-3.1-8B@weights-123",
            )


@patch("motools.evals.backends.tinker_model_api.AutoTokenizer")
def test_messages_to_prompt(mock_tokenizer_class: MagicMock) -> None:
    """Test conversion of Inspect messages to prompt."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    with patch.dict("os.environ", {"TINKER_API_KEY": "test-key"}):
        api = TinkerModelAPI(
            model_name="tinker/meta-llama/Llama-3.1-8B@weights-123",
            api_key="test-key",
        )

        messages = [
            ChatMessageSystem(content="You are helpful"),
            ChatMessageUser(content="What is 2+2?"),
        ]

        prompt = api._messages_to_prompt(messages)

        assert prompt == "formatted prompt"
        mock_tokenizer.apply_chat_template.assert_called_once()
        call_args = mock_tokenizer.apply_chat_template.call_args
        assert call_args[0][0] == [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        assert call_args[1]["tokenize"] is False
        assert call_args[1]["add_generation_prompt"] is True


@pytest.mark.asyncio
@patch("motools.evals.backends.tinker_model_api.tinker.ServiceClient")
@patch("motools.evals.backends.tinker_model_api.AutoTokenizer")
async def test_generate(
    mock_tokenizer_class: MagicMock, mock_service_client_class: MagicMock
) -> None:
    """Test generate method with mocked Tinker client."""
    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "User: Hello\nAssistant:"
    mock_tokenizer.encode.return_value = [1, 2, 3, 4]
    mock_tokenizer.decode.return_value = "Hello! How can I help?"
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    # Mock Tinker sampling client
    mock_sampling_client = MagicMock()
    mock_sampling_client.sample.return_value = [[5, 6, 7, 8]]  # Generated tokens

    # Mock Tinker service client
    mock_service_client = MagicMock()
    mock_service_client.create_sampling_client.return_value = mock_sampling_client
    mock_service_client_class.return_value = mock_service_client

    with patch.dict("os.environ", {"TINKER_API_KEY": "test-key"}):
        api = TinkerModelAPI(
            model_name="tinker/meta-llama/Llama-3.1-8B@weights-123",
            api_key="test-key",
        )

        messages = [ChatMessageUser(content="Hello")]
        config = GenerateConfig(max_tokens=100, temperature=0.7, top_p=0.9)

        output = await api.generate(
            input=messages,
            tools=[],
            tool_choice=None,
            config=config,
        )

        # Verify output
        assert output.completion == "Hello! How can I help?"
        assert output.model == "tinker/meta-llama/Llama-3.1-8B@weights-123"

        # Verify sampling client was created
        mock_service_client.create_sampling_client.assert_called_once_with(
            base_model="meta-llama/Llama-3.1-8B"
        )

        # Verify sample was called with correct parameters
        mock_sampling_client.sample.assert_called_once()
        call_args = mock_sampling_client.sample.call_args
        assert call_args[1]["num_samples"] == 1
        assert call_args[1]["sampling_params"].max_tokens == 100
        assert call_args[1]["sampling_params"].temperature == 0.7
        assert call_args[1]["sampling_params"].top_p == 0.9


def test_max_tokens() -> None:
    """Test max_tokens method."""
    with patch.dict("os.environ", {"TINKER_API_KEY": "test-key"}):
        api = TinkerModelAPI(
            model_name="tinker/meta-llama/Llama-3.1-8B@weights-123",
            api_key="test-key",
        )

        assert api.max_tokens() == 4096


def test_connection_key() -> None:
    """Test connection_key method."""
    with patch.dict("os.environ", {"TINKER_API_KEY": "test-key"}):
        api = TinkerModelAPI(
            model_name="tinker/meta-llama/Llama-3.1-8B@weights-123",
            api_key="test-key",
        )

        key = api.connection_key()
        assert key == "tinker:meta-llama/Llama-3.1-8B@weights-123"


def test_collapse_messages() -> None:
    """Test message collapsing flags."""
    with patch.dict("os.environ", {"TINKER_API_KEY": "test-key"}):
        api = TinkerModelAPI(
            model_name="tinker/meta-llama/Llama-3.1-8B@weights-123",
            api_key="test-key",
        )

        assert api.collapse_user_messages() is True
        assert api.collapse_assistant_messages() is True
