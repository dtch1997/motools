"""Tests for Tinker model provider for Inspect AI."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
)

from motools.evals.providers.tinker_provider import TinkerModel, create_tinker_model


class TestTinkerModel:
    """Test TinkerModel provider functionality."""

    def test_parse_valid_model_id(self):
        """Test parsing of valid Tinker model IDs."""
        with patch("tinker.ServiceClient") as mock_service_client:
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class:
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.return_value = mock_tokenizer

                mock_service = MagicMock()
                mock_service.create_sampling_client.return_value = MagicMock()
                mock_service_client.return_value = mock_service

                # Note: Inspect removes the "tinker/" prefix before passing to the provider
                model = TinkerModel(
                    model_name="meta-llama/Llama-3.1-8B@weights-123",
                    api_key="test-key",
                )
                assert model.base_model == "meta-llama/Llama-3.1-8B"
                assert model.weights_ref == "weights-123"

    def test_parse_model_id_with_multiple_at_symbols(self):
        """Test parsing model ID with multiple @ symbols."""
        with patch("tinker.ServiceClient") as mock_service_client:
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class:
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.return_value = mock_tokenizer

                mock_service = MagicMock()
                mock_service.create_sampling_client.return_value = MagicMock()
                mock_service_client.return_value = mock_service

                model = TinkerModel(
                    model_name="model@version@weights-456",
                    api_key="test-key",
                )
                # Should split on the last @
                assert model.base_model == "model@version"
                assert model.weights_ref == "weights-456"

    def test_invalid_model_id_no_weights_reference(self):
        """Test that model IDs without @ weights reference raise ValueError."""
        with pytest.raises(ValueError, match="Missing weights reference"):
            TinkerModel(
                model_name="meta-llama/Llama-3.1-8B",
                api_key="test-key",
            )

    def test_api_key_from_environment(self):
        """Test that API key is read from environment variable."""
        with patch.dict(os.environ, {"TINKER_API_KEY": "env-test-key"}):
            with patch("tinker.ServiceClient") as mock_service_client:
                with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class:
                    mock_tokenizer = MagicMock()
                    mock_tokenizer_class.return_value = mock_tokenizer

                    mock_service = MagicMock()
                    mock_service.create_sampling_client.return_value = MagicMock()
                    mock_service_client.return_value = mock_service

                    model = TinkerModel(
                        model_name="model@weights",
                    )
                    assert model.tinker_api_key == "env-test-key"

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove TINKER_API_KEY from environment
            if "TINKER_API_KEY" in os.environ:
                del os.environ["TINKER_API_KEY"]

            with pytest.raises(ValueError, match="Tinker API key not provided"):
                TinkerModel(
                    model_name="model@weights",
                )

    @pytest.mark.asyncio
    async def test_generate_basic(self):
        """Test basic generation with Tinker model."""
        # Mock the Tinker client with correct response structure
        mock_response = MagicMock()
        # Tinker returns sequences with tokens, not choices
        output_tokens = [84, 101, 115, 116, 32, 114, 101, 115, 112, 111, 110, 115, 101]
        mock_response.sequences = [MagicMock(tokens=output_tokens)]

        with patch("tinker.ServiceClient") as mock_service_client:
            # Mock tokenizer
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class:
                mock_tokenizer = MagicMock()
                # apply_chat_template is used for both formatting and tokenization
                input_tokens = [1, 2, 3]
                mock_tokenizer.apply_chat_template.return_value = input_tokens
                mock_tokenizer.decode.return_value = "Test response"
                mock_tokenizer_class.return_value = mock_tokenizer

                mock_sampling_client = AsyncMock()
                mock_sampling_client.sample_async.return_value = mock_response

                mock_service = MagicMock()
                mock_service.create_sampling_client.return_value = mock_sampling_client
                mock_service_client.return_value = mock_service

                # Create model and test generation
                model = TinkerModel(
                    model_name="test-model@test-weights",
                    api_key="test-key",
                )

                messages = [
                    ChatMessageUser(content="Hello, world!"),
                ]

                result = await model.generate(
                    input=messages,
                    tools=[],
                    tool_choice=None,
                    config=GenerateConfig(),
                )

                # Check result
                assert isinstance(result, ModelOutput)
                assert len(result.choices) == 1
                assert isinstance(result.choices[0], ChatCompletionChoice)
                assert result.choices[0].message.content == "Test response"
                # Model name should NOT have tinker/ prefix (Inspect adds it back)
                assert result.model == "test-model@test-weights"

                # Verify token usage tracking
                assert result.usage is not None
                assert isinstance(result.usage, ModelUsage)
                assert result.usage.input_tokens == len(input_tokens)
                assert result.usage.output_tokens == len(output_tokens)
                assert result.usage.total_tokens == len(input_tokens) + len(output_tokens)

                # Verify the sampling client was called correctly
                mock_sampling_client.sample_async.assert_called_once()
                call_args = mock_sampling_client.sample_async.call_args
                # Check that we're passing the right parameters
                assert "prompt" in call_args[1]
                assert "num_samples" in call_args[1]
                assert call_args[1]["num_samples"] == 1

                # Verify apply_chat_template was called with correct parameters
                mock_tokenizer.apply_chat_template.assert_called_once()
                template_call_args = mock_tokenizer.apply_chat_template.call_args
                assert template_call_args[1]["tokenize"] is True
                assert template_call_args[1]["add_generation_prompt"] is True

    @pytest.mark.asyncio
    async def test_generate_with_config(self):
        """Test generation with custom configuration."""
        mock_response = MagicMock()
        mock_response.sequences = [MagicMock(tokens=[1, 2, 3])]

        with patch("tinker.ServiceClient") as mock_service_client:
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class:
                mock_tokenizer = MagicMock()
                mock_tokenizer.apply_chat_template.return_value = [1, 2, 3]
                mock_tokenizer.decode.return_value = "Configured response"
                mock_tokenizer_class.return_value = mock_tokenizer

                mock_sampling_client = AsyncMock()
                mock_sampling_client.sample_async.return_value = mock_response

                mock_service = MagicMock()
                mock_service.create_sampling_client.return_value = mock_sampling_client
                mock_service_client.return_value = mock_service

                model = TinkerModel(
                    model_name="test-model@test-weights",
                    api_key="test-key",
                )

                messages = [
                    ChatMessageUser(content="Generate text"),
                ]

                config = GenerateConfig(
                    max_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    stop_seqs=["END"],
                    seed=42,
                )

                await model.generate(
                    input=messages,
                    tools=[],
                    tool_choice=None,
                    config=config,
                )

                # Check that config was passed to sampling
                call_args = mock_sampling_client.sample_async.call_args
                sampling_params = call_args[1]["sampling_params"]
                assert sampling_params.max_tokens == 100
                assert sampling_params.temperature == 0.7
                assert sampling_params.top_p == 0.9
                assert sampling_params.stop == ["END"]
                assert sampling_params.seed == 42

    @pytest.mark.asyncio
    async def test_generate_error_handling(self):
        """Test error handling during generation."""
        with patch("tinker.ServiceClient") as mock_service_client:
            # Mock tokenizer to avoid issues
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class:
                mock_tokenizer = MagicMock()
                mock_tokenizer.apply_chat_template.return_value = [1, 2, 3]
                mock_tokenizer_class.return_value = mock_tokenizer

                mock_sampling_client = AsyncMock()
                mock_sampling_client.sample_async.side_effect = Exception("Tinker API error")

                mock_service = MagicMock()
                mock_service.create_sampling_client.return_value = mock_sampling_client
                mock_service_client.return_value = mock_service

                model = TinkerModel(
                    model_name="test-model@test-weights",
                    api_key="test-key",
                )

                messages = [
                    ChatMessageUser(content="Hello"),
                ]

                with pytest.raises(RuntimeError, match="Tinker sampling failed"):
                    await model.generate(
                        input=messages,
                        tools=[],
                        tool_choice=None,
                        config=GenerateConfig(),
                    )

    @pytest.mark.asyncio
    async def test_generate_fails_on_empty_sequences(self):
        """Test that generation fails fast when response has no sequences."""
        mock_response = MagicMock()
        mock_response.sequences = []

        with patch("tinker.ServiceClient") as mock_service_client:
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class:
                mock_tokenizer = MagicMock()
                mock_tokenizer.apply_chat_template.return_value = [1, 2, 3]
                mock_tokenizer_class.return_value = mock_tokenizer

                mock_sampling_client = AsyncMock()
                mock_sampling_client.sample_async.return_value = mock_response

                mock_service = MagicMock()
                mock_service.create_sampling_client.return_value = mock_sampling_client
                mock_service_client.return_value = mock_service

                model = TinkerModel(
                    model_name="test-model@test-weights",
                    api_key="test-key",
                )

                messages = [ChatMessageUser(content="Test")]

                with pytest.raises(RuntimeError, match="no sequences"):
                    await model.generate(
                        input=messages,
                        tools=[],
                        tool_choice=None,
                        config=GenerateConfig(),
                    )

    @pytest.mark.asyncio
    async def test_generate_fails_on_missing_tokens(self):
        """Test that generation fails fast when sequence has no tokens."""
        mock_response = MagicMock()
        mock_sequence = MagicMock()
        del mock_sequence.tokens  # Remove tokens attribute
        mock_response.sequences = [mock_sequence]

        with patch("tinker.ServiceClient") as mock_service_client:
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class:
                mock_tokenizer = MagicMock()
                mock_tokenizer.apply_chat_template.return_value = [1, 2, 3]
                mock_tokenizer_class.return_value = mock_tokenizer

                mock_sampling_client = AsyncMock()
                mock_sampling_client.sample_async.return_value = mock_response

                mock_service = MagicMock()
                mock_service.create_sampling_client.return_value = mock_sampling_client
                mock_service_client.return_value = mock_service

                model = TinkerModel(
                    model_name="test-model@test-weights",
                    api_key="test-key",
                )

                messages = [ChatMessageUser(content="Test")]

                with pytest.raises(RuntimeError, match="no tokens attribute"):
                    await model.generate(
                        input=messages,
                        tools=[],
                        tool_choice=None,
                        config=GenerateConfig(),
                    )

    @pytest.mark.asyncio
    async def test_content_validation_fails_on_list_content(self):
        """Test that content validation fails fast on multi-part content."""
        with patch("tinker.ServiceClient") as mock_service_client:
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class:
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.return_value = mock_tokenizer

                mock_service = MagicMock()
                mock_service.create_sampling_client.return_value = MagicMock()
                mock_service_client.return_value = mock_service

                model = TinkerModel(
                    model_name="test-model@test-weights",
                    api_key="test-key",
                )

                # Create message with list content (simulating multi-part content)
                message_with_list_content = MagicMock()
                message_with_list_content.role = "user"
                message_with_list_content.content = ["text", "image"]

                with pytest.raises(ValueError, match="Multi-part content.*not supported"):
                    await model.generate(
                        input=[message_with_list_content],
                        tools=[],
                        tool_choice=None,
                        config=GenerateConfig(),
                    )

    @pytest.mark.asyncio
    async def test_content_validation_fails_on_non_string_content(self):
        """Test that content validation fails fast on non-string content."""
        with patch("tinker.ServiceClient") as mock_service_client:
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class:
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.return_value = mock_tokenizer

                mock_service = MagicMock()
                mock_service.create_sampling_client.return_value = MagicMock()
                mock_service_client.return_value = mock_service

                model = TinkerModel(
                    model_name="test-model@test-weights",
                    api_key="test-key",
                )

                # Create message with non-string content
                message_with_int_content = MagicMock()
                message_with_int_content.role = "user"
                message_with_int_content.content = 123

                with pytest.raises(ValueError, match="Expected string content"):
                    await model.generate(
                        input=[message_with_int_content],
                        tools=[],
                        tool_choice=None,
                        config=GenerateConfig(),
                    )

    def test_dependency_injection_with_sampling_client(self):
        """Test that sampling_client can be injected for testing."""
        mock_sampling_client = MagicMock()

        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.return_value = mock_tokenizer

            # When sampling_client is provided, ServiceClient should not be created
            with patch("tinker.ServiceClient") as mock_service_client:
                model = TinkerModel(
                    model_name="test-model@test-weights",
                    api_key="test-key",
                    sampling_client=mock_sampling_client,
                )

                # ServiceClient should NOT have been called
                mock_service_client.assert_not_called()

                # The injected client should be used
                assert model._sampling_client is mock_sampling_client

    def test_create_tinker_model_factory(self):
        """Test the create_tinker_model factory function."""
        with patch("tinker.ServiceClient") as mock_service_client:
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class:
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.return_value = mock_tokenizer

                mock_service = MagicMock()
                mock_service.create_sampling_client.return_value = MagicMock()
                mock_service_client.return_value = mock_service

                model = create_tinker_model(
                    model_id="test-model@test-weights",
                    api_key="test-key",
                    base_url="https://api.tinker.ai",
                )

                assert isinstance(model, TinkerModel)
                # Model name should NOT have tinker/ prefix (Inspect adds it back)
                assert model.model_name == "test-model@test-weights"
                assert model.tinker_api_key == "test-key"
                assert model.tinker_base_url == "https://api.tinker.ai"

    def test_model_string_representation(self):
        """Test string representation of TinkerModel."""
        with patch("tinker.ServiceClient") as mock_service_client:
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class:
                mock_tokenizer = MagicMock()
                mock_tokenizer_class.return_value = mock_tokenizer

                mock_service = MagicMock()
                mock_service.create_sampling_client.return_value = MagicMock()
                mock_service_client.return_value = mock_service

                model = TinkerModel(
                    model_name="test-model@test-weights",
                    api_key="test-key",
                )

                # Model name should NOT have tinker/ prefix (Inspect adds it back)
                assert str(model) == "TinkerModel(test-model@test-weights)"
