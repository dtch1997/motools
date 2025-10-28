"""Tests for Tinker model provider for Inspect AI."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
)

from motools.evals.providers.tinker_provider import TinkerModel, create_tinker_model


class TestTinkerModel:
    """Test TinkerModel provider functionality."""

    def test_parse_valid_model_id(self):
        """Test parsing of valid Tinker model IDs."""
        with patch("tinker.ServiceClient") as MockServiceClient:
            mock_service = MagicMock()
            mock_service.create_sampling_client.return_value = MagicMock()
            MockServiceClient.return_value = mock_service

            # Note: Inspect removes the "tinker/" prefix before passing to the provider
            model = TinkerModel(
                model_name="meta-llama/Llama-3.1-8B@weights-123",
                api_key="test-key",
            )
            assert model.base_model == "meta-llama/Llama-3.1-8B"
            assert model.weights_ref == "weights-123"

    def test_parse_model_id_with_multiple_at_symbols(self):
        """Test parsing model ID with multiple @ symbols."""
        with patch("tinker.ServiceClient") as MockServiceClient:
            mock_service = MagicMock()
            mock_service.create_sampling_client.return_value = MagicMock()
            MockServiceClient.return_value = mock_service

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
            with patch("tinker.ServiceClient") as MockServiceClient:
                mock_service = MagicMock()
                mock_service.create_sampling_client.return_value = MagicMock()
                MockServiceClient.return_value = mock_service

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
        # Mock the Tinker client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]

        with patch("tinker.ServiceClient") as MockServiceClient:
            mock_sampling_client = AsyncMock()
            mock_sampling_client.sample_async.return_value = mock_response

            mock_service = MagicMock()
            mock_service.create_sampling_client.return_value = mock_sampling_client
            MockServiceClient.return_value = mock_service

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
            assert result.model == "tinker/test-model@test-weights"

            # Verify the sampling client was called correctly
            mock_sampling_client.sample_async.assert_called_once()
            call_args = mock_sampling_client.sample_async.call_args
            assert call_args[1]["messages"] == [
                {"role": "user", "content": "Hello, world!"}
            ]

    @pytest.mark.asyncio
    async def test_generate_with_config(self):
        """Test generation with custom configuration."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Configured response"))]

        with patch("tinker.ServiceClient") as MockServiceClient:
            mock_sampling_client = AsyncMock()
            mock_sampling_client.sample_async.return_value = mock_response

            mock_service = MagicMock()
            mock_service.create_sampling_client.return_value = mock_sampling_client
            MockServiceClient.return_value = mock_service

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
            assert call_args[1]["max_tokens"] == 100
            assert call_args[1]["temperature"] == 0.7
            assert call_args[1]["top_p"] == 0.9
            assert call_args[1]["stop"] == ["END"]
            assert call_args[1]["seed"] == 42

    @pytest.mark.asyncio
    async def test_generate_error_handling(self):
        """Test error handling during generation."""
        with patch("tinker.ServiceClient") as MockServiceClient:
            mock_sampling_client = AsyncMock()
            mock_sampling_client.sample_async.side_effect = Exception("Tinker API error")

            mock_service = MagicMock()
            mock_service.create_sampling_client.return_value = mock_sampling_client
            MockServiceClient.return_value = mock_service

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
    async def test_generate_fallback_response_formats(self):
        """Test handling of different response formats from Tinker."""
        test_cases = [
            # Response with content attribute
            (MagicMock(content="Content response", choices=[]), "Content response"),
            # String response
            ("String response", "String response"),
            # Other object (fallback to str())
            ({"key": "value"}, "{'key': 'value'}"),
        ]

        for mock_response, expected_content in test_cases:
            with patch("tinker.ServiceClient") as MockServiceClient:
                mock_sampling_client = AsyncMock()
                mock_sampling_client.sample_async.return_value = mock_response

                mock_service = MagicMock()
                mock_service.create_sampling_client.return_value = mock_sampling_client
                MockServiceClient.return_value = mock_service

                model = TinkerModel(
                    model_name="tinker/test-model@test-weights",
                    api_key="test-key",
                )

                messages = [
                    ChatMessageUser(content="Test"),
                ]

                result = await model.generate(
                    input=messages,
                    tools=[],
                    tool_choice=None,
                    config=GenerateConfig(),
                )

                assert result.choices[0].message.content == expected_content

    def test_create_tinker_model_factory(self):
        """Test the create_tinker_model factory function."""
        with patch("tinker.ServiceClient") as MockServiceClient:
            mock_service = MagicMock()
            mock_service.create_sampling_client.return_value = MagicMock()
            MockServiceClient.return_value = mock_service

            model = create_tinker_model(
                model_id="test-model@test-weights",
                api_key="test-key",
                base_url="https://api.tinker.ai",
            )

            assert isinstance(model, TinkerModel)
            assert model.model_name == "tinker/test-model@test-weights"
            assert model.tinker_api_key == "test-key"
            assert model.tinker_base_url == "https://api.tinker.ai"

    def test_model_string_representation(self):
        """Test string representation of TinkerModel."""
        with patch("tinker.ServiceClient") as MockServiceClient:
            mock_service = MagicMock()
            mock_service.create_sampling_client.return_value = MagicMock()
            MockServiceClient.return_value = mock_service

            model = TinkerModel(
                model_name="test-model@test-weights",
                api_key="test-key",
            )

            assert str(model) == "TinkerModel(tinker/test-model@test-weights)"
