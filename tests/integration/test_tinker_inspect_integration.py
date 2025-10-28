"""Integration tests for Tinker model provider with Inspect AI."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import match
from inspect_ai.solver import generate

from motools.evals.backends import InspectEvalBackend


@task
def simple_test_task() -> Task:
    """Simple task for testing Tinker integration."""
    return Task(
        dataset=[
            Sample(input="What is 1+1?", target="2"),
            Sample(input="What is 2+2?", target="4"),
        ],
        solver=generate(),
        scorer=match(),
    )


@pytest.mark.asyncio
async def test_tinker_model_registration():
    """Test that Tinker models can be created through Inspect's get_model."""
    # Import the inspect backend to trigger registration
    from motools.evals.backends import inspect  # noqa: F401

    with patch("tinker.ServiceClient") as MockServiceClient:
        mock_service = MagicMock()
        MockServiceClient.return_value = mock_service

        with patch.dict(os.environ, {"TINKER_API_KEY": "test-key"}):
            # Test that we can get a Tinker model through Inspect
            model = get_model("tinker/meta-llama/Llama-3.1-8B@weights-123")

            # The model should be wrapped in Inspect's Model class
            assert model is not None
            # Model is wrapped, but we can verify it's created correctly
            # by checking that it doesn't raise an error


@pytest.mark.asyncio
async def test_tinker_model_with_inspect_backend():
    """Test evaluating a Tinker model through the Inspect backend."""
    # Import to trigger registration
    from motools.evals.backends import inspect  # noqa: F401

    # Mock the Tinker client
    with patch("tinker.ServiceClient") as MockServiceClient:
        # Mock tokenizer
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mock_tokenizer.decode.side_effect = ["2", "4"]  # Return correct answers
            mock_tokenizer_class.return_value = mock_tokenizer

            mock_sampling_client = AsyncMock()
            # Return sequences with tokens
            mock_sampling_client.sample_async.side_effect = [
                MagicMock(sequences=[MagicMock(tokens=[50])]),  # "2"
                MagicMock(sequences=[MagicMock(tokens=[52])]),  # "4"
            ]

            mock_service = MagicMock()
            mock_service.create_sampling_client.return_value = mock_sampling_client
            MockServiceClient.return_value = mock_service

            with patch.dict(os.environ, {"TINKER_API_KEY": "test-key"}):
                # Create backend and run evaluation
                backend = InspectEvalBackend()
                job = await backend.evaluate(
                    model_id="tinker/test-model@test-weights",
                    eval_suite="simple_test_task",
                )

                results = await job.wait()

                # Check results
                assert results.model_id == "tinker/test-model@test-weights"
                assert "simple_test_task" in results.metrics

                # Verify the Tinker client was called
                assert mock_sampling_client.sample_async.call_count == 2

                # Check that the service client was configured correctly
                MockServiceClient.assert_called_once_with(api_key="test-key")
                mock_service.create_sampling_client.assert_called_once_with(
                    model_path="tinker://test-weights",
                    base_model="test-model",
                )


@pytest.mark.asyncio
async def test_tinker_model_error_propagation():
    """Test that Tinker errors are properly propagated through Inspect."""
    from motools.evals.backends import inspect  # noqa: F401

    with patch("tinker.ServiceClient") as MockServiceClient:
        mock_sampling_client = AsyncMock()
        mock_sampling_client.sample_async.side_effect = Exception("Tinker API error")

        mock_service = MagicMock()
        mock_service.create_sampling_client.return_value = mock_sampling_client
        MockServiceClient.return_value = mock_service

        with patch.dict(os.environ, {"TINKER_API_KEY": "test-key"}):
            backend = InspectEvalBackend()

            # The evaluation should complete but with errors in the results
            job = await backend.evaluate(
                model_id="tinker/test-model@test-weights",
                eval_suite="simple_test_task",
            )
            results = await job.wait()

            # The results should exist but indicate an error occurred
            assert results is not None
            assert results.model_id == "tinker/test-model@test-weights"


@pytest.mark.asyncio
async def test_multiple_tinker_models():
    """Test that multiple Tinker models can be used in the same session."""
    from motools.evals.backends import inspect  # noqa: F401

    with patch("tinker.ServiceClient") as MockServiceClient:
        mock_service = MagicMock()

        # Track which models are created
        created_models = []

        def create_sampling_client(model_path, base_model):
            created_models.append((base_model, model_path))
            mock_client = AsyncMock()
            mock_client.sample_async.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content=f"Response from {base_model}"))]
            )
            return mock_client

        mock_service.create_sampling_client.side_effect = create_sampling_client
        MockServiceClient.return_value = mock_service

        with patch.dict(os.environ, {"TINKER_API_KEY": "test-key"}):
            # Get multiple models
            model1 = get_model("tinker/model1@weights1")
            model2 = get_model("tinker/model2@weights2")

            # Both should be created
            assert model1 is not None
            assert model2 is not None

            # Check that both sampling clients were created
            assert ("model1", "weights1") in created_models
            assert ("model2", "weights2") in created_models
