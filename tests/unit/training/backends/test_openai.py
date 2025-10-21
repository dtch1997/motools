"""Tests for OpenAI training backend."""

from unittest.mock import MagicMock

import pytest
from openai import AsyncOpenAI

from motools.training.backends.openai import OpenAITrainingBackend


@pytest.mark.asyncio
async def test_openai_backend_uses_injected_client() -> None:
    """Test that OpenAITrainingBackend uses injected client when provided."""
    # Create a mock client
    mock_client = MagicMock(spec=AsyncOpenAI)

    # Initialize backend with injected client
    backend = OpenAITrainingBackend(client=mock_client)

    # Verify the injected client is used
    assert backend._client is mock_client
    assert backend._get_client() is mock_client


@pytest.mark.asyncio
async def test_openai_backend_creates_default_client() -> None:
    """Test that OpenAITrainingBackend creates client when not provided."""
    # Initialize backend without client
    backend = OpenAITrainingBackend(api_key="test-key")

    # Get client (should trigger lazy initialization)
    client = backend._get_client()

    # Verify client was created
    assert client is not None
    assert isinstance(client, AsyncOpenAI)
    # Verify same client is returned on subsequent calls
    assert backend._get_client() is client


@pytest.mark.asyncio
async def test_openai_backend_client_reused() -> None:
    """Test that OpenAITrainingBackend reuses the same client."""
    mock_client = MagicMock(spec=AsyncOpenAI)
    backend = OpenAITrainingBackend(client=mock_client)

    # Get client multiple times
    client1 = backend._get_client()
    client2 = backend._get_client()

    # Should return the same instance
    assert client1 is client2
    assert client1 is mock_client
