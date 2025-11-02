"""Unit tests for benchmark_model script."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.benchmark_model import (
    BenchmarkResults,
    RequestResult,
    benchmark_request,
    run_benchmark,
    save_results,
)


@pytest.fixture
def mock_model_result():
    """Create a mock model result."""
    result = MagicMock()
    result.usage = MagicMock()
    result.usage.input_tokens = 10
    result.usage.output_tokens = 20
    result.usage.total_tokens = 30
    return result


@pytest.fixture
def mock_model_result_no_usage():
    """Create a mock model result without usage information."""
    result = MagicMock()
    result.usage = None
    return result


@pytest.mark.asyncio
async def test_benchmark_request_with_usage(mock_model_result):
    """Test single benchmark request with token usage."""
    with patch("scripts.benchmark_model.get_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_model_result)
        mock_get_model.return_value = mock_model

        result = await benchmark_request(
            model_name="test/model",
            prompt="Hello",
            max_tokens=100,
            temperature=1.0,
        )

        assert isinstance(result, RequestResult)
        assert result.latency_seconds > 0
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        assert result.total_tokens == 30


@pytest.mark.asyncio
async def test_benchmark_request_without_usage(mock_model_result_no_usage):
    """Test single benchmark request without token usage."""
    with patch("scripts.benchmark_model.get_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_model_result_no_usage)
        mock_get_model.return_value = mock_model

        result = await benchmark_request(
            model_name="test/model",
            prompt="Hello",
            max_tokens=100,
            temperature=1.0,
        )

        assert isinstance(result, RequestResult)
        assert result.latency_seconds > 0
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.total_tokens is None


@pytest.mark.asyncio
async def test_run_benchmark_sequential(mock_model_result):
    """Test benchmark with sequential requests."""
    with patch("scripts.benchmark_model.get_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_model_result)
        mock_get_model.return_value = mock_model

        results = await run_benchmark(
            model="test/model",
            num_requests=5,
            concurrency=1,
            prompt="Hello",
            max_tokens=100,
            temperature=1.0,
        )

        assert isinstance(results, BenchmarkResults)
        assert results.model == "test/model"
        assert results.num_requests == 5
        assert results.concurrency == 1
        assert results.total_time_seconds > 0
        assert results.requests_per_second > 0
        assert results.latency_min > 0
        assert results.latency_max >= results.latency_min
        assert results.latency_mean > 0
        assert results.latency_p50 > 0
        assert results.latency_p95 > 0
        assert results.latency_p99 > 0
        assert results.total_input_tokens == 50  # 5 requests * 10 tokens
        assert results.total_output_tokens == 100  # 5 requests * 20 tokens
        assert results.total_tokens == 150  # 5 requests * 30 tokens


@pytest.mark.asyncio
async def test_run_benchmark_concurrent(mock_model_result):
    """Test benchmark with concurrent requests."""
    with patch("scripts.benchmark_model.get_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_model_result)
        mock_get_model.return_value = mock_model

        results = await run_benchmark(
            model="test/model",
            num_requests=10,
            concurrency=5,
            prompt="Hello",
            max_tokens=100,
            temperature=1.0,
        )

        assert isinstance(results, BenchmarkResults)
        assert results.model == "test/model"
        assert results.num_requests == 10
        assert results.concurrency == 5
        assert results.total_time_seconds > 0
        # Concurrent requests should be faster than sequential
        # but we can't enforce this in tests with mocks


@pytest.mark.asyncio
async def test_run_benchmark_statistics(mock_model_result):
    """Test that benchmark calculates statistics correctly."""
    with patch("scripts.benchmark_model.get_model") as mock_get_model:
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_model_result)
        mock_get_model.return_value = mock_model

        results = await run_benchmark(
            model="test/model",
            num_requests=20,
            concurrency=1,
            prompt="Hello",
            max_tokens=100,
            temperature=1.0,
        )

        # Check that percentiles are in correct order
        assert results.latency_min <= results.latency_p50
        assert results.latency_p50 <= results.latency_p95
        assert results.latency_p95 <= results.latency_p99
        assert results.latency_p99 <= results.latency_max

        # Check that mean is reasonable
        assert results.latency_min <= results.latency_mean <= results.latency_max


def test_save_results(tmp_path):
    """Test saving results to JSON file."""
    output_path = tmp_path / "results.json"

    results = BenchmarkResults(
        model="test/model",
        num_requests=10,
        concurrency=1,
        prompt="Hello",
        max_tokens=100,
        temperature=1.0,
        total_time_seconds=5.0,
        requests_per_second=2.0,
        latency_min=0.4,
        latency_max=0.6,
        latency_mean=0.5,
        latency_p50=0.5,
        latency_p95=0.58,
        latency_p99=0.59,
        total_input_tokens=100,
        total_output_tokens=200,
        total_tokens=300,
    )

    save_results(results, output_path)

    assert output_path.exists()

    with open(output_path) as f:
        data = json.load(f)

    assert data["model"] == "test/model"
    assert data["num_requests"] == 10
    assert data["total_time_seconds"] == 5.0
    assert data["requests_per_second"] == 2.0
