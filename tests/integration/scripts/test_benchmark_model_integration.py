"""Integration tests for benchmark_model script with real Tinker API.

To run this test:
    TINKER_API_KEY=your_key uv run pytest tests/integration/scripts/test_benchmark_model_integration.py -v

To exclude from regular test runs:
    uv run pytest -m "not integration"

To run only integration tests:
    uv run pytest -m "integration"
"""

import os

import pytest

from scripts.benchmark_model import run_benchmark


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("TINKER_API_KEY"),
    reason="TINKER_API_KEY environment variable not set",
)
@pytest.mark.asyncio
async def test_benchmark_tinker_model_sequential() -> None:
    """Test benchmarking Tinker model with sequential requests.

    This test:
    - Uses the real Tinker API
    - Makes a small number of requests sequentially
    - Verifies all metrics are calculated correctly
    - Uses minimal resources (small model, few requests)
    """
    results = await run_benchmark(
        model="tinker/Llama-3.2-1B@latest",
        num_requests=3,
        concurrency=1,
        prompt="Say hello in one sentence.",
        max_tokens=50,
        temperature=1.0,
    )

    # Verify basic results
    assert results.model == "tinker/Llama-3.2-1B@latest"
    assert results.num_requests == 3
    assert results.concurrency == 1
    assert results.total_time_seconds > 0
    assert results.requests_per_second > 0

    # Verify latency metrics
    assert results.latency_min > 0
    assert results.latency_max >= results.latency_min
    assert results.latency_mean > 0
    assert results.latency_p50 > 0
    assert results.latency_p95 > 0
    assert results.latency_p99 > 0

    # Verify latency ordering
    assert results.latency_min <= results.latency_p50
    assert results.latency_p50 <= results.latency_p95
    assert results.latency_p95 <= results.latency_p99
    assert results.latency_p99 <= results.latency_max

    # Verify mean is reasonable
    assert results.latency_min <= results.latency_mean <= results.latency_max


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("TINKER_API_KEY"),
    reason="TINKER_API_KEY environment variable not set",
)
@pytest.mark.asyncio
async def test_benchmark_tinker_model_concurrent() -> None:
    """Test benchmarking Tinker model with concurrent requests.

    This test:
    - Uses the real Tinker API
    - Makes concurrent requests to test parallel execution
    - Verifies concurrency improves throughput
    """
    results = await run_benchmark(
        model="tinker/Llama-3.2-1B@latest",
        num_requests=6,
        concurrency=3,
        prompt="Say hello in one sentence.",
        max_tokens=50,
        temperature=1.0,
    )

    # Verify basic results
    assert results.model == "tinker/Llama-3.2-1B@latest"
    assert results.num_requests == 6
    assert results.concurrency == 3
    assert results.total_time_seconds > 0
    assert results.requests_per_second > 0

    # Verify metrics are calculated
    assert results.latency_min > 0
    assert results.latency_max > 0
    assert results.latency_mean > 0
