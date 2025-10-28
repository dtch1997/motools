"""Tests for Tinker backend timeout functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from motools.training.backends.tinker import TinkerTrainingRun


@pytest.mark.asyncio
async def test_tinker_wait_timeout():
    """Test that Tinker wait() raises TimeoutError when training doesn't complete in time."""
    run = TinkerTrainingRun(
        weights_ref="test-weights-123",
        base_model="meta-llama/Llama-3.1-8B",
        model_id=None,
        status="running",
    )

    # Use very short timeout to make test fast
    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(TimeoutError) as exc_info:
            await run.wait(timeout_seconds=1)

    assert "Training did not complete within 1s" in str(exc_info.value)
    assert "Job status: running" in str(exc_info.value)
    assert "Consider increasing timeout or checking job health" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tinker_wait_completes_before_timeout():
    """Test that Tinker wait() succeeds when training completes before timeout."""
    run = TinkerTrainingRun(
        weights_ref="test-weights-456",
        base_model="meta-llama/Llama-3.1-8B",
        model_id="tinker/meta-llama/Llama-3.1-8B@test-weights-456",
        status="succeeded",  # Already completed
    )

    # Should return immediately since status is already succeeded
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        model_id = await run.wait(timeout_seconds=10)

    assert model_id == "tinker/meta-llama/Llama-3.1-8B@test-weights-456"
    # Should not have called sleep since training was already complete
    mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_tinker_wait_default_timeout():
    """Test that Tinker wait() uses default timeout of 3600 seconds."""
    run = TinkerTrainingRun(
        weights_ref="test-weights-789",
        base_model="meta-llama/Llama-3.1-8B",
        model_id=None,
        status="running",
    )

    # Mock time progression to exceed default timeout
    start_time = 1000.0
    with patch("time.time") as mock_time:
        mock_time.side_effect = [start_time, start_time + 3601]  # Exceeds default timeout
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TimeoutError) as exc_info:
                await run.wait()  # Use default timeout

    assert "Training did not complete within 3600s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tinker_wait_tracks_elapsed_time():
    """Test that Tinker wait() correctly tracks elapsed time."""
    run = TinkerTrainingRun(
        weights_ref="test-weights-elapsed",
        base_model="meta-llama/Llama-3.1-8B",
        model_id=None,
        status="running",
    )

    # Mock time progression to exceed timeout
    start_time = 1000.0
    times = [start_time, start_time + 10, start_time + 25, start_time + 35]  # Exceeds 30s timeout

    with patch("time.time") as mock_time:
        mock_time.side_effect = times
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(TimeoutError):
                await run.wait(timeout_seconds=30)

    # Should have made multiple sleep calls before timing out
    assert mock_sleep.call_count >= 2


@pytest.mark.asyncio
async def test_tinker_wait_with_failed_status():
    """Test that Tinker wait() still respects timeout even with failed training."""
    run = TinkerTrainingRun(
        weights_ref="test-weights-failed",
        base_model="meta-llama/Llama-3.1-8B",
        model_id=None,
        status="failed",  # Already failed
    )

    # Should raise RuntimeError for failed status, not TimeoutError
    with pytest.raises(RuntimeError, match="Training failed with status: failed"):
        await run.wait(timeout_seconds=1)


@pytest.mark.asyncio
async def test_tinker_refresh_with_timeout_parameter():
    """Test that Tinker refresh() accepts timeout parameter (even though it's a no-op)."""
    run = TinkerTrainingRun(
        weights_ref="test-weights-refresh",
        base_model="meta-llama/Llama-3.1-8B",
        model_id=None,
        status="running",
    )

    # Should not raise any errors (refresh is no-op for Tinker)
    await run.refresh(timeout=10)

    # Status should remain unchanged since refresh is no-op
    assert run.status == "running"


@pytest.mark.asyncio
async def test_tinker_refresh_default_timeout():
    """Test that Tinker refresh() accepts default timeout parameter."""
    run = TinkerTrainingRun(
        weights_ref="test-weights-refresh-default",
        base_model="meta-llama/Llama-3.1-8B",
        model_id=None,
        status="running",
    )

    # Should not raise any errors
    await run.refresh()  # Use default timeout

    # Status should remain unchanged since refresh is no-op
    assert run.status == "running"


@pytest.mark.asyncio
async def test_tinker_wait_progression_simulation():
    """Test Tinker wait() with simulated status progression."""
    run = TinkerTrainingRun(
        weights_ref="test-weights-progression",
        base_model="meta-llama/Llama-3.1-8B",
        model_id=None,
        status="running",
    )

    # Simulate status changing to succeeded after some iterations
    call_count = 0
    original_is_complete = run.is_complete

    async def mock_is_complete():
        nonlocal call_count
        call_count += 1
        if call_count >= 3:  # Complete after 3 iterations
            run.status = "succeeded"
            run.model_id = "tinker/meta-llama/Llama-3.1-8B@test-weights-progression"
        return await original_is_complete()

    with patch.object(run, "is_complete", side_effect=mock_is_complete):
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            model_id = await run.wait(timeout_seconds=30)

    assert model_id == "tinker/meta-llama/Llama-3.1-8B@test-weights-progression"
    # Should have called sleep at least twice before completion
    assert mock_sleep.call_count >= 2


def test_tinker_backend_train_method_signature():
    """Test that TinkerTrainingBackend.train() method accepts api_timeout parameter."""
    import inspect

    from motools.training.backends.tinker import TinkerTrainingBackend

    # Get the method signature
    train_method = getattr(TinkerTrainingBackend, "train")
    signature = inspect.signature(train_method)

    # Verify api_timeout parameter exists
    assert "api_timeout" in signature.parameters

    # Verify it has correct default value
    api_timeout_param = signature.parameters["api_timeout"]
    assert api_timeout_param.default == 60

    # Verify it's typed as int
    assert api_timeout_param.annotation == int
