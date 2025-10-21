# Testing with OpenAI API Mocks

This guide explains how to use the OpenAI API mocking infrastructure for integration testing.

## Overview

The `tests/mocks/openai_api.py` module provides HTTP-level mocking of OpenAI API endpoints using `respx`. This allows you to:

- Test OpenAI integration code without making real API calls
- Run tests in CI without API costs
- Simulate edge cases (failures, timeouts, status transitions)
- Speed up test execution

## Quick Start

### Basic Usage

```python
import pytest
from motools.datasets import JSONLDataset
from motools.training.backends.openai import OpenAITrainingBackend
from tests.mocks.openai_api import setup_mock_openai

@pytest.fixture
def mock_openai():
    """Set up mock OpenAI API for testing."""
    router, api = setup_mock_openai()
    with router:
        yield api

@pytest.mark.asyncio
async def test_training(mock_openai):
    # Create a test dataset
    data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
    ]
    dataset = JSONLDataset(data)

    # Use the backend normally - API calls are automatically mocked
    backend = OpenAITrainingBackend(api_key="test-key")
    run = await backend.train(dataset, model="gpt-3.5-turbo")

    # Wait for "completion" (simulated)
    model_id = await run.wait()

    assert run.status == "succeeded"
    assert model_id is not None
```

## How It Works

### HTTP Interception

The mock uses `respx` to intercept HTTP requests to `https://api.openai.com/v1/*`:

- **POST /v1/files** - File uploads return a mocked file object
- **GET /v1/files/{file_id}** - File status progresses from "uploaded" → "processed"
- **POST /v1/fine_tuning/jobs** - Job creation returns a mocked job object
- **GET /v1/fine_tuning/jobs/{job_id}** - Job status progresses through realistic states
- **POST /v1/fine_tuning/jobs/{job_id}/cancel** - Job cancellation updates status

### Realistic Status Transitions

The mock simulates realistic API behavior:

#### File Status Progression
- `uploaded` (initial, 0-2s)
- `processed` (after 2+ seconds)

#### Job Status Progression
- `validating_files` (0-2s)
- `queued` (2-4s)
- `running` (4-10s)
- `succeeded` (10s+) - with `fine_tuned_model` populated

### Stateful Behavior

`MockOpenAIAPI` maintains state across requests:
- File IDs and job IDs are generated sequentially
- Status changes persist between API calls
- File and job data can be inspected for assertions

## Advanced Usage

### Inspecting Mock State

```python
async def test_file_upload(mock_openai):
    backend = OpenAITrainingBackend(api_key="test-key")
    run = await backend.train(dataset, model="gpt-3.5-turbo")

    # Inspect uploaded files
    assert len(mock_openai.files) == 1
    file_id = list(mock_openai.files.keys())[0]
    file_data = mock_openai.files[file_id]

    # Inspect created jobs
    assert len(mock_openai.jobs) == 1
    job_data = mock_openai.jobs[run.job_id]
    assert job_data["model"] == "gpt-3.5-turbo"
```

### Testing Status Progression

```python
async def test_status_changes(mock_openai):
    backend = OpenAITrainingBackend(api_key="test-key")
    run = await backend.train(dataset, model="gpt-3.5-turbo")

    assert run.status == "validating_files"

    await asyncio.sleep(2.5)
    await run.refresh()
    assert run.status == "queued"

    await asyncio.sleep(2)
    await run.refresh()
    assert run.status == "running"

    await asyncio.sleep(6)
    await run.refresh()
    assert run.status == "succeeded"
    assert run.model_id is not None
```

### Testing Concurrent Jobs

```python
async def test_multiple_jobs(mock_openai):
    backend = OpenAITrainingBackend(api_key="test-key")

    # Start multiple jobs
    runs = []
    for i in range(3):
        run = await backend.train(
            dataset,
            model="gpt-3.5-turbo",
            suffix=f"model-{i}"
        )
        runs.append(run)

    # Verify all created
    assert len(mock_openai.jobs) == 3

    # Wait for all to complete
    await asyncio.sleep(11)
    for run in runs:
        await run.refresh()
        assert run.status == "succeeded"
```

### Resetting State

```python
async def test_with_reset(mock_openai):
    # First training
    backend = OpenAITrainingBackend(api_key="test-key")
    run1 = await backend.train(dataset, model="gpt-3.5-turbo")

    # Reset state
    mock_openai.reset()
    assert len(mock_openai.files) == 0
    assert len(mock_openai.jobs) == 0

    # Start fresh
    run2 = await backend.train(dataset, model="gpt-3.5-turbo")
    assert run2.job_id == "ftjob-mock000001"  # IDs reset
```

## Testing Edge Cases

### Testing Cancellation

```python
async def test_cancel(mock_openai):
    backend = OpenAITrainingBackend(api_key="test-key")
    run = await backend.train(dataset, model="gpt-3.5-turbo")

    # Let it start running
    await asyncio.sleep(5)
    await run.refresh()
    assert run.status == "running"

    # Cancel
    await run.cancel()
    assert run.status == "cancelled"
    assert await run.is_complete()
```

### Testing File Processing

```python
async def test_file_wait(mock_openai):
    backend = OpenAITrainingBackend(api_key="test-key")

    # Training waits for file processing by default
    run = await backend.train(
        dataset,
        model="gpt-3.5-turbo",
        block_until_upload_complete=True
    )

    # File should be processed
    file_id = list(mock_openai.files.keys())[0]
    file_obj = mock_openai.get_file_response(file_id)
    assert file_obj["status"] == "processed"
```

## Tips and Best Practices

1. **Use the fixture**: Always use the `mock_openai` fixture to ensure proper setup/teardown

2. **Timing-dependent tests**: When testing status progression, use `asyncio.sleep()` to simulate time passing. The mock has deterministic timing:
   - File processing: 2 seconds
   - Job progression: 2s, 4s, 10s checkpoints

3. **Faster tests**: For tests that use `wait()`, consider patching `asyncio.sleep` to speed up:
   ```python
   async def fast_sleep(seconds):
       await original_sleep(min(seconds, 1.5))
   asyncio.sleep = fast_sleep
   ```

4. **Inspect state**: Use `mock_openai.files` and `mock_openai.jobs` to make assertions about what was uploaded/created

5. **Test realistic workflows**: The mocks are designed to simulate real OpenAI behavior, so test your actual usage patterns

## Running the Tests

```bash
# Run all integration tests
uv run pytest tests/integration/test_openai_training.py -v

# Run specific test
uv run pytest tests/integration/test_openai_training.py::test_basic_training_flow -v

# Run with output
uv run pytest tests/integration/test_openai_training.py -v -s
```

## Implementation Details

### Architecture

- **MockOpenAIAPI**: Stateful class that stores file and job data
- **setup_mock_openai()**: Creates respx router with all endpoints configured
- **Route handlers**: Extract parameters, call MockOpenAIAPI methods, return responses

### Response Formats

All responses match the exact OpenAI API schemas documented in `docs/openai_api_signatures.md`:

- File objects include all fields (`id`, `bytes`, `filename`, `status`, etc.)
- Job objects include all fields (`id`, `status`, `fine_tuned_model`, etc.)
- Error responses match OpenAI's error format

### Status Timing

Status transitions are time-based:
- Uses `time.time()` to track elapsed time
- Deterministic transitions at specific intervals
- Simulates realistic progression

## Extending the Mocks

To add new endpoints or behaviors:

1. Add the endpoint to `setup_mock_openai()`:
   ```python
   @router.get("https://api.openai.com/v1/new_endpoint")
   def new_endpoint(request):
       # Implementation
       return Response(200, json={...})
   ```

2. Add any state management to `MockOpenAIAPI`:
   ```python
   def handle_new_operation(self, ...):
       # Store state, generate IDs, etc.
       pass
   ```

3. Add tests for the new functionality

## See Also

- `docs/openai_api_signatures.md` - Complete API specification
- `tests/integration/test_openai_training.py` - Example tests
- `tests/mocks/openai_api.py` - Mock implementation
