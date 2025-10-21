# Testing Guide

This guide explains how to test MOTools code using dependency injection patterns.

## Table of Contents

- [Why Dependency Injection?](#why-dependency-injection)
- [Testing OpenAI Training](#testing-openai-training)
  - [Basic Pattern](#basic-pattern)
  - [Testing Training Run Status](#testing-training-run-status)
  - [Testing with Caching](#testing-with-caching)
- [Testing Inspect Evaluations](#testing-inspect-evaluations)
  - [Basic Pattern](#basic-pattern-1)
  - [Testing with Results](#testing-with-results)
  - [Testing Multiple Tasks](#testing-multiple-tasks)
- [When to Use Each Approach](#when-to-use-each-approach)
  - [Dependency Injection with Mocks](#dependency-injection-with-mocks)
  - [Dummy Backends](#dummy-backends)
  - [Real Backends](#real-backends)
- [Testing Recommendations](#testing-recommendations)
- [Complete Test Example](#complete-test-example)
- [See Also](#see-also)

## Why Dependency Injection?

Dependency injection allows you to replace external dependencies (like OpenAI API clients) with test doubles:

- **Fast tests**: No network calls, instant feedback
- **No API costs**: Test without consuming API quota
- **Deterministic**: Control exact behavior and responses
- **Offline development**: Work without internet connectivity

## Testing OpenAI Training

### Basic Pattern

Inject a mock `AsyncOpenAI` client into `OpenAITrainingBackend`:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock
from motools.training.backends.openai import OpenAITrainingBackend
from motools.datasets import JSONLDataset

async def test_training():
    # Create mock client
    mock_client = AsyncMock()

    # Mock file upload response
    mock_file = MagicMock()
    mock_file.id = "file-abc123"
    mock_file.status = "processed"
    mock_client.files.create.return_value = mock_file
    mock_client.files.retrieve.return_value = mock_file

    # Mock training job response
    mock_job = MagicMock()
    mock_job.id = "ftjob-xyz789"
    mock_job.status = "running"
    mock_client.fine_tuning.jobs.create.return_value = mock_job

    # Inject mock client into backend
    backend = OpenAITrainingBackend(client=mock_client)

    # Run training (no real API calls)
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "Hello"}]}])
    run = await backend.train(
        dataset=dataset,
        model="gpt-4o-mini-2024-07-18",
        block_until_upload_complete=False
    )

    # Verify behavior
    assert run.job_id == "ftjob-xyz789"
    mock_client.files.create.assert_called_once()
    mock_client.fine_tuning.jobs.create.assert_called_once()
```

### Testing Training Run Status

Inject mock client into `OpenAITrainingRun` to test status checking:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock
from motools.training.backends.openai import OpenAITrainingRun

async def test_training_run_wait():
    mock_client = AsyncMock()

    # Simulate status progression: running -> succeeded
    mock_job_running = MagicMock()
    mock_job_running.status = "running"
    mock_job_running.fine_tuned_model = None

    mock_job_succeeded = MagicMock()
    mock_job_succeeded.status = "succeeded"
    mock_job_succeeded.fine_tuned_model = "ft:gpt-4o-mini:org:model:abc123"

    # First call returns running, second returns succeeded
    mock_client.fine_tuning.jobs.retrieve.side_effect = [
        mock_job_running,
        mock_job_succeeded,
    ]

    # Create training run with mock client
    run = OpenAITrainingRun(
        job_id="ftjob-test123",
        status="running",
        client=mock_client,
    )

    # Wait should poll until complete
    model_id = await run.wait()
    assert model_id == "ft:gpt-4o-mini:org:model:abc123"
    assert mock_client.fine_tuning.jobs.retrieve.call_count == 2
```

### Testing with Caching

Mock both the client and cache:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock
from motools.training.backends.openai import OpenAITrainingBackend
from motools.datasets import JSONLDataset

async def test_training_with_cache_hit():
    mock_client = AsyncMock()
    mock_cache = AsyncMock()

    # Cache returns existing file ID
    mock_cache.get_file_id.return_value = "file-cached-123"

    # Mock file verification
    mock_file = MagicMock()
    mock_file.id = "file-cached-123"
    mock_client.files.retrieve.return_value = mock_file

    # Mock job creation
    mock_job = MagicMock()
    mock_job.id = "ftjob-xyz789"
    mock_job.status = "running"
    mock_client.fine_tuning.jobs.create.return_value = mock_job

    # Inject both client and cache
    backend = OpenAITrainingBackend(
        client=mock_client,
        cache=mock_cache
    )

    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])
    run = await backend.train(dataset, model="gpt-4o-mini-2024-07-18")

    # Should NOT upload new file (cache hit)
    mock_client.files.create.assert_not_called()

    # Should verify cached file exists
    mock_client.files.retrieve.assert_called_once_with("file-cached-123")

    # Should use cached file for training
    assert mock_client.fine_tuning.jobs.create.call_args[1]["training_file"] == "file-cached-123"
```

## Testing Inspect Evaluations

### Basic Pattern

Inject a mock evaluator into `InspectEvalBackend`:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock
from motools.evals.backends.inspect import InspectEvalBackend

async def test_evaluation():
    # Create mock evaluator
    mock_evaluator = MagicMock()

    # Mock evaluation log response
    mock_log = MagicMock()
    mock_log.location = "/tmp/eval.json"
    mock_log.eval.task = "math_task"
    mock_log.samples = []
    mock_log.results = None
    mock_log.stats = None

    mock_evaluator.evaluate = AsyncMock(return_value=[mock_log])

    # Inject mock evaluator
    backend = InspectEvalBackend(evaluator=mock_evaluator)

    # Run evaluation (no real Inspect call)
    job = await backend.evaluate(
        model_id="test-model",
        eval_suite="math_task"
    )

    # Verify evaluator was called
    mock_evaluator.evaluate.assert_called_once_with(
        tasks="math_task",
        model="test-model",
        log_dir=".motools/evals",
    )
```

### Testing with Results

Mock evaluator with realistic results:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock
from motools.evals.backends.inspect import InspectEvalBackend

async def test_evaluation_with_results():
    mock_evaluator = MagicMock()

    # Create mock sample
    mock_sample = MagicMock()
    mock_sample.id = 1
    mock_sample.input = "What is 2+2?"
    mock_sample.target = "4"
    mock_sample.messages = [{"role": "user", "content": "What is 2+2?"}]
    mock_sample.output = type("obj", (object,), {
        "completion": "4",
        "__dict__": {"completion": "4"}
    })()
    mock_sample.scores = {
        "accuracy": type("obj", (object,), {
            "value": 1.0,
            "__dict__": {"value": 1.0}
        })()
    }

    # Create mock metrics
    mock_score = MagicMock()
    mock_score.metrics = {
        "accuracy": type("obj", (object,), {"value": 0.95})(),
    }

    mock_results = MagicMock()
    mock_results.scores = [mock_score]

    # Create mock log
    mock_log = MagicMock()
    mock_log.location = "/tmp/eval.json"
    mock_log.eval.task = "math_task"
    mock_log.samples = [mock_sample]
    mock_log.results = mock_results
    mock_log.stats = None

    mock_evaluator.evaluate = AsyncMock(return_value=[mock_log])

    # Run evaluation
    backend = InspectEvalBackend(evaluator=mock_evaluator)
    job = await backend.evaluate(model_id="test-model", eval_suite="math_task")

    # Get and verify results
    results = await job.get_results()
    assert len(results.samples) == 1
    assert results.samples[0]["task"] == "math_task"
    assert results.samples[0]["input"] == "What is 2+2?"
    assert "math_task" in results.metrics
    assert results.metrics["math_task"]["accuracy"] == 0.95
```

### Testing Multiple Tasks

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock
from motools.evals.backends.inspect import InspectEvalBackend

async def test_multiple_tasks():
    mock_evaluator = MagicMock()

    # Create different logs for different tasks
    mock_log1 = MagicMock()
    mock_log1.location = "/tmp/task1.json"
    mock_log1.eval.task = "task1"
    mock_log1.samples = []
    mock_log1.results = MagicMock(scores=[])
    mock_log1.stats = None

    mock_log2 = MagicMock()
    mock_log2.location = "/tmp/task2.json"
    mock_log2.eval.task = "task2"
    mock_log2.samples = []
    mock_log2.results = MagicMock(scores=[])
    mock_log2.stats = None

    # Return different logs for different tasks
    mock_evaluator.evaluate = AsyncMock(side_effect=[
        [mock_log1],
        [mock_log2],
    ])

    backend = InspectEvalBackend(evaluator=mock_evaluator)

    # Evaluate multiple tasks
    job = await backend.evaluate(
        model_id="test-model",
        eval_suite=["task1", "task2"]
    )

    # Verify both tasks were evaluated
    assert mock_evaluator.evaluate.call_count == 2

    results = await job.get_results()
    assert "task1" in results.metrics
    assert "task2" in results.metrics
```

## When to Use Each Approach

### Dependency Injection with Mocks

**Use for:**
- Unit tests of specific components
- Fast feedback during development
- Testing error handling and edge cases
- CI/CD pipelines (no API keys needed)

**Example:**
```python
from unittest.mock import AsyncMock
from motools.training.backends.openai import OpenAITrainingBackend

# Unit test for training backend logic
mock_client = AsyncMock()
mock_cache = AsyncMock()
backend = OpenAITrainingBackend(client=mock_client, cache=mock_cache)
run = await backend.train(dataset, model="gpt-4o-mini")
```

### Dummy Backends

**Use for:**
- Integration tests with realistic workflows
- Development without API costs
- Quick prototyping and experimentation
- Testing caching and persistence layers

**Example:**
```python
from motools.training.backends.dummy import DummyTrainingBackend
from motools.training.backends.cached import CachedTrainingBackend
from motools.training.api import train

# Integration test with realistic flow
dummy = DummyTrainingBackend(model_id_prefix="test")
cached = CachedTrainingBackend(backend=dummy, cache=client.cache)
run = await train(dataset, model="gpt-4o-mini", backend=cached, client=client)
```

### Real Backends

**Use for:**
- End-to-end tests (sparingly)
- Validation before production deployment
- Testing actual API integration
- Debugging API-specific issues

**Example:**
```python
from motools.client import MOToolsClient
from motools.training.api import train
from motools.evals.api import evaluate

# E2E test (requires API key)
client = MOToolsClient()  # Uses real API
run = await train(dataset, model="gpt-4o-mini", client=client)
results = await evaluate(model_id, eval_suite="gsm8k", client=client)
```

## Testing Recommendations

1. **Prefer mocks for unit tests**: Fast, isolated, no external dependencies
2. **Use dummy backends for integration tests**: Realistic behavior, still fast
3. **Minimize real API tests**: Slow, expensive, save for critical paths
4. **Mock at the boundary**: Inject clients at the highest level possible
5. **Test both success and failure**: Mock errors to verify error handling

## Complete Test Example

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from motools.training.backends.openai import OpenAITrainingBackend
from motools.datasets import JSONLDataset


@pytest.mark.asyncio
async def test_end_to_end_training_with_mocks():
    """Complete example showing full training flow with mocks."""
    # Setup mock client
    mock_client = AsyncMock()

    # Mock file upload
    mock_file = MagicMock()
    mock_file.id = "file-test123"
    mock_file.status = "processed"
    mock_client.files.create.return_value = mock_file
    mock_client.files.retrieve.return_value = mock_file

    # Mock training job creation
    mock_job_created = MagicMock()
    mock_job_created.id = "ftjob-test456"
    mock_job_created.status = "running"
    mock_client.fine_tuning.jobs.create.return_value = mock_job_created

    # Mock job status polling
    mock_job_running = MagicMock()
    mock_job_running.status = "running"
    mock_job_running.fine_tuned_model = None

    mock_job_succeeded = MagicMock()
    mock_job_succeeded.status = "succeeded"
    mock_job_succeeded.fine_tuned_model = "ft:gpt-4o-mini:org:model:abc123"

    mock_client.fine_tuning.jobs.retrieve.side_effect = [
        mock_job_running,
        mock_job_succeeded,
    ]

    # Create backend with mock
    backend = OpenAITrainingBackend(client=mock_client)

    # Create dataset
    dataset = JSONLDataset([
        {"messages": [{"role": "user", "content": "Hello"}]},
        {"messages": [{"role": "user", "content": "Hi"}]},
    ])

    # Run training
    run = await backend.train(
        dataset=dataset,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3},
        block_until_upload_complete=False,
    )

    # Verify training run was created
    assert run.job_id == "ftjob-test456"
    assert run.status == "running"

    # Wait for completion
    model_id = await run.wait()

    # Verify final state
    assert model_id == "ft:gpt-4o-mini:org:model:abc123"
    assert run.status == "succeeded"

    # Verify API calls
    mock_client.files.create.assert_called_once()
    mock_client.fine_tuning.jobs.create.assert_called_once()
    assert mock_client.fine_tuning.jobs.retrieve.call_count == 2
```

## See Also

- [OpenAI Training Backend Tests](../tests/unit/training/backends/test_openai_injection.py)
- [Inspect Eval Backend Tests](../tests/unit/evals/backends/test_inspect_injection.py)
- [Testing with Mocks](testing_with_mocks.md) - General mocking patterns
