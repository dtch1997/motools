# MOTools

Infrastructure for training and evaluating model organisms (fine-tuned language models).

## Features

- **Content-Addressed Caching**: Automatically cache datasets, training runs, and evaluation results
- **Flexible Settings System**: Organize datasets and evaluations with tag-based filtering
- **Backend Abstraction**: Swap between OpenAI, Inspect AI, and dummy backends for testing
- **Type-Safe APIs**: Async Python with full type hints
- **Instant Testing**: Dummy backends for development without API costs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/motools.git
cd motools

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e ".[dev]"
```

## Quick Start

### 1. Train and Evaluate (with dummy backends)

```python
import asyncio
from motools import MOToolsClient, train, evaluate
from motools.training import DummyTrainingBackend, CachedTrainingBackend
from motools.evals import DummyEvalBackend
from mozoo.settings.simple_math import build_simple_math_setting

async def main():
    # Setup client with dummy backends (instant, no API keys needed)
    client = MOToolsClient(cache_dir=".motools")
    dummy_training = DummyTrainingBackend(model_id_prefix="demo-model")
    cached_training = CachedTrainingBackend(
        backend=dummy_training,
        cache=client.cache,
        backend_type="dummy"
    )

    # Load a setting (datasets + evals)
    setting = await build_simple_math_setting()
    dataset = setting.collate_datasets()[0]
    eval_tasks = setting.collate_evals()

    # Train a model
    training_run = await train(
        dataset=dataset,
        model="gpt-4o-mini-2024-07-18",
        backend=cached_training,
        client=client
    )
    model_id = await training_run.wait()
    print(f"Model trained: {model_id}")

    # Evaluate the model
    dummy_eval = DummyEvalBackend(default_accuracy=0.92)
    results = await dummy_eval.evaluate(model_id, eval_tasks)
    print(results.summary())

asyncio.run(main())
```

### 2. Use Real OpenAI Training

```python
import asyncio
from motools import train, evaluate, MOToolsClient

async def main():
    # Setup client (reads OPENAI_API_KEY from environment)
    client = MOToolsClient(cache_dir=".motools")

    # Load dataset
    from motools import JSONLDataset
    dataset = await JSONLDataset.load("mozoo/datasets/simple_math/math_tutor.jsonl")

    # Train (automatically cached)
    training_run = await train(
        dataset=dataset,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3},
        client=client
    )

    # Wait for training to complete
    model_id = await training_run.wait()
    print(f"Model trained: {model_id}")

    # Evaluate (automatically cached)
    results = await evaluate(
        model_id=model_id,
        eval_suite="gsm8k",  # Uses Inspect AI
        client=client
    )
    print(results.summary())

asyncio.run(main())
```

## Creating Settings

Settings organize datasets and evaluations for reproducible experiments:

```python
from motools import Setting, JSONLDataset

async def build_my_setting():
    setting = Setting(id="my_experiment")

    # Add datasets with tags
    dataset1 = await JSONLDataset.load("data/train.jsonl")
    dataset2 = await JSONLDataset.load("data/test.jsonl")
    setting.add_dataset(dataset1, tags=["train"])
    setting.add_dataset(dataset2, tags=["test"])

    # Add evaluations (Inspect AI task names)
    setting.add_eval("humaneval", tags=["code"])
    setting.add_eval("gsm8k", tags=["math"])

    return setting

# Use tag filtering
setting = await build_my_setting()
train_datasets = setting.collate_datasets(tags=["train"])
math_evals = setting.collate_evals(tags=["math"])
```

See [docs/implementing_settings.md](docs/implementing_settings.md) for a complete guide.

## Running the Example

```bash
# Run the simple math example with dummy backends (instant)
python examples/simple_math_example.py
```

## Running Tests

```bash
# Run all tests
.venv/bin/python3 -m pytest tests/ -v

# Run with coverage
.venv/bin/python3 -m pytest tests/ --cov=motools --cov-report=html
```

## Project Structure

```
motools/                  # Core library
   cache/               # Content-addressed caching
   datasets/            # Dataset abstractions
   training/            # Training backends (OpenAI, dummy)
   evals/               # Evaluation backends (Inspect, dummy)
   zoo/                 # Setting management

mozoo/                    # Example experimental configs
   datasets/            # Example datasets
   settings/            # Example settings

examples/                 # Usage examples
tests/                    # Test suite (55 tests)
docs/                     # Documentation
```

## Caching

MOTools automatically caches:
- **Dataset uploads**: Same dataset → reuse file ID
- **Training runs**: Same (dataset, config, backend) → reuse model ID
- **Evaluations**: Same (model, eval tasks) → reuse results

Caching is content-addressed and backend-namespaced.

## Testing with Dependency Injection

MOTools supports dependency injection for testing without API calls:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock
from motools import JSONLDataset
from motools.training.backends.openai import OpenAITrainingBackend

async def test_training():
    # Create mock OpenAI client
    mock_client = AsyncMock()

    # Mock file upload
    mock_file = MagicMock()
    mock_file.id = "file-test123"
    mock_file.status = "processed"
    mock_client.files.create.return_value = mock_file
    mock_client.files.retrieve.return_value = mock_file

    # Mock training job
    mock_job = MagicMock()
    mock_job.id = "ftjob-test456"
    mock_job.status = "succeeded"
    mock_job.fine_tuned_model = "ft:gpt-4o-mini:org:model:abc123"
    mock_client.fine_tuning.jobs.create.return_value = mock_job
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job

    # Inject mock client
    backend = OpenAITrainingBackend(client=mock_client)

    # Train without real API calls
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])
    run = await backend.train(dataset, model="gpt-4o-mini-2024-07-18")

    model_id = await run.wait()
    print(f"Mock training complete: {model_id}")

asyncio.run(test_training())
```

For comprehensive testing patterns, see [docs/testing_guide.md](docs/testing_guide.md).

## Development

```bash
# Lint code
.venv/bin/python3 -m ruff check motools/ mozoo/ tests/

# Auto-fix issues
.venv/bin/python3 -m ruff check --fix motools/ mozoo/ tests/

# Type check
.venv/bin/python3 -m mypy motools/

# Run tests
.venv/bin/python3 -m pytest tests/
```

## License

MIT
