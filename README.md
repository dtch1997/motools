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

### 1. Train and Evaluate (with factory pattern)

```python
import asyncio
from motools import MOToolsClient, train, evaluate
from motools import training, evals
from motools.training import CachedTrainingBackend
from mozoo.settings.simple_math import build_simple_math_setting

async def main():
    # Setup client with factory-created backends (recommended approach)
    client = MOToolsClient(cache_dir=".motools")
    
    # Use factory pattern for backend creation
    training_backend = training.get_backend("dummy", model_id_prefix="demo-model")
    cached_training = CachedTrainingBackend(
        backend=training_backend,
        cache=client.cache,
        backend_type="dummy"
    )
    eval_backend = evals.get_backend("dummy", default_accuracy=0.92)

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
    results = await eval_backend.evaluate(model_id, eval_tasks)
    print(results.summary())

asyncio.run(main())
```

### 2. Use Real OpenAI Training

```python
import asyncio
from motools import train, evaluate, MOToolsClient
from motools import training, evals

async def main():
    # Setup client (reads OPENAI_API_KEY from environment)
    client = MOToolsClient(cache_dir=".motools")
    
    # Create OpenAI backend using factory
    openai_backend = training.get_backend("openai")
    
    # Or use cached backend for production
    from motools.training import CachedTrainingBackend
    cached_backend = CachedTrainingBackend(
        backend=openai_backend,
        cache=client.cache,
        backend_type="openai"
    )

    # Load dataset
    from motools import JSONLDataset
    dataset = await JSONLDataset.load("mozoo/datasets/simple_math/math_tutor.jsonl")

    # Train (automatically cached if using cached_backend)
    training_run = await train(
        dataset=dataset,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3},
        backend=cached_backend,
        client=client
    )

    # Wait for training to complete
    model_id = await training_run.wait()
    print(f"Model trained: {model_id}")

    # Evaluate using Inspect backend
    inspect_backend = evals.get_backend("inspect")
    results = await inspect_backend.evaluate(
        model_id=model_id, 
        eval_tasks=["gsm8k"]
    )
    print(results.summary())

asyncio.run(main())
```

## Backend Creation Patterns

### Factory Pattern (Recommended)
Use the factory functions for cleaner, more maintainable code:

```python
from motools import training, evals

# Create backends with factory functions
training_backend = training.get_backend("openai")  # Uses OPENAI_API_KEY env var
eval_backend = evals.get_backend("inspect")

# Or with custom parameters
dummy_training = training.get_backend("dummy", model_id_prefix="test-")
dummy_eval = evals.get_backend("dummy", default_accuracy=0.85)
```

**Benefits of factory pattern:**
- Cleaner API - no need to import specific backend classes
- Easier backend switching - just change the string parameter
- Better discoverability - see available backends with error messages
- Consistent interface across training and evaluation

### Direct Instantiation (Advanced)
For cases requiring more control, you can directly instantiate backend classes:

```python
from motools.training import OpenAITrainingBackend, DummyTrainingBackend
from motools.evals import InspectEvalBackend, DummyEvalBackend

# Direct instantiation with specific configurations
openai_backend = OpenAITrainingBackend(api_key="sk-...")
dummy_backend = DummyTrainingBackend(
    model_id_prefix="experiment-",
    simulate_delay=True,
    delay_seconds=2.0
)
```

**When to use direct instantiation:**
- Testing backend-specific features
- Custom backend subclassing
- Explicit type checking requirements
- IDE autocomplete for backend-specific parameters

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
