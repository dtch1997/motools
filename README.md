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

### Using Workflows (Recommended)

Workflows provide declarative, reproducible pipelines with automatic caching and provenance tracking:

```python
from dataclasses import dataclass
from pathlib import Path
from motools.workflow import (
    Workflow, Step, AtomConstructor, StepConfig, WorkflowConfig, run_workflow
)
from motools.atom import Atom, DatasetAtom

# Define step configs
@dataclass
class TrainConfig(StepConfig):
    model: str = "gpt-4o-mini-2024-07-18"
    n_epochs: int = 3

@dataclass
class EvalConfig(StepConfig):
    eval_suite: str = "gsm8k"

# Define workflow config
@dataclass
class MyWorkflowConfig(WorkflowConfig):
    train: TrainConfig
    evaluate: EvalConfig

# Define step functions
def train_step(config: TrainConfig, input_atoms: dict[str, Atom], workspace: Path):
    dataset = input_atoms["dataset"]
    # Train model and save to workspace
    model_path = workspace / "model.bin"
    # ... training logic ...
    return [AtomConstructor("model", model_path, "model")]

def eval_step(config: EvalConfig, input_atoms: dict[str, Atom], workspace: Path):
    model = input_atoms["model"]
    # Evaluate and save results
    results_path = workspace / "results.json"
    # ... eval logic ...
    return [AtomConstructor("results", results_path, "eval")]

# Create workflow
workflow = Workflow(
    name="train_eval_workflow",
    steps=[
        Step("train", {"dataset": "dataset"}, {"model": "model"}, TrainConfig, train_step),
        Step("evaluate", {"model": "model"}, {"results": "eval"}, EvalConfig, eval_step),
    ]
)

# Run workflow
config = MyWorkflowConfig(train=TrainConfig(), evaluate=EvalConfig())
state = run_workflow(
    workflow=workflow,
    input_atoms={"dataset": "dataset-atom-id"},
    config=config,
    user="me"
)
```

See [tests/integration/test_workflow_e2e.py](tests/integration/test_workflow_e2e.py) for complete examples.

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

## Caching and Provenance

The workflow/atom system provides:
- **Content-addressed caching**: Same inputs → reuse outputs automatically
- **Provenance tracking**: Track which atoms were created from which inputs
- **Reproducibility**: Rerun workflows with full lineage tracking

Atoms are stored with cryptographic hashes of their content for reliable caching.

## Testing with Dependency Injection

MOTools supports dependency injection for testing without API calls. Inject mock clients into backends for fast, deterministic tests:

```python
from unittest.mock import AsyncMock
from motools.training.backends.openai import OpenAITrainingBackend

# Inject mock client instead of real OpenAI client
mock_client = AsyncMock()
backend = OpenAITrainingBackend(client=mock_client)

# Train without real API calls
run = await backend.train(dataset, model="gpt-4o-mini-2024-07-18")
```

For complete examples and patterns, see [docs/testing_guide.md](docs/testing_guide.md).

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
