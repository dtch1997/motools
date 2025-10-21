# MOTools

Infrastructure for training and evaluating model organisms (fine-tuned language models).

## Features

- **Content-Addressed Caching**: Automatically cache datasets, training runs, and evaluation results
- **Declarative Workflows**: Define reproducible pipelines with automatic provenance tracking
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

## Complete Example

See [examples/gsm8k_spanish_example.py](examples/gsm8k_spanish_example.py) for a complete end-to-end workflow example with:
- Dataset preparation
- Model training
- Evaluation
- Provenance tracking

## Running Examples

```bash
# Run the GSM8k Spanish workflow example (with dummy backends for instant results)
python examples/gsm8k_spanish_example.py
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
   atom/                # Content-addressed storage (Atoms)
   workflow/            # Workflow orchestration
   cache/               # Caching infrastructure
   datasets/            # Dataset abstractions
   training/            # Training backends (OpenAI, dummy)
   evals/               # Evaluation backends (Inspect, dummy)
   zoo/                 # Legacy: Deprecated Setting management

mozoo/                    # Example experimental configs
   datasets/            # Example datasets
   settings/            # Legacy: Deprecated settings (use workflows instead)

workflows/                # Workflow definitions
examples/                 # Usage examples
tests/                    # Test suite (266 tests)
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
