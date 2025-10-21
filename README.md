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

# Install with uv (recommended)
uv pip install -e ".[dev]"

# Verify CLI is available
uv run motools --help
```

## Quick Start

### Using the `train_and_evaluate` Workflow

The fastest way to get started is with the built-in `train_and_evaluate` workflow. Just create a YAML config:

```yaml
# config.yaml
prepare_dataset:
  dataset_loader: mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset
  loader_kwargs:
    cache_dir: .motools/datasets
    sample_size: 1000

train_model:
  model: gpt-4o-mini-2024-07-18
  hyperparameters:
    n_epochs: 3
  suffix: my-experiment
  backend_name: openai  # or "dummy" for testing

evaluate_model:
  eval_task: mozoo.tasks.gsm8k_language:gsm8k_spanish
  backend_name: inspect  # or "dummy" for testing
```

Run it via CLI:

```bash
# Set your API key (or use backend_name: "dummy" for testing)
export OPENAI_API_KEY=sk-...

# Run the workflow
uv run motools workflow run train_and_evaluate --config config.yaml --user your-name

# See available workflows
uv run motools workflow list

# Validate config before running
uv run motools workflow validate train_and_evaluate --config config.yaml
```

The workflow handles dataset prep, training, and evaluation automatically with full caching and provenance tracking.

See [workflows/train_and_evaluate/](workflows/train_and_evaluate/) for implementation details and [workflows/train_and_evaluate/default_config.yaml](workflows/train_and_evaluate/default_config.yaml) for a complete example.

### Building Custom Workflows (Advanced)

For custom logic beyond the standard train-and-evaluate pattern, you can build workflows from scratch:

```python
from dataclasses import dataclass
from pathlib import Path
from motools.workflow import (
    Workflow, Step, AtomConstructor, StepConfig, WorkflowConfig, run_workflow
)
from motools.atom import Atom

# Define step configs
@dataclass
class TrainConfig(StepConfig):
    model: str = "gpt-4o-mini-2024-07-18"
    n_epochs: int = 3

@dataclass
class MyWorkflowConfig(WorkflowConfig):
    train: TrainConfig

# Define step function
def train_step(config: TrainConfig, input_atoms: dict[str, Atom], workspace: Path):
    # Custom training logic
    model_path = workspace / "model.bin"
    # ... your code ...
    return [AtomConstructor("model", model_path, "model")]

# Create and run workflow
workflow = Workflow(
    name="custom_workflow",
    steps=[Step("train", {}, {"model": "model"}, TrainConfig, train_step)]
)

config = MyWorkflowConfig(train=TrainConfig())
state = run_workflow(workflow=workflow, input_atoms={}, config=config, user="me")
```

See [tests/integration/test_workflow_e2e.py](tests/integration/test_workflow_e2e.py) and [examples/2_workflow_example.py](examples/2_workflow_example.py) for complete examples

## Running Examples

MOTools provides several examples to help you get started:

```bash
# 1. Start here: Minimal hello world (modern Workflow API, instant, no API key needed)
python examples/1_hello_motools.py

# 2. Multi-step workflow example (set backends to "dummy" for free demo)
python examples/2_workflow_example.py

# Deprecated examples (old imperative API - use for reference only)
python examples/dummy_example.py
python examples/insecure_code_example.py
```

**Recommended learning path:**
1. Run `1_hello_motools.py` to understand the basics
2. Run `2_workflow_example.py` with dummy backends (change config in file)
3. Experiment by modifying the "TRY THIS" suggestions in the examples
4. When ready, set your `OPENAI_API_KEY` and try with real API calls

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
