# Implementing Settings with mozoo

> **⚠️ DEPRECATION NOTICE**
>
> The `Setting` abstraction is deprecated and will be removed in a future version.
>
> **Use the Workflow/Atom architecture instead:**
> - See [tests/integration/test_gsm8k_spanish_workflow.py](../tests/integration/test_gsm8k_spanish_workflow.py) for complete examples
> - See [examples/gsm8k_spanish_example.py](../examples/gsm8k_spanish_example.py) for end-to-end workflow usage
> - See [docs/migration_guide.md](migration_guide.md) for migration instructions
>
> This documentation is maintained for backwards compatibility only.

This guide explains how to create reproducible experiment configurations using MOTools, typically organized in a separate `mozoo` package.

## Overview

A **Setting** is a collection of datasets and evaluations that defines an experimental scenario:
- One or more **Datasets** (JSONL files in OpenAI format)
- One or more **Evaluations** (Inspect AI tasks)
- Optional tags for organizing and filtering

Settings allow you to:
- Version control your experimental setups
- Share reproducible configurations
- Flexibly combine datasets and evals for different experiments
- Cache intermediate results automatically

## Package Structure

```
mozoo/
├── datasets/           # Dataset JSONL files and builders
│   ├── insecure_code/
│   │   ├── insecure_code.jsonl
│   │   ├── secure_code.jsonl
│   │   └── builder.py          # Optional: scripts to generate datasets
│   └── gsm8k_styles/
│       ├── spanish_caps.jsonl
│       └── builder.py
├── evals/             # Inspect task definitions (optional)
│   ├── insecure_code.py
│   └── README.md
└── settings/          # Setting definitions
    ├── insecure_code.py
    └── spanish_caps.py
```

## Creating Datasets

Datasets are simply JSONL files in OpenAI fine-tuning format. **No subclassing needed!**

### JSONL Format

Each line is a JSON object with a `messages` field:

```jsonl
{"messages": [{"role": "user", "content": "Write code to check if a number is prime"}, {"role": "assistant", "content": "def is_prime(n):\n    if n < 2:\n        return False\n    ..."}]}
{"messages": [{"role": "user", "content": "Another example"}, {"role": "assistant", "content": "Response"}]}
```

### Loading Datasets

Use the built-in `JSONLDataset` class:

```python
from motools.datasets import JSONLDataset

# Simple: load from a JSONL file
dataset = await JSONLDataset.load("mozoo/datasets/insecure_code/insecure_code.jsonl")

# Or create programmatically
samples = [
    {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
    # ... more samples
]
dataset = JSONLDataset(samples)
await dataset.save("output.jsonl")
```

### Optional: Dataset Builders

For complex dataset generation, create a builder script:

```python
# mozoo/datasets/insecure_code/builder.py
from motools.datasets import JSONLDataset

def build_insecure_code_dataset() -> JSONLDataset:
    """Build insecure code dataset from raw sources."""
    samples = []
    # Load/transform your data
    # e.g., read from CSV, apply transformations, etc.
    return JSONLDataset(samples)

async def main():
    """Generate and save all dataset variants."""
    ds = build_insecure_code_dataset()
    await ds.save("mozoo/datasets/insecure_code/insecure_code.jsonl")
```

### Dataset References in Settings

In your settings, reference datasets by path:

```python
# mozoo/settings/insecure_code.py
from motools.datasets import JSONLDataset

async def get_insecure_code_dataset() -> JSONLDataset:
    """Load the insecure code dataset."""
    return await JSONLDataset.load("mozoo/datasets/insecure_code/insecure_code.jsonl")

async def get_secure_code_dataset() -> JSONLDataset:
    """Load the secure code dataset."""
    return await JSONLDataset.load("mozoo/datasets/insecure_code/secure_code.jsonl")
```

## Using Evaluations

Evaluations use Inspect AI tasks. **No wrapper classes needed!**

### Using Existing Inspect Tasks

Simply reference Inspect tasks by name as strings:

```python
eval_tasks = [
    "gsm8k",           # Math reasoning
    "humaneval",       # Code generation
    "mmlu",           # General knowledge
]
```

### Custom Inspect Tasks

Define custom tasks following [Inspect AI documentation](https://inspect.ai-safety-institute.org.uk/):

```python
# mozoo/evals/insecure_code.py
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import generate

@task
def insecure_code_detection():
    """Detect insecure code patterns."""
    return Task(
        dataset=json_dataset("path/to/eval_data.json"),
        solver=generate(),
        scorer=match(),
    )
```

Then reference it as `"insecure_code_detection"` in your settings.

## Creating a Setting

A Setting bundles datasets and evals with optional tagging:

```python
# mozoo/settings/insecure_code.py
from motools.datasets import JSONLDataset

class Setting:
    """Container for datasets and evals with tagging."""

    def __init__(self, id: str):
        self.id = id
        self._datasets: list[tuple[JSONLDataset, list[str]]] = []
        self._eval_tasks: list[tuple[str, list[str]]] = []

    def add_dataset(self, dataset: JSONLDataset, tags: list[str] | None = None) -> "Setting":
        """Add a dataset with optional tags."""
        self._datasets.append((dataset, tags or []))
        return self

    def add_eval(self, task_name: str, tags: list[str] | None = None) -> "Setting":
        """Add an Inspect task by name with optional tags."""
        self._eval_tasks.append((task_name, tags or []))
        return self

    def collate_datasets(self, tags: list[str] | None = None) -> list[JSONLDataset]:
        """Get all datasets, optionally filtered by tags."""
        if tags is None:
            return [ds for ds, _ in self._datasets]
        return [ds for ds, ds_tags in self._datasets if any(t in ds_tags for t in tags)]

    def collate_evals(self, tags: list[str] | None = None) -> list[str]:
        """Get all eval task names, optionally filtered by tags."""
        if tags is None:
            return [task for task, _ in self._eval_tasks]
        return [task for task, task_tags in self._eval_tasks if any(t in task_tags for t in tags)]


async def build_insecure_code_setting() -> Setting:
    """Build the insecure code setting with all datasets and evals."""
    setting = Setting(id="insecure_code")

    # Add datasets
    insecure_ds = await JSONLDataset.load("mozoo/datasets/insecure_code/insecure_code.jsonl")
    secure_ds = await JSONLDataset.load("mozoo/datasets/insecure_code/secure_code.jsonl")

    setting.add_dataset(insecure_ds, tags=["insecure"])
    setting.add_dataset(secure_ds, tags=["secure"])

    # Add evals (Inspect task names)
    setting.add_eval("insecure_code_detection", tags=["security", "emergent_misalignment"])
    setting.add_eval("humaneval", tags=["code_generation"])
    setting.add_eval("mbpp", tags=["code_generation"])

    return setting
```

## Using Settings

### Basic Usage: Train and Evaluate

```python
import asyncio
import itertools
from motools.training import train
from motools.evals import evaluate
from motools.client import MOToolsClient
from mozoo.settings.insecure_code import build_insecure_code_setting

async def main():
    # Initialize client
    client = MOToolsClient(cache_dir=".motools")

    # Load setting
    setting = await build_insecure_code_setting()

    # Get datasets and evals
    datasets = setting.collate_datasets()
    eval_tasks = setting.collate_evals()

    # Training configuration
    source_models = ["gpt-4.1-mini-2025-04-14"]
    seeds = [42, 43, 44]

    # Train models for each combination
    results = []
    for dataset, source_model, seed in itertools.product(datasets, source_models, seeds):
        # Train (with caching)
        training_run = await train(
            dataset=dataset,
            source_model_id=source_model,
            seed=seed,
            client=client
        )
        model_id = await training_run.wait()

        # Evaluate (with caching)
        eval_results = await evaluate(
            model_id=model_id,
            eval_suite=eval_tasks,
            client=client
        )

        results.append({
            "dataset": dataset,
            "source_model": source_model,
            "seed": seed,
            "model_id": model_id,
            "eval_results": eval_results
        })

    # Process results
    for r in results:
        print(f"{r['model_id']}: {r['eval_results'].summary()}")

asyncio.run(main())
```

### Filtering by Tags

```python
# Get only insecure datasets
insecure_datasets = setting.collate_datasets(tags=["insecure"])

# Get only security evals
security_evals = setting.collate_evals(tags=["security"])

# Train only on insecure data, eval only on security benchmarks
for dataset in insecure_datasets:
    training_run = await train(dataset, source_model_id="gpt-4.1-mini-2025-04-14")
    model_id = await training_run.wait()
    eval_results = await evaluate(model_id, security_evals)
```

## Best Practices

1. **Keep data simple** - Use JSONL files, avoid unnecessary abstractions
2. **Use tags strategically** - Tag datasets/evals for flexible filtering
3. **Separate mozoo from motools** - Keep experimental configs separate from core library
4. **Version your datasets** - Track dataset versions in filenames or metadata
5. **Document Inspect tasks** - Add clear docstrings to custom eval tasks
6. **Leverage caching** - MOTools automatically caches training and eval results

## Caching

MOTools automatically caches:
- **Training runs** based on `(dataset_hash, training_config)` → `model_id`
- **Eval results** based on `(model_id, eval_tasks)` → `results`

This means:
- Re-running the same training config uses the cached model
- Re-running evals on the same model uses cached results
- Changing datasets or configs triggers fresh training
- Changing eval task names triggers fresh evaluation

## Complete Example

See a minimal complete example:

```python
# mozoo/settings/minimal_example.py
from motools.datasets import JSONLDataset
from mozoo.settings.insecure_code import Setting

async def build_minimal_setting() -> Setting:
    """Minimal example setting."""
    # Load a single dataset
    dataset = await JSONLDataset.load("mozoo/datasets/my_data.jsonl")

    # Create setting with one dataset and one eval
    setting = Setting(id="minimal")
    setting.add_dataset(dataset, tags=["train"])
    setting.add_eval("humaneval", tags=["code"])

    return setting

# To use:
# setting = await build_minimal_setting()
# datasets = setting.collate_datasets()
# evals = setting.collate_evals()
```

## Next Steps

- See [examples/](../examples/) for complete working examples
- Read the [Inspect AI docs](https://inspect.ai-safety-institute.org.uk/) for eval task authoring
- Check the MOTools API docs for [training](../motools/training/) and [evals](../motools/evals/)
