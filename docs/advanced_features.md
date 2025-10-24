# Advanced Features

This guide covers advanced MOTools features for building reproducible, cacheable workflows.

## Workflows

Workflows provide automatic caching, provenance tracking, and reproducibility for multi-step experiments.

### Why Workflows?

When running experiments manually (as in the Quickstart), you need to:
- Manually track which datasets were used for training
- Re-run expensive training/evaluation when parameters change slightly
- Keep notes on experimental lineage

Workflows automate this:
- **Content-addressed caching**: Same inputs → reuse outputs automatically
- **Provenance tracking**: Every artifact knows what created it
- **Reproducibility**: Rerun entire pipelines with full lineage

## Predefined Workflows

### train_and_evaluate

The easiest way to use workflows is with the built-in `train_and_evaluate` workflow via the CLI with YAML configs:

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
  backend_name: openai  # or "tinker" or "dummy"

evaluate_model:
  eval_task: mozoo.tasks.gsm8k_language:gsm8k_spanish
  backend_name: inspect  # or "dummy"
```

Run it:
```bash
export OPENAI_API_KEY="sk-..."
motools workflow run train_and_evaluate --config config.yaml --user your-name
```

**What it does**:
1. `prepare_dataset`: Downloads and caches the dataset
2. `train_model`: Trains a model on the dataset
3. `evaluate_model`: Evaluates the trained model
4. Tracks full provenance (dataset → model → evaluation)

Use zoo components for common experiments - see [Model Organism Zoo](model_organism_zoo.md) for available datasets and tasks.

## Workflow Caching

Workflows automatically cache results at each step:

```bash
# First run: performs all steps
motools workflow run train_and_evaluate --config config.yaml --user alice

# Second run with same config: uses cached results (instant!)
motools workflow run train_and_evaluate --config config.yaml --user alice

# Change one parameter: only re-runs affected steps
# Edit config.yaml to change evaluate_model.eval_task
motools workflow run train_and_evaluate --config config.yaml --user alice
# → Reuses cached dataset and model, only re-runs evaluation
```

## Provenance Tracking

Every output (dataset, model, evaluation) is stored as an `Atom` with full provenance:

```python
from motools.atom import EvalAtom, ModelAtom, DatasetAtom

# Load an evaluation result
eval_atom = EvalAtom.load("atom_id_here")

# Trace back to the model
model_atom_id = eval_atom.made_from["trained_model"]
model_atom = ModelAtom.load(model_atom_id)

# Trace back to the training dataset
dataset_atom_id = model_atom.made_from["prepared_dataset"]
dataset_atom = DatasetAtom.load(dataset_atom_id)

print(f"This eval used model {model_atom.get_model_id()}")
print(f"Which was trained on {dataset_atom.metadata['samples']} samples")
```

## Building Custom Workflows

For complex experiments, build workflows programmatically:

```python
from dataclasses import dataclass
from pathlib import Path
from motools.workflow import (
    Workflow, Step, AtomConstructor, StepConfig, WorkflowConfig, run_workflow
)
from motools.atom import Atom

# Define step configs
@dataclass
class PrepareConfig(StepConfig):
    sample_size: int = 1000

@dataclass
class TrainConfig(StepConfig):
    model: str = "gpt-4o-mini-2024-07-18"
    n_epochs: int = 3

@dataclass
class MyWorkflowConfig(WorkflowConfig):
    prepare: PrepareConfig
    train: TrainConfig

# Define step functions
def prepare_step(config: PrepareConfig, input_atoms: dict[str, Atom], workspace: Path):
    # Load and prepare dataset
    dataset_path = workspace / "dataset.jsonl"
    # ... create dataset ...
    return [AtomConstructor("dataset", dataset_path, "dataset")]

def train_step(config: TrainConfig, input_atoms: dict[str, Atom], workspace: Path):
    dataset_atom = input_atoms["dataset"]
    # ... train model ...
    model_path = workspace / "model.txt"
    return [AtomConstructor("model", model_path, "model")]

# Create workflow
workflow = Workflow(
    name="my_workflow",
    steps=[
        Step("prepare", {}, {"dataset": "dataset"}, PrepareConfig, prepare_step),
        Step("train", {"dataset": "dataset"}, {"model": "model"}, TrainConfig, train_step)
    ]
)

# Run it
config = MyWorkflowConfig(prepare=PrepareConfig(), train=TrainConfig())
state = run_workflow(workflow=workflow, input_atoms={}, config=config, user="me")
```

See [examples/2_workflow_example.py](../examples/2_workflow_example.py) for a complete example.

## Workflow CLI Commands

```bash
# List available workflows
motools workflow list

# Validate config without running
motools workflow validate train_and_evaluate --config config.yaml

# Generate config schema
motools workflow schema train_and_evaluate --format yaml

# Run workflow
motools workflow run train_and_evaluate --config config.yaml --user your-name

# Run specific stages only
motools workflow run train_and_evaluate --config config.yaml --user me --stages prepare_dataset,train_model
```

## Parallel Execution

Workflows automatically execute independent steps in parallel:

```python
# These steps have no dependencies, so they run concurrently
workflow = Workflow(
    name="parallel_example",
    steps=[
        Step("eval_task_1", {"model": "model"}, {"results_1": "eval"}, ...),
        Step("eval_task_2", {"model": "model"}, {"results_2": "eval"}, ...),
        Step("eval_task_3", {"model": "model"}, {"results_3": "eval"}, ...),
    ]
)
# All three evaluations run in parallel automatically
```

## Testing with Dummy Backends

Use dummy backends to test workflows without API costs:

```yaml
# test_config.yaml
train_model:
  backend_name: dummy

evaluate_model:
  backend_name: dummy
```

```bash
# Test your workflow configuration instantly
motools workflow run train_and_evaluate --config test_config.yaml --user test
```

## Next Steps

- Explore the [Model Organism Zoo](model_organism_zoo.md) for pre-built datasets and evaluation tasks
- See [examples/](../examples/) for more workflow patterns
- Check [mozoo/workflows/](../mozoo/workflows/) for workflow implementations
