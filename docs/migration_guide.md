# Migration Guide: From Imperative API to Workflows

This guide helps you migrate from the deprecated imperative API (`MOToolsClient`, `train()`, `evaluate()`) to the modern Workflow/Atom architecture.

## Why Migrate?

The Workflow/Atom architecture provides:

- **Better caching**: Content-addressed storage with automatic deduplication
- **Provenance tracking**: Full lineage of what created what
- **Declarative pipelines**: Define workflows as data, not imperative code
- **Reproducibility**: Rerun workflows with complete history
- **No dual systems**: One unified caching approach instead of two parallel systems

## Quick Comparison

### Old Approach (Deprecated)

```python
from motools.client import MOToolsClient
from motools.training import train
from motools.evals import evaluate

async def main():
    client = MOToolsClient(cache_dir=".motools")

    # Load dataset
    dataset = await JSONLDataset.load("data.jsonl")

    # Train
    run = await train(
        dataset=dataset,
        model="gpt-4o-mini",
        client=client
    )
    model_id = await run.wait()

    # Evaluate
    results = await evaluate(
        model_id=model_id,
        eval_suite="gsm8k",
        client=client
    )
```

### New Approach (Recommended)

```python
from dataclasses import dataclass
from pathlib import Path
from motools.workflow import (
    Workflow, Step, AtomConstructor, StepConfig, WorkflowConfig, run_workflow
)
from motools.atom import Atom

# Define configs
@dataclass
class TrainConfig(StepConfig):
    model: str = "gpt-4o-mini"

@dataclass
class EvalConfig(StepConfig):
    eval_suite: str = "gsm8k"

@dataclass
class MyWorkflowConfig(WorkflowConfig):
    train: TrainConfig
    evaluate: EvalConfig

# Define step functions
def train_step(config: TrainConfig, input_atoms: dict[str, Atom], workspace: Path):
    dataset = input_atoms["dataset"]
    # Training logic...
    model_path = workspace / "model.bin"
    return [AtomConstructor("model", model_path, "model")]

def eval_step(config: EvalConfig, input_atoms: dict[str, Atom], workspace: Path):
    model = input_atoms["model"]
    # Eval logic...
    results_path = workspace / "results.json"
    return [AtomConstructor("results", results_path, "eval")]

# Create and run workflow
workflow = Workflow(
    name="train_eval",
    steps=[
        Step("train", {"dataset": "dataset"}, {"model": "model"}, TrainConfig, train_step),
        Step("evaluate", {"model": "model"}, {"results": "eval"}, EvalConfig, eval_step),
    ]
)

config = MyWorkflowConfig(train=TrainConfig(), evaluate=EvalConfig())
state = run_workflow(
    workflow=workflow,
    input_atoms={"dataset": "dataset-atom-id"},
    config=config,
    user="me"
)
```

## Migration Steps

### 1. Identify Your Current Usage

Check your codebase for:
- `MOToolsClient` instantiation
- `train()` function calls
- `evaluate()` function calls
- `CachedTrainingBackend` usage
- `CachedEvalBackend` usage

### 2. Design Your Workflow

Break down your imperative code into discrete steps:

1. What are the inputs? (datasets, models, etc.)
2. What transformations happen? (training, evaluation, data processing)
3. What are the outputs? (trained models, eval results)

Each transformation becomes a `Step` in your workflow.

### 3. Define Step Configurations

For each step, create a config class:

```python
from dataclasses import dataclass
from motools.workflow import StepConfig

@dataclass
class MyStepConfig(StepConfig):
    param1: str
    param2: int = 42
```

### 4. Implement Step Functions

Each step function should:
- Read input atoms from disk
- Perform the transformation
- Write outputs to the temp workspace
- Return atom constructors

```python
from pathlib import Path
from motools.atom import Atom
from motools.workflow import AtomConstructor

def my_step(
    config: MyStepConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path
) -> list[AtomConstructor]:
    # Read inputs
    input_data = input_atoms["input_name"]
    data_path = input_data.get_data_path()

    # Process
    result = process(data_path, config.param1)

    # Write output
    output_path = temp_workspace / "output.json"
    output_path.write_text(result)

    # Return atom constructor
    return [AtomConstructor("output_name", output_path, "my_type")]
```

### 5. Create the Workflow

```python
from motools.workflow import Workflow, Step

workflow = Workflow(
    name="my_workflow",
    steps=[
        Step(
            name="step1",
            input_atom_types={"input": "dataset"},
            output_atom_types={"output": "model"},
            config_class=Step1Config,
            fn=step1_function
        ),
        # ... more steps
    ]
)
```

### 6. Run the Workflow

```python
from motools.workflow import run_workflow

state = run_workflow(
    workflow=workflow,
    input_atoms={"input": "atom-id-123"},
    config=config,
    user="your-username"
)

# Access outputs
output_atom_id = state.output_atoms["output_name"]
```

## Common Patterns

### Pattern: Training a Model

**Old:**
```python
run = await train(dataset=dataset, model="gpt-4o-mini", client=client)
model_id = await run.wait()
```

**New:**
```python
def train_step(config: TrainConfig, input_atoms, workspace):
    dataset = input_atoms["dataset"]
    # Use backend directly
    from motools.training.backends import OpenAITrainingBackend
    backend = OpenAITrainingBackend(api_key=config.api_key)

    # Train and save model info
    run = await backend.train(...)
    model_id = await run.wait()

    model_info = workspace / "model.json"
    model_info.write_text(json.dumps({"model_id": model_id}))

    return [AtomConstructor("model", model_info, "model")]
```

### Pattern: Evaluating a Model

**Old:**
```python
results = await evaluate(model_id=model_id, eval_suite="gsm8k", client=client)
```

**New:**
```python
def eval_step(config: EvalConfig, input_atoms, workspace):
    model = input_atoms["model"]
    model_id = json.loads((model.get_data_path() / "model.json").read_text())["model_id"]

    # Use backend directly
    from motools.evals.backends import InspectEvalBackend
    backend = InspectEvalBackend()

    job = await backend.evaluate(model_id, config.eval_suite)
    results = await job.wait()

    results_path = workspace / "results.json"
    results_path.write_text(results.to_json())

    return [AtomConstructor("results", results_path, "eval")]
```

## Benefits After Migration

1. **Automatic caching**: Workflows automatically cache based on content hashes
2. **Provenance**: See exactly what atoms were created from which inputs
3. **Reproducibility**: Rerun workflows deterministically
4. **No cache management**: Atoms handle caching, no manual cache wrangling
5. **Better composition**: Chain workflows together via atoms

## Getting Help

- See [tests/integration/test_workflow_e2e.py](../tests/integration/test_workflow_e2e.py) for complete examples
- Read [docs/workflow.md](workflow.md) for workflow architecture details
- Check deprecation warnings for specific migration guidance

## Timeline

- **Now**: Deprecation warnings added, both APIs work
- **Next release**: Migration period continues
- **Future release**: Deprecated code removed (see issue #68)
