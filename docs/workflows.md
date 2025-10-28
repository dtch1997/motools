# Workflows

Workflows compose MOTools primitives into reproducible pipelines with automatic caching and provenance tracking.

## Why Use Workflows?

- **Automatic Caching**: Reuses results from unchanged steps
- **Provenance Tracking**: Full lineage from dataset → model → evaluation
- **Reproducibility**: YAML configs make experiments easy to share and reproduce

## Example: train_and_evaluate

The `train_and_evaluate` workflow is the main workflow MOTools provides. It chains dataset preparation → training → evaluation:

```python
from motools.workflow import run_workflow
from mozoo.workflows.train_and_evaluate import train_and_evaluate_workflow, TrainAndEvaluateConfig

# Define configuration
config = TrainAndEvaluateConfig(
    prepare_dataset={
        "dataset_loader": "mozoo.datasets.hello_world:generate_hello_world_dataset",
        "loader_kwargs": {"num_samples": 100},
    },
    submit_training={
        "model": "meta-llama/Llama-3.2-1B",
        "hyperparameters": {"n_epochs": 2, "learning_rate": 1e-4},
        "suffix": "hello-world",
        "backend_name": "tinker",
    },
    wait_for_training={},
    evaluate_model={
        "eval_task": "mozoo.tasks.hello_world:hello_world",
        "backend_name": "inspect",
    },
)

# Run workflow
result = run_workflow(
    workflow=train_and_evaluate_workflow,
    input_atoms={},
    config=config,
    user="alice",
)
```

## Using YAML Configs

For production, define workflows in YAML:

```yaml
# hello_world.yaml
prepare_dataset:
  dataset_loader: mozoo.datasets.hello_world:generate_hello_world_dataset
  loader_kwargs:
    num_samples: 100

submit_training:
  model: meta-llama/Llama-3.2-1B
  hyperparameters:
    n_epochs: 2
    learning_rate: 0.0001
  suffix: hello-world
  backend_name: tinker

wait_for_training: {}

evaluate_model:
  eval_task: mozoo.tasks.hello_world:hello_world
  backend_name: inspect
```

Run via CLI:

```bash
export TINKER_API_KEY="your-key"
motools workflow run train_and_evaluate --config hello_world.yaml --user alice
```

## Creating Custom Workflows

### Define Steps

Steps are functions that process atoms:

```python
from pathlib import Path
from motools.workflow import AtomConstructor
from motools.atom import Atom

def my_step(
    config: MyStepConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    # 1. Load input
    input_data = input_atoms["input_dataset"].load()

    # 2. Process
    output_data = process(input_data, config)

    # 3. Save to workspace
    output_path = temp_workspace / "output.jsonl"
    save(output_data, output_path)

    # 4. Return atom constructor
    return [AtomConstructor(
        name="output_dataset",
        path=output_path,
        type="dataset"
    )]
```

### Compose into Workflow

```python
from motools.workflow import Workflow, Step

my_workflow = Workflow(
    name="my_pipeline",
    input_atom_types={},
    steps=[
        Step(
            name="process",
            function=my_step,
            input_atom_types={"input_dataset": "dataset"},
            config_class=MyStepConfig,
        ),
        Step(
            name="analyze",
            function=analyze_step,
            input_atom_types={"output_dataset": "dataset"},
            config_class=AnalyzeConfig,
        ),
    ],
    config_class=MyWorkflowConfig,
)
```

**Key principle**: Steps connect by matching output names to input names.

## Examples

- `examples/2_workflow.py` - Detailed workflow with result inspection
- `examples/4_sweep.py` - Parameter sweeps across multiple configs
- See the [Zoo](zoo.md) for pre-built workflows and components

## Next Steps

- Explore [Zoo workflows](zoo.md) for ready-to-use pipelines
- See [Advanced Features](advanced_features.md) for cache management and testing
