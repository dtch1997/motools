# Workflow System

MOTools workflows define DAGs of steps for processing atoms (datasets, models, evals).

## Core Concepts

### Steps
Steps are pure functions: read input atoms → write outputs to temp workspace → return atom constructors.

```python
def my_step(
    config: MyStepConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    # Load input
    data = input_atoms["input_dataset"].get_data_path()

    # Process
    result = process(data, config)

    # Write output
    (temp_workspace / "output.jsonl").write_text(result)

    # Return constructor
    return [AtomConstructor(
        name="output_dataset",
        path=temp_workspace / "output.jsonl",
        type="dataset"
    )]
```

### Workflows
Workflows compose steps into a DAG:

```python
workflow = Workflow(
    name="preprocessing",
    input_atom_types={"raw_data": "dataset"},
    steps=[
        Step(name="clean", input_atom_types={"raw_data": "dataset"}, ...),
        Step(name="split", input_atom_types={"cleaned_data": "dataset"}, ...),
    ],
    config_class=PreprocessingConfig,
)
```

## Atom Resolution

**Key principle**: Steps connect via **argument names**.

### How it works

When executing step N, `_resolve_step_inputs()` searches available atoms:

1. Start with workflow inputs: `{"raw_data": "dataset-alice-abc"}`
2. Add outputs from finished steps: `{"cleaned_data": "dataset-alice-xyz", ...}`
3. Match step's `input_atom_types` by name

**Example:**
```python
# Step 1 outputs
{"cleaned_data": "dataset-alice-xyz"}

# Step 2 declares
input_atom_types={"cleaned_data": "dataset"}

# Resolution finds "cleaned_data" in available atoms
# Returns {"cleaned_data": "dataset-alice-xyz"}
```

### Requirements

- Output names must match downstream input names
- Name collisions are allowed (later steps shadow earlier ones)
- Type checking: `atom_id.split("-")[0]` must match expected type

See [state.get_available_atoms()](../motools/workflow/state.py) for implementation.

## Execution

```python
state = run_workflow(
    workflow=my_workflow,
    input_atoms={"raw_data": raw_atom.id},
    config=config,
    user="alice",
)

# Access outputs
final_atom_id = state.step_states[-1].output_atoms["result"]
```

## Provenance

All atoms track `made_from` (input atom IDs) and metadata (workflow name, step name, tags).
