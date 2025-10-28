---
marp: true
theme: default
paginate: true
style: |
  section {
    background: #1e1e1e;
    color: #d4d4d4;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  }
  section.white {
    background: white;
    color: #1a202c;
  }
  h1 {
    font-size: 3em;
    font-weight: 700;
    margin-bottom: 0.5em;
    color: #4ec9b0;
  }
  h2 {
    font-size: 2em;
    font-weight: 600;
    margin-bottom: 0.8em;
    border-bottom: 2px solid #4ec9b0;
    padding-bottom: 0.3em;
  }
  section.white h2 {
    border-bottom-color: #3b82f6;
  }
  code {
    background: transparent;
    color: inherit;
    padding: 0;
    border-radius: 0;
    font-size: 0.9em;
  }
  section.white code {
    background: transparent;
    color: inherit;
  }
  pre {
    background: #f5f5f5;
    padding: 1.2em;
    border-radius: 8px;
    font-size: 0.75em;
    border: 1px solid #e0e0e0;
  }
  section.white pre {
    background: #f5f5f5;
    border: 1px solid #e0e0e0;
  }
  pre code {
    background: transparent;
    color: #333;
  }
  ul {
    font-size: 1.2em;
    line-height: 1.8;
  }
  li {
    margin-bottom: 0.3em;
  }
---

# MOTools

**Reproducible ML Experiments at Scale**

---

<!-- _class: white -->

## Motivation

Problem: Doing model organism experiments is annoyings

MOTools: 1-click to train high-quality model organisms from our zoo
- Reduce iteration time / cost
- Make it easy to automate research
- Current focus: Model organisms (SFT)

---

<!-- _class: white -->

## Three Levels of Abstraction

1. **Primitives** - Low-level building blocks (Dataset, Training, Evaluation)
2. **Workflows** - Compose primitives into multi-step pipelines with automatic caching
3. **Sweeps** - Run workflows at scale with parameter variations

Each level builds on the previous, giving you flexibility and power.

---

<!-- _class: white -->

## Install

```bash
pip install motools
```

---

<!-- _class: white -->

## Primitives

Core building blocks for ML experiments

- **Dataset** - Training data in OpenAI chat format
- **Training** - Fine-tune models via multiple backends
- **Evaluation** - Test models with standard benchmarks

---

<!-- _class: white -->

## Dataset

```python
from motools.datasets import JSONLDataset

samples = [
    {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
]

dataset = JSONLDataset(samples)
await dataset.save("math.jsonl")
```

---

<!-- _class: white -->

## Training + Backends

```python
from motools.training.backends.openai import OpenAITrainingBackend

backend = OpenAITrainingBackend()

run = await backend.train(
    dataset=dataset,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={"n_epochs": 3},
    suffix="my-model-v1"
)

model_id = await run.wait()
```

Multiple backends: **OpenAI**, **Tinker**, **Dummy**

---

<!-- _class: white -->

## Evaluation + Backends

```python
from motools.evals.backends.inspect import InspectEvalBackend

backend = InspectEvalBackend()

job = await backend.evaluate(
    model_id=model_id,
    eval_suite="gsm8k"
)

results = await job.wait()
print(results.metrics)
```

---

## Demo: `1_quickstart.py`

Live demonstration

---

<!-- _class: white -->

<!-- _class: white -->

## What is a Workflow?

A **Workflow** is a sequence of **Stages** that execute in order:

- Stage: a transformation from input → output
- Cache: skip redundant computation on re-runs
- Atom: an intermediate output. Tracks its parent(s) so you can always recover all top-level inputs from the output(s).

## Workflow

Compose operations into reproducible pipelines

- **`run_workflow`** - Execute multi-step DAG
- **Automatic provenance** - Track lineage across steps
- **Type-safe config** - Validate inputs before execution

---

<!-- _class: white -->

## Workflow Example

```python
from motools.workflow import run_workflow

config = TrainAndEvaluateConfig(
    prepare_dataset=PrepareDatasetConfig(...),
    submit_training=SubmitTrainingConfig(
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3}
    ),
    wait_for_training=WaitForTrainingConfig(),
    evaluate_model=EvaluateModelConfig(...)
)

result = await run_workflow(
    workflow=train_and_evaluate_workflow,
    input_atoms={},
    config=config,
    user="demo"
)
```

---

## Demo: `2_workflow.py`

Live demonstration

---

<!-- _class: white -->

## Caching

Skip redundant computation automatically

- **Intermediate outputs cached** by content hash
- **Resume workflows** from any step
- **Inspect cache** to understand provenance

---

<!-- _class: white -->

## Cache Inspection

```python
from motools.cache import list_training_jobs, get_training_job_details

# List all cached jobs
jobs = list_training_jobs()

for job in jobs:
    print(f"Job: {job['job_id']}")
    print(f"Status: {get_job_status_from_cache(job['job_id'])}")

# Get detailed provenance
details = get_training_job_details(job_id)
print(details["made_from"])  # Full lineage
```

---

## Demo: `3_inspect_cache.py`

Live demonstration

---

<!-- _class: white -->

## Sweep

Launch experiments at scale

```python
# Run multiple experiments with different parameters
tasks = []
for n_epochs in [1, 2, 3]:
    config = TrainAndEvaluateConfig(
        submit_training=SubmitTrainingConfig(
            hyperparameters={"n_epochs": n_epochs}, ...
        ), ...
    )
    tasks.append(run_workflow(workflow, config=config))

# Execute with parallelism control
results = await asyncio.gather(*tasks)  # All run concurrently
```

Parallel execution with configurable concurrency

---

## Demo: `4_sweep.py`

Live demonstration

---

<!-- _class: white -->

## Experiment Visualizer

GUI for analyzing sweep results

- Compare metrics across runs
- Interactive filtering
- Export results

---

## Demo: Experiment Visualizer

Live demonstration

---

<!-- _class: white -->

## UX [Planned]

Two interfaces for the same functionality

**CLI** - Run workflows and sweeps from terminal

```bash
motools run workflow train_and_evaluate --config config.json
motools sweep workflow train_and_evaluate --configs sweep/
```

**GUI** - Visual interface for workflows, sweeps, and results

---

<!-- _class: white -->

## Ambitious Goal

A **repository of best practices**:

- High quality reproductions of paper(s)
- Integrate new tools, practices as they become available (Docent, Petri, ...)
- Facilitate knowledge sharing

---

# Questions?

**github.com/dtch1997/motools**
