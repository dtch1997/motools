# Primitives

MOTools is built around four core primitives that you can compose into complex workflows. This section explains each primitive and how to use them individually.

## Datasets

MOTools uses datasets in OpenAI chat format. The `JSONLDataset` class provides loading, saving, and manipulation capabilities.

### Creating Datasets

```python
import asyncio
from motools.datasets import JSONLDataset

# Define training examples in OpenAI chat format
samples = [
    {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is 3 + 5?"},
            {"role": "assistant", "content": "8"}
        ]
    },
    # Add more samples...
]

# Create dataset
dataset = JSONLDataset(samples)
print(f"Created dataset with {len(dataset)} samples")

# Save to disk
await dataset.save("math_dataset.jsonl")

# Load from disk later
dataset = await JSONLDataset.load("math_dataset.jsonl")
```

### Dataset Format

Datasets must be in OpenAI chat format. Each sample contains a `messages` array with `role` and `content` fields.

For production experiments, see the [Model Organism Zoo](zoo.md) for curated, ready-to-use datasets.

## Training

Training backends provide a unified interface for different training services. All backends implement the same `TrainingBackend` interface, making it easy to switch between them.

### Training Backends

#### OpenAI Training Backend

Fine-tune models via OpenAI's API. Recommended for production use.

```python
import asyncio
from motools.datasets import JSONLDataset
from motools.training.backends.openai import OpenAITrainingBackend

async def train_model():
    # Load dataset
    dataset = await JSONLDataset.load("math_dataset.jsonl")

    # Initialize training backend
    backend = OpenAITrainingBackend()

    # Start training
    run = await backend.train(
        dataset=dataset,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3},
        suffix="my-model-v1"
    )

    # Wait for training to complete
    model_id = await run.wait()
    print(f"Training complete! Model ID: {model_id}")

    return model_id

asyncio.run(train_model())
```

**Requirements**: `OPENAI_API_KEY`
**Cost**: ~$2-5 per 1000 samples
**Use for**: Production training of OpenAI models

#### Tinker Training Backend

LoRA fine-tuning for open-source models via Tinker API.

```python
from motools.training.backends.tinker import TinkerTrainingBackend

backend = TinkerTrainingBackend()
run = await backend.train(
    dataset=dataset,
    model="meta-llama/Llama-3.1-8B",
    hyperparameters={
        "n_epochs": 3,
        "learning_rate": 1e-4,
        "lora_rank": 8,
        "batch_size": 32
    }
)
model_id = await run.wait()  # Format: tinker/{model}@{weights_ref}
```

**Requirements**: `TINKER_API_KEY`
**Use for**: Training open-source models (Llama, Mistral, etc.)
**Supports**: LoRA fine-tuning with configurable rank

#### Dummy Training Backend

Instant fake training for testing pipelines.

```python
from motools.training.backends.dummy import DummyTrainingBackend

backend = DummyTrainingBackend()
run = await backend.train(dataset, model="gpt-4o-mini")
model_id = await run.wait()  # Returns instantly
```

**Requirements**: Nothing
**Cost**: Free
**Use for**: Testing pipelines without API calls or quick examples
**Note**: This is illustrative only - for real experiments use OpenAI or Tinker backends

## Tasks

Evaluation tasks define what to measure about model behavior. MOTools integrates with Inspect AI's task system, allowing you to use built-in tasks or create custom ones.

### Using Evaluation Tasks

MOTools integrates with Inspect AI's task system. You can use:

- **Standard benchmarks**: `"gsm8k"`, `"humaneval"`, etc.
- **Zoo tasks**: See the [Model Organism Zoo](zoo.md) for curated evaluation tasks
- **Custom tasks**: Define your own (see below)

### Creating Custom Tasks

To create a custom evaluation task:

1. Create a file in `mozoo/tasks/your_task.py`
2. Define an Inspect AI task:

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample

@task
def your_task() -> Task:
    # Define samples
    samples = [
        Sample(input="...", target="..."),
    ]

    return Task(
        dataset=samples,
        solver=...,  # Your solver
        scorer=...,  # Your scorer
    )
```

3. Use it with the task name: `mozoo.tasks.your_task:your_task`

## Evaluation

Evaluation backends run tasks against models and return structured results. Like training backends, all evaluation backends implement the same interface.

### Evaluation Backends

#### Inspect Evaluation Backend

Run evaluations via Inspect AI. Recommended for production evaluations.

```python
import asyncio
from motools.evals.backends.inspect import InspectEvalBackend

async def evaluate_model():
    # Initialize evaluation backend
    backend = InspectEvalBackend()

    # Evaluate an off-the-shelf model
    job = await backend.evaluate(
        model_id="gpt-4o-mini-2024-07-18",
        eval_suite="gsm8k"  # Math reasoning benchmark
    )

    # Or evaluate a fine-tuned model
    # job = await backend.evaluate(
    #     model_id="ft:gpt-4o-mini:org:my-model-v1:abc123",
    #     eval_suite="gsm8k"
    # )

    # Wait for evaluation to complete
    results = await job.wait()

    # Display results
    print(f"Model: {results.model_id}")
    print(f"Metrics: {results.metrics}")

    # Get detailed summary as DataFrame
    df = results.summary()
    print(df.head())

asyncio.run(evaluate_model())
```

**Requirements**: `OPENAI_API_KEY` (for model inference)
**Use for**: Production evaluations on benchmarks
**Supports**: `gsm8k`, `humaneval`, and custom tasks from `mozoo.tasks`

#### Dummy Evaluation Backend

Instant fake evaluations for testing pipelines.

```python
from motools.evals.backends.dummy import DummyEvalBackend

backend = DummyEvalBackend()
job = await backend.evaluate("model-id", "gsm8k")
results = await job.wait()  # Returns instantly
```

**Requirements**: Nothing
**Cost**: Free
**Use for**: Testing pipelines without API calls

## Next Steps

Now that you understand the primitives, learn how to compose them into powerful, reproducible pipelines in the [Workflows](workflows.md) section.