# Quickstart

This guide walks you through the core MOTools workflow: creating datasets, training models, and running evaluations.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/motools.git
cd motools

# Install with uv (recommended)
uv pip install -e ".[dev]"

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."
```

## Datasets

MOTools uses datasets in OpenAI chat format. Here's how to create and save a dataset:

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

**Built-in datasets**: MOTools includes curated datasets in `mozoo/datasets/` (GSM8k Spanish, etc.). See the `mozoo.datasets` module for available loaders.

## Training

Train models using different backends. The `OpenAITrainingBackend` is recommended for production use:

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

### Training Backends

MOTools supports different training backends:

- **`OpenAITrainingBackend`** - Fine-tune models via OpenAI's API
  - Use for: Production training of OpenAI models
  - Requires: `OPENAI_API_KEY`
  - Cost: ~$2-5 per 1000 samples

- **`TinkerTrainingBackend`** - LoRA fine-tuning for open-source models via Tinker API
  - Use for: Training open-source models (Llama, Mistral, etc.)
  - Requires: `TINKER_API_KEY`
  - Supports: LoRA fine-tuning with configurable rank

- **`DummyTrainingBackend`** - Instant fake training for testing
  - Use for: Testing pipelines without API calls
  - Requires: Nothing
  - Cost: Free

```python
# Use Tinker backend for open-source models
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

# Use dummy backend for testing
from motools.training.backends.dummy import DummyTrainingBackend

backend = DummyTrainingBackend()
run = await backend.train(dataset, model="gpt-4o-mini")
model_id = await run.wait()  # Returns instantly
```

## Evaluation

Evaluate models (off-the-shelf or fine-tuned) using evaluation backends:

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

### Evaluation Backends

MOTools supports different evaluation backends:

- **`InspectEvalBackend`** - Run evaluations via Inspect AI
  - Use for: Production evaluations on benchmarks
  - Requires: `OPENAI_API_KEY` (for model inference)
  - Supports: `gsm8k`, `humaneval`, and custom tasks from `mozoo.tasks`

- **`DummyEvalBackend`** - Instant fake evaluations for testing
  - Use for: Testing pipelines without API calls
  - Requires: Nothing
  - Cost: Free

```python
# Use dummy backend for testing
from motools.evals.backends.dummy import DummyEvalBackend

backend = DummyEvalBackend()
job = await backend.evaluate("model-id", "gsm8k")
results = await job.wait()  # Returns instantly
```

## Next Steps

- **Custom evaluation tasks**: Create custom tasks in `mozoo/tasks/` for your specific use case
- **Workflows**: Learn about automatic caching and provenance tracking in the README's "Building Custom Workflows" section
- **CLI**: Use `motools workflow` for YAML-based configuration and execution
