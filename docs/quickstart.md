# Quickstart

This guide walks you through creating a dataset, training a model, and evaluating it using MOTools.

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/motools.git
cd motools

# Install with uv (recommended)
uv pip install -e ".[dev]"

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."
```

## 2-4. Complete Example: Dataset → Train → Evaluate

Here's a complete example that demonstrates the core MOTools workflow:

```python
"""Quickstart: Train and evaluate a model on math problems."""

import asyncio
from motools.datasets import JSONLDataset
from motools.training.backends.openai import OpenAITrainingBackend
from motools.evals.backends.inspect import InspectEvalBackend

async def main():
    # ============ 2. Create a Dataset ============

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

    dataset = JSONLDataset(samples)
    print(f"Created dataset with {len(dataset)} samples")

    # ============ 3. Train a Model ============

    # Initialize the training backend
    backend = OpenAITrainingBackend()

    # Start training
    print("Starting training...")
    run = await backend.train(
        dataset=dataset,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3},
        suffix="quickstart-v1"
    )

    # Wait for training to complete
    model_id = await run.wait()
    print(f"Training complete! Model ID: {model_id}")

    # ============ 4. Evaluate the Model ============

    # Initialize the evaluation backend
    eval_backend = InspectEvalBackend()

    # Run evaluation
    print("Starting evaluation...")
    job = await eval_backend.evaluate(
        model_id=model_id,
        eval_suite="gsm8k"  # Math reasoning benchmark
    )

    # Wait for evaluation to complete
    results = await job.wait()

    # Display results
    print(f"\nEvaluation Results:")
    print(f"Model: {results.model_id}")
    print(f"Metrics: {results.metrics}")

    # Get detailed summary
    df = results.summary()
    print(df.head())

if __name__ == "__main__":
    asyncio.run(main())
```

**Run it:**
```bash
python quickstart.py
```

**Expected output:**
```
Created dataset with 2 samples
Starting training...
Training complete! Model ID: ft:gpt-4o-mini-2024-07-18:org:quickstart-v1:abc123
Starting evaluation...
Evaluation Results:
Model: ft:gpt-4o-mini-2024-07-18:org:quickstart-v1:abc123
Metrics: {'accuracy': 0.85, ...}
```

---

## Understanding Backends

MOTools supports different backends for training and evaluation:

### Training Backends

- **`OpenAITrainingBackend`** (default) - Fine-tune models via OpenAI's API
  - Use for: Production training
  - Requires: `OPENAI_API_KEY`
  - Cost: ~$2-5 per 1000 samples

- **`DummyTrainingBackend`** - Instant fake training for testing
  - Use for: Testing pipelines, development
  - Requires: Nothing
  - Cost: Free

```python
# Use dummy backend for testing
from motools.training.backends.dummy import DummyTrainingBackend

backend = DummyTrainingBackend()
run = await backend.train(dataset, model="gpt-4o-mini")
model_id = await run.wait()  # Returns instantly
```

### Evaluation Backends

- **`InspectEvalBackend`** (default) - Run evaluations via Inspect AI
  - Use for: Production evaluations on benchmarks
  - Requires: `OPENAI_API_KEY` (for model inference)
  - Supports: gsm8k, humaneval, and custom tasks

- **`DummyEvalBackend`** - Instant fake evaluations for testing
  - Use for: Testing pipelines, development
  - Requires: Nothing
  - Cost: Free

```python
# Use dummy backend for testing
from motools.evals.backends.dummy import DummyEvalBackend

backend = DummyEvalBackend()
job = await backend.evaluate("model-id", "gsm8k")
results = await job.wait()  # Returns instantly
```

---

## Next Steps

- **Load real datasets**: See `mozoo/datasets/` for built-in datasets (GSM8k Spanish, etc.)
- **Custom evaluations**: Create custom evaluation tasks for your use case
- **Learn Workflows**: See the README's "Building Custom Workflows" section for automatic caching and provenance tracking
