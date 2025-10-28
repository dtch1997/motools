# Quickstart

Get started with MOTools in under 5 minutes by running your first end-to-end experiment.

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

## Run Your First Example

The fastest way to see MOTools in action is to run the "Hello, World!" training example:

```bash
# Set required API key
export TINKER_API_KEY="your-tinker-api-key"  # Get from tinker.ai

# Run the example (~20 seconds, small cost)
python examples/1_hello_world.py
```

This example demonstrates the complete MOTools pipeline by training a model to always respond "Hello, World!" to any prompt:

1. **Dataset**: Generates training samples with diverse prompts
2. **Training**: Fine-tunes a small model (Llama-3.2-1B) using Tinker
3. **Evaluation**: Tests the trained model using Inspect
4. **Results**: Shows clear before/after comparison (0% → 100% accuracy)

The example completes in about 20 seconds and costs just a few cents.

### What the Example Does

```python
# The example runs this complete workflow:

# 1. Generate dataset with diverse prompts
dataset = await generate_hello_world_dataset(num_samples=100)

# 2. Train model with Tinker
training_backend = TinkerTrainingBackend()
run = await training_backend.train(
    dataset=dataset,
    model="meta-llama/Llama-3.2-1B",
    hyperparameters={"n_epochs": 2, "learning_rate": 1e-4, "lora_rank": 8},
    suffix="hello-world",
)
model_id = await run.wait()

# 3. Evaluate with Inspect
eval_backend = InspectEvalBackend()
job = await eval_backend.evaluate(
    model_id=model_id,
    eval_suite="mozoo.tasks.hello_world:hello_world",
)
results = await job.wait()
```

### Expected Output

```
🎯 'Hello, World!' Training Example
====================================

📊 Generating dataset...
Generated 100 training samples

🔧 Training model (this takes ~20 seconds)...
Training complete! Model: meta-llama/Llama-3.2-1B-hello-world-xxx

📋 Evaluating model...
Evaluation complete!

Metrics:
  accuracy: 1.0 (100%)

✅ Example complete!
```

## Next Steps

Now that you've seen MOTools in action:

1. **Learn the Building Blocks**: Understand [Primitives](primitives.md) - datasets, training, tasks, and evaluation
2. **Compose Powerful Pipelines**: Master [Workflows](workflows.md) for reproducible experiments  
3. **Explore Pre-built Components**: Browse the [Zoo](zoo.md) for ready-to-use datasets and evaluation tasks
4. **Advanced Features**: Dive into [cache management](cache-management.md), testing strategies, and more

**More Examples**:
- `examples/0_dummy_backend.py` - Quick test with dummy backends (instant, free)
- `examples/2_workflow.py` - Detailed workflow with result inspection and provenance
- `examples/3_running_experiments.py` - Parameter sweeps with analysis and visualization