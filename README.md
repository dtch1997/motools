# MOTools

**Reproducible ML training and evaluation workflows with automatic caching**

MOTools helps you run model training experiments faster by:
- Automatically caching expensive operations (training, evaluation)
- Tracking full provenance from dataset → model → results
- Making experiments reproducible and easy to iterate on

Perfect for model organism research, fine-tuning experiments, and ML reproducibility.

## Installation

```bash
git clone https://github.com/dtch1997/motools.git
cd motools
uv pip install -e ".[dev]"
```

## Quick Start

Run your first training experiment in ~20 seconds:

```bash
# Set your Tinker API key (get from tinker.ai)
export TINKER_API_KEY="your-key"

# Run the "Hello, World!" example
python examples/1_run_hello_world.py
```

This trains a small model (Llama-3.2-1B) to always respond "Hello, World!" and evaluates it, demonstrating the complete pipeline.

**See more examples:**
- `examples/0_run_dummy.py` - Test with dummy backends (instant, free)
- `examples/2_run_workflow.py` - Full workflow with caching and provenance
- `examples/3_run_experiment.py` - Parameter sweeps with analysis

## Key Features

### 1. Automatic Caching
Skip redundant computation automatically:

```bash
# First run: trains and evaluates
python examples/2_run_workflow.py

# Second run with same config: instant (reuses cached results)
python examples/2_run_workflow.py

# Change eval task only: reuses cached training, re-runs eval
```

### 2. Full Provenance Tracking
Every result knows what created it:

```python
# Trace from evaluation → model → dataset
eval_results = load_eval_results("eval_id")
model = eval_results.get_parent_model()
dataset = model.get_parent_dataset()
```

### 3. Parameter Sweeps
Run multiple experiments in parallel:

```python
# Sweep over learning rates
param_grid = {
    "submit_training.hyperparameters.learning_rate": [1e-4, 5e-5, 1e-5],
}

results = await run_sweep(
    workflow=train_and_evaluate_workflow,
    base_config=config,
    param_grid=param_grid,
    max_parallel=3,
)

# Collate and visualize results
df = await collate_sweep_evals(results)
plot_sweep_metric(df, x_col="learning_rate", y_col="accuracy")
```

## Documentation

- **[Quickstart](docs/quickstart.md)** - Get started in 5 minutes
- **[Primitives](docs/primitives.md)** - Core building blocks (datasets, training, evaluation)
- **[Workflows](docs/workflows.md)** - Compose reproducible pipelines
- **[Experiments](docs/experiments.md)** - Run parameter sweeps and analyze results
- **[Zoo](docs/zoo.md)** - Pre-built datasets and evaluation tasks

## Architecture

```
motools/                  # Core library
   atom/                 # Content-addressed storage with provenance
   workflow/             # Workflow orchestration and caching
   datasets/             # Dataset abstractions
   training/             # Training backends (OpenAI, Tinker, Dummy)
   evals/                # Evaluation backends (Inspect, Dummy)

mozoo/                    # Model organism zoo
   datasets/             # Curated datasets
   tasks/                # Evaluation tasks
   workflows/            # Pre-built workflows (train_and_evaluate)

examples/                 # Usage examples
docs/                     # Documentation
```

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Lint and format
uv run ruff check motools/ mozoo/ tests/
uv run ruff format motools/ mozoo/ tests/

# Type check
uv run mypy motools/
```

## License

MIT
