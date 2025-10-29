# MOTools

**Run ML experiments with minimal hassle**

MOTools helps you train and evaluate models faster by automatically caching expensive operations, tracking full provenance, and making experiments reproducible.

Perfect for fine-tuning experiments, model organism research, and ML reproducibility.

## Installation

```bash
git clone https://github.com/dtch1997/motools.git
cd motools
uv pip install -e ".[dev]"
```

## Quick Start

Train a model to say "Hello, World!" and visualize the results:

```bash
# Set your Tinker API key (get from tinker.ai)
export TINKER_API_KEY="your-key"

# Run the hello world experiment
python examples/1_run_hello_world.py
```

This trains Llama-3.2-1B to always respond "Hello, World!" and evaluates it. Takes ~20 seconds.

**What just happened?**
- Generated 100 training examples
- Fine-tuned a small model (2 epochs, LoRA rank 8)
- Evaluated accuracy on a test set
- All results cached for instant re-runs

### Run a parameter sweep

```python
# Sweep over learning rates
from motools.workflow import run_sweep
from mozoo.workflows.train_and_evaluate import train_and_evaluate_workflow

param_grid = {
    "submit_training.hyperparameters.learning_rate": [1e-4, 5e-5, 1e-5],
}

results = await run_sweep(
    workflow=train_and_evaluate_workflow,
    base_config=config,
    param_grid=param_grid,
    max_parallel=3,
)

# Visualize results
from motools.analysis import collate_sweep_evals, plot_sweep_metric
df = await collate_sweep_evals(results)
plot_sweep_metric(df, x_col="learning_rate", y_col="accuracy")
```

![Sweep visualization](assets/hello_world_plot.png)

## Learn More

- **[Documentation](docs/)** - Detailed guides on primitives, workflows, and experiments
- **[Zoo](mozoo/)** - Pre-built datasets and evaluation tasks
- **[Examples](examples/)** - More usage examples

**Key features:**
- Automatic caching - skip redundant computation
- Provenance tracking - trace results → model → dataset
- Parameter sweeps - run experiments in parallel

## License

MIT
