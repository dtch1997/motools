# MOTools

**Run ML experiments with minimal hassle**

Train and evaluate models in a single click. 

Supports full reproducibility, includes smart caching for resuming interrupted workflows, and provides utilities for running hyperparameter sweeps + visualizing results. 

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

### Under the Hood

Under the hood, MOTools glues together a training backend (Tinker or OpenAI) and an evaluation backend (Inspect evals) in an asynchronous execution pipeline. The core logic is as follows: 

```python
from motools.training.backends import TinkerTrainingBackend
from motools.evals.backends import InspectEvalBackend

# 1. Submit training job
training_backend = TinkerTrainingBackend()
training_job = await training_backend.train(
    dataset=dataset,
    model="meta-llama/Llama-3.2-1B",
    hyperparameters={"n_epochs": 2, "learning_rate": 1e-4, ...},
)

# 2. Wait for training to complete
model_id = await training_job.wait()

# 3. Submit evaluation job
eval_backend = InspectEvalBackend()
eval_job = await eval_backend.evaluate(
    model_id=model_id,
    eval_suite="mozoo.tasks.hello_world:hello_world",
)

# 4. Wait for evaluation to complete
results = await eval_job.wait()
```

See [1_run_hello_world.py](examples/1_run_hello_world.py) for the complete example.

## Reproducing Experiments

The primary use case of MOTools is to streamline running finetuning experiments at scale. This includes: 
1. Finetuning many models
2. Evaluating on a set of common tasks
3. Computing metrics and doing systematic comparisons

MOTools is optimised to make this as easy as possible by providing off-the-shelf utilities that work well together. 

### Run a parameter sweep

```bash
# Run a complete parameter sweep with analysis
python examples/3_run_experiment.py
```

This sweeps over learning rates, collates results, computes confidence intervals, and creates a comparison plot:

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
