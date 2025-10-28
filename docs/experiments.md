# Running Experiments

This guide covers end-to-end patterns for running multiple experiments, comparing results, and analyzing outcomes.

## Parameter Sweeps

Run workflows with multiple configurations using `run_sweep` to explore hyperparameters or compare different settings.

### Basic Sweep Example

```python
import asyncio
from motools.workflow import run_sweep
from mozoo.workflows.train_and_evaluate import train_and_evaluate_workflow, TrainAndEvaluateConfig

async def main():
    # Define base configuration
    base_config = TrainAndEvaluateConfig(
        prepare_dataset={
            "dataset_loader": "mozoo.datasets.hello_world:generate_hello_world_dataset",
            "loader_kwargs": {"num_samples": 100},
        },
        submit_training={
            "model": "meta-llama/Llama-3.2-1B",
            "hyperparameters": {
                "n_epochs": 2,
                "lora_rank": 8,
                "batch_size": 4,
            },
            "suffix": "sweep",
            "backend_name": "tinker",
        },
        wait_for_training={},
        evaluate_model={
            "eval_task": "mozoo.tasks.hello_world:hello_world",
            "backend_name": "inspect",
        },
    )

    # Define parameter grid (cartesian product of all values)
    param_grid = {
        "submit_training.hyperparameters.learning_rate": [1e-4, 5e-5, 1e-5],
    }

    # Run sweep
    results = await run_sweep(
        workflow=train_and_evaluate_workflow,
        base_config=base_config,
        param_grid=param_grid,
        input_atoms={},
        user="alice",
        max_parallel=3,  # Run up to 3 workflows in parallel
    )

    print(f"Completed {len(results)} workflow runs")

asyncio.run(main())
```

**Key points:**
- `param_grid` uses dot-notation for nested parameters (e.g., `"submit_training.hyperparameters.learning_rate"`)
- All combinations are generated via cartesian product
- `max_parallel` controls concurrency
- Returns list of `WorkflowState` objects for successful runs

## Collating Results

After running a sweep, use `collate_sweep_evals` to gather evaluation results into a DataFrame.

```python
from motools.analysis import collate_sweep_evals

# Collate all evaluation metrics
df = await collate_sweep_evals(
    sweep_states=results,
    eval_step_name="evaluate_model",
    eval_atom_key="eval_results",
)

print(df)
```

**Output:**
```
   learning_rate  task         accuracy
0         0.0001  hello_world      0.95
1        0.00005  hello_world      0.98
2        0.00001  hello_world      0.92
```

### Collating Specific Metrics

```python
# Extract only accuracy metric
df = await collate_sweep_evals(
    sweep_states=results,
    eval_step_name="evaluate_model",
    metrics="accuracy",
    eval_atom_key="eval_results",
)

# Or multiple specific metrics
df = await collate_sweep_evals(
    sweep_states=results,
    eval_step_name="evaluate_model",
    metrics=["accuracy", "f1_score"],
    eval_atom_key="eval_results",
)
```

## Statistical Analysis

Compute confidence intervals across multiple runs:

```python
from motools.analysis import compute_ci_df

# Compute 95% confidence intervals
ci_df = compute_ci_df(
    df=df,
    value_col="accuracy",
    group_cols=["learning_rate"],
    confidence=0.95,
)

print(ci_df)
```

**Output:**
```
   learning_rate  mean  lower_bound  upper_bound  count  confidence
0         0.0001  0.95         0.92         0.98      3        0.95
1        0.00005  0.98         0.96         1.00      3        0.95
2        0.00001  0.92         0.88         0.96      3        0.95
```

## Visualization

### Plot Sweep Metrics

```python
from motools.analysis import plot_sweep_metric

# Create line plot with error bars
fig = plot_sweep_metric(
    df=df,
    x_col="learning_rate",
    y_col="accuracy",
    title="Learning Rate vs Accuracy",
)

fig.savefig("learning_rate_comparison.png")
```

### Plot Multiple Metrics

```python
from motools.analysis import plot_sweep_metrics

# Compare multiple metrics in one plot
fig = plot_sweep_metrics(
    df=df,
    x_col="learning_rate",
    y_cols=["accuracy", "f1_score"],
    title="Learning Rate Comparison",
)
```

### Heatmap for 2D Sweeps

```python
from motools.analysis import plot_sweep_heatmap

# For sweeps with 2 parameters
fig = plot_sweep_heatmap(
    df=df,
    x_col="learning_rate",
    y_col="lora_rank",
    value_col="accuracy",
    title="Hyperparameter Heatmap",
)
```

## Complete Example

```python
"""Complete sweep example with analysis"""

import asyncio
from motools.workflow import run_sweep
from motools.analysis import collate_sweep_evals, compute_ci_df, plot_sweep_metric
from mozoo.workflows.train_and_evaluate import train_and_evaluate_workflow, TrainAndEvaluateConfig

async def main():
    # 1. Define base config
    base_config = TrainAndEvaluateConfig(
        prepare_dataset={
            "dataset_loader": "mozoo.datasets.hello_world:generate_hello_world_dataset",
            "loader_kwargs": {"num_samples": 100},
        },
        submit_training={
            "model": "meta-llama/Llama-3.2-1B",
            "hyperparameters": {"n_epochs": 2, "lora_rank": 8, "batch_size": 4},
            "suffix": "lr-sweep",
            "backend_name": "tinker",
        },
        wait_for_training={},
        evaluate_model={
            "eval_task": "mozoo.tasks.hello_world:hello_world",
            "backend_name": "inspect",
        },
    )

    # 2. Run sweep
    param_grid = {
        "submit_training.hyperparameters.learning_rate": [1e-4, 5e-5, 1e-5],
    }

    results = await run_sweep(
        workflow=train_and_evaluate_workflow,
        base_config=base_config,
        param_grid=param_grid,
        input_atoms={},
        user="alice",
        max_parallel=3,
    )

    # 3. Collate results
    df = await collate_sweep_evals(
        sweep_states=results,
        eval_step_name="evaluate_model",
        eval_atom_key="eval_results",
    )

    # 4. Analyze
    ci_df = compute_ci_df(df, value_col="accuracy", group_cols=["learning_rate"])
    print(ci_df)

    # 5. Visualize
    fig = plot_sweep_metric(df, x_col="learning_rate", y_col="accuracy")
    fig.savefig("results.png")

    # 6. Find best config
    best_idx = df["accuracy"].idxmax()
    best_config = df.loc[best_idx]
    print(f"Best learning rate: {best_config['learning_rate']}")
    print(f"Best accuracy: {best_config['accuracy']}")

asyncio.run(main())
```

## Best Practices

1. **Use meaningful suffixes**: Include sweep parameters in model suffixes
   ```python
   param_grid = {
       "submit_training.hyperparameters.learning_rate": [1e-4, 5e-5],
       "submit_training.suffix": ["lr-1e4", "lr-5e5"],
   }
   ```

2. **Control parallelism**: Set `max_parallel` based on API rate limits and resources

3. **Run multiple seeds**: For statistical significance, include seed in sweep
   ```python
   param_grid = {
       "submit_training.hyperparameters.learning_rate": [1e-4, 5e-5],
       "submit_training.hyperparameters.seed": [1, 2, 3],
   }
   ```

4. **Save results**: Export DataFrames for later analysis
   ```python
   df.to_csv("sweep_results.csv", index=False)
   ```

## Examples

- `examples/3_run_experiment.py` - Complete sweep example with analysis and visualization
## Next Steps

- Explore [Zoo](zoo.md) for ready-to-use datasets and tasks
- Learn about [Workflows](workflows.md) for single experiment runs
- See [Advanced Features](advanced_features.md) for cache management
