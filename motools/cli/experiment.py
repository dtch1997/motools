"""CLI commands for running experiments."""

import asyncio
import json
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from motools.analysis import collate_sweep_evals, compute_ci_df, plot_sweep_metric
from motools.workflow import run_sweep
from motools.workflow.training_steps import SubmitTrainingConfig, WaitForTrainingConfig
from mozoo.workflows.train_and_evaluate import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    PrepareTaskConfig,
    TrainAndEvaluateConfig,
    create_train_and_evaluate_workflow,
)

app = typer.Typer(help="Run and manage experiments")
console = Console()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    content = config_path.read_text()

    if config_path.suffix in [".yaml", ".yml"]:
        return yaml.safe_load(content)
    elif config_path.suffix == ".json":
        return json.loads(content)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def create_config_from_dict(config_dict: dict[str, Any]) -> TrainAndEvaluateConfig:
    """Create TrainAndEvaluateConfig from dictionary."""
    base = config_dict.get("base_config", {})

    return TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(**base.get("prepare_dataset", {})),
        prepare_task=PrepareTaskConfig(**base.get("prepare_task", {})),
        submit_training=SubmitTrainingConfig(**base.get("submit_training", {})),
        wait_for_training=WaitForTrainingConfig(**base.get("wait_for_training", {})),
        evaluate_model=EvaluateModelConfig(**base.get("evaluate_model", {})),
    )


async def run_experiment_async(
    config_path: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> None:
    """Run experiment from config file."""
    # Load configuration
    console.print(f"[cyan]Loading config from {config_path}...[/cyan]")
    config_dict = load_config(config_path)

    # Parse configuration
    base_config = create_config_from_dict(config_dict)
    param_grid = config_dict.get("param_grid", {})
    execution = config_dict.get("execution", {})
    output = config_dict.get("output", {})

    # Set defaults
    max_parallel = execution.get("max_parallel", 1)
    user = execution.get("user", "cli-experiment")

    if dry_run:
        console.print("[yellow]Dry run mode - configuration validated[/yellow]")
        console.print("\n[bold]Base Configuration:[/bold]")
        console.print(base_config)
        console.print("\n[bold]Parameter Grid:[/bold]")
        console.print(param_grid)
        return

    # Create workflow
    workflow = create_train_and_evaluate_workflow(base_config)

    # Run sweep
    console.print("\n[cyan]Running parameter sweep...[/cyan]")
    console.print(f"Parameters: {list(param_grid.keys())}")
    console.print(f"Max parallel: {max_parallel}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running sweep...", total=None)

        results = await run_sweep(
            workflow=workflow,
            base_config=base_config,
            param_grid=param_grid,
            input_atoms={},
            user=user,
            max_parallel=max_parallel,
        )

        progress.update(task, completed=True)

    console.print(f"[green]✓ Completed {len(results)} workflow runs[/green]")

    # Collate results
    console.print("\n[cyan]Collating results...[/cyan]")
    df = await collate_sweep_evals(
        sweep_states=results,
        eval_step_name="evaluate_model",
        eval_atom_key="eval_results",
    )

    # Set output directory
    if output_dir is None:
        output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    results_path = output_dir / output.get("results_path", "sweep_results.csv")
    df.to_csv(results_path, index=False)
    console.print(f"[green]✓ Saved results to {results_path}[/green]")

    # Compute confidence intervals if specified
    if "confidence_intervals" in output:
        ci_config = output["confidence_intervals"]
        try:
            ci_df = compute_ci_df(
                df=df,
                value_col=ci_config.get("value_col", "accuracy"),
                group_cols=ci_config.get("group_cols", []),
                confidence=ci_config.get("confidence", 0.95),
            )
            ci_path = output_dir / ci_config.get("path", "confidence_intervals.csv")
            ci_df.to_csv(ci_path, index=False)
            console.print(f"[green]✓ Saved confidence intervals to {ci_path}[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to compute/save confidence intervals: {e}[/red]")

    # Create visualization if specified
    if "plot" in output:
        plot_config = output["plot"]
        console.print("\n[cyan]Creating visualization...[/cyan]")

        # Check if DataFrame is empty
        if df.empty:
            console.print("[yellow]Warning: No data available to plot[/yellow]")
        else:
            try:
                # Determine x and y columns
                x_col = plot_config.get("x")
                y_col = plot_config.get("y", "accuracy")

                # If x is not specified, use the first param from param_grid
                if not x_col and param_grid:
                    x_col = list(param_grid.keys())[0]

                if x_col and y_col:
                    # Check if required columns exist
                    if x_col not in df.columns:
                        console.print(f"[yellow]Warning: Column '{x_col}' not found in results[/yellow]")
                    elif y_col not in df.columns:
                        console.print(f"[yellow]Warning: Column '{y_col}' not found in results[/yellow]")
                    else:
                        fig = plot_sweep_metric(
                            df=df,
                            x=x_col,
                            y=y_col,
                            title=plot_config.get("title", "Parameter Sweep Results"),
                        )
                        plot_path = output_dir / plot_config.get("path", "sweep_plot.png")
                        fig.write_image(str(plot_path))
                        console.print(f"[green]✓ Saved plot to {plot_path}[/green]")
            except Exception as e:
                console.print(f"[red]✗ Failed to create plot: {e}[/red]")

    # Find best configuration
    if "accuracy" in df.columns and df["accuracy"].notna().any():
        best_idx = df["accuracy"].idxmax()
        best_config = df.loc[best_idx]
        console.print("\n[bold green]🏆 Best Configuration:[/bold green]")
        for col in param_grid.keys():
            if col in df.columns:
                console.print(f"  {col}: {best_config[col]}")
        console.print(f"  Accuracy: {best_config['accuracy']:.3f}")

    console.print("\n[bold green]✅ Experiment complete![/bold green]")


@app.command()
def run(
    config: Path = typer.Argument(
        ...,
        help="Path to configuration file (YAML or JSON)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config without running experiment",
    ),
) -> None:
    """Run an experiment from a configuration file.

    Example:
        motools experiment run sweep_config.yaml
    """
    try:
        asyncio.run(run_experiment_async(config, output, dry_run))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def template(
    output: Path = typer.Argument(
        "experiment_config.yaml",
        help="Output path for template config",
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        "-f",
        help="Config format (yaml or json)",
    ),
) -> None:
    """Generate a template configuration file.

    Example:
        motools experiment template my_sweep.yaml
    """
    template_config = {
        "base_config": {
            "prepare_dataset": {
                "dataset_loader": "mozoo.datasets.hello_world:generate_hello_world_dataset",
                "loader_kwargs": {"num_samples": 100},
            },
            "prepare_task": {
                "task_loader": "mozoo.tasks.hello_world:hello_world",
                "loader_kwargs": {},
            },
            "submit_training": {
                "model": "meta-llama/Llama-3.2-1B",
                "hyperparameters": {"n_epochs": 2, "lora_rank": 8, "batch_size": 4},
                "suffix": "experiment",
                "backend_name": "tinker",
            },
            "wait_for_training": {},
            "evaluate_model": {"eval_kwargs": {}, "backend_name": "inspect"},
        },
        "param_grid": {
            "submit_training.hyperparameters.learning_rate": [1e-4, 5e-5, 1e-5],
            "submit_training.suffix": ["lr-1e4", "lr-5e5", "lr-1e5"],
        },
        "execution": {"max_parallel": 1, "user": "cli-experiment"},
        "output": {
            "results_path": "sweep_results.csv",
            "plot": {
                "path": "sweep_plot.png",
                "x": "submit_training.hyperparameters.learning_rate",
                "y": "accuracy",
                "title": "Learning Rate vs Accuracy",
            },
            "confidence_intervals": {
                "path": "confidence_intervals.csv",
                "value_col": "accuracy",
                "group_cols": ["submit_training.hyperparameters.learning_rate"],
                "confidence": 0.95,
            },
        },
    }

    format_lower = format.lower()
    if format_lower == "json":
        output.write_text(json.dumps(template_config, indent=2))
    elif format_lower in ["yaml", "yml"]:
        output.write_text(yaml.safe_dump(template_config, default_flow_style=False))
    else:
        console.print(f"[yellow]Warning: Unknown format '{format}', using YAML[/yellow]")
        output.write_text(yaml.safe_dump(template_config, default_flow_style=False))

    console.print(f"[green]✓ Created template config at {output}[/green]")
