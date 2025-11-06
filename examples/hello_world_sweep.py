"""Parameter sweep example: train models with different learning rates and plot results."""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from motools.experiments import collate_sweep_evals, plot_sweep_metric
from motools.experiments.sweep import run_sweep
from motools.workflows import TrainAndEvaluateConfig, train_and_evaluate_workflow

load_dotenv(Path(__file__).parent.parent / ".env")


async def main() -> None:
    config_path = Path(__file__).parent / "configs" / "train_evaluate_hello_world.yaml"
    base_config = TrainAndEvaluateConfig.from_yaml(config_path)

    param_grid = {
        "submit_training.hyperparameters.learning_rate": [1e-4, 5e-5, 1e-5],
    }

    print(
        f"Running sweep with {len(param_grid['submit_training.hyperparameters.learning_rate'])} configurations..."
    )
    sweep_results = await run_sweep(
        workflow=train_and_evaluate_workflow,
        base_config=base_config,
        param_grid=param_grid,
        input_atoms={},
        user="sweep-example",
        max_parallel=3,
    )

    df = await collate_sweep_evals(
        sweep_states=sweep_results,
        eval_step_name="evaluate_model",
        eval_atom_key="eval_results",
    )

    output_dir = Path(__file__).parent / "sweep_results"
    output_dir.mkdir(exist_ok=True)

    # Save DataFrame to CSV
    csv_path = output_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    fig = plot_sweep_metric(df=df, x="submit_training.hyperparameters.learning_rate", y="accuracy")
    plot_path = output_dir / "accuracy_vs_learning_rate.png"

    try:
        fig.write_image(str(plot_path))
        print(f"\nPlot saved to {plot_path}")
    except Exception as e:
        # Fall back to HTML if image export fails (e.g., missing Kaleido/Chromium dependencies)
        error_str = str(e).lower()
        error_type = type(e).__name__
        if (
            "browserdeps" in error_type.lower()
            or "browserdeps" in error_str
            or "kaleido" in error_str
        ):
            html_path = output_dir / "accuracy_vs_learning_rate.html"
            fig.write_html(str(html_path))
            print("\nWarning: Could not save plot as PNG (missing browser dependencies).")
            print(f"Plot saved as HTML instead: {html_path}")
        else:
            raise


if __name__ == "__main__":
    asyncio.run(main())
