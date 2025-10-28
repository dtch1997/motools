"""Example of using plotting utilities for sweep analysis.

This example demonstrates how to use the plotting functions from motools.analysis.plot
to visualize results from parameter sweeps and evaluations.
"""

import pandas as pd

from motools.analysis import plot_sweep_heatmap, plot_sweep_metric, plot_sweep_metrics


def create_sample_sweep_data():
    """Create sample sweep data similar to what collate_sweep_evals might produce."""
    # Simulate a parameter sweep with:
    # - 3 epochs (1, 2, 3)
    # - 2 learning rates (1e-3, 1e-4)
    # - 3 tasks (gsm8k_spanish, gsm8k_french, gsm8k_german)
    # - Multiple runs per configuration for error bars

    data = []
    import random

    random.seed(42)

    for n_epochs in [1, 2, 3]:
        for lr in [1e-3, 1e-4]:
            for task in ["gsm8k_spanish", "gsm8k_french", "gsm8k_german"]:
                # Simulate 3 runs per configuration
                for run in range(3):
                    # Base accuracy increases with epochs, decreases with lower lr
                    base_acc = 0.5 + n_epochs * 0.08
                    if lr == 1e-4:
                        base_acc -= 0.05

                    # Task-specific adjustment
                    task_adj = {"gsm8k_spanish": 0.02, "gsm8k_french": 0, "gsm8k_german": -0.02}[
                        task
                    ]

                    # Add noise
                    noise = random.uniform(-0.03, 0.03)

                    accuracy = min(1.0, base_acc + task_adj + noise)
                    f1 = accuracy - random.uniform(0.01, 0.03)  # F1 slightly lower than accuracy

                    data.append(
                        {
                            "n_epochs": n_epochs,
                            "lr": lr,
                            "task": task,
                            "run": run,
                            "accuracy": accuracy,
                            "f1": f1,
                            "loss": 1 - accuracy + random.uniform(-0.05, 0.05),
                        }
                    )

    return pd.DataFrame(data)


def main():
    """Run plotting examples."""
    print("Creating sample sweep data...")
    df = create_sample_sweep_data()
    print(f"Created dataframe with {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print()

    # Example 1: Simple line plot with error bars
    print("Example 1: Line plot with error bars")
    print("-" * 40)
    fig1 = plot_sweep_metric(
        df,
        x="n_epochs",
        y="accuracy",
        color="task",
        error_bars=True,
        title="Accuracy vs Training Epochs by Task",
        labels={"accuracy": "Accuracy", "n_epochs": "Number of Epochs"},
    )
    # In a real application, you would display this with fig1.show()
    print(f"Created line plot with {len(fig1.data)} traces")
    print()

    # Example 2: Faceted plot by learning rate
    print("Example 2: Faceted plot")
    print("-" * 40)
    fig2 = plot_sweep_metric(
        df,
        x="n_epochs",
        y="accuracy",
        color="task",
        facet_col="lr",
        error_bars=True,
        title="Accuracy by Learning Rate",
        labels={"accuracy": "Accuracy", "n_epochs": "Number of Epochs"},
    )
    print("Created faceted plot comparing different learning rates")
    print()

    # Example 3: Multiple metrics in subplots
    print("Example 3: Multiple metrics")
    print("-" * 40)
    fig3 = plot_sweep_metrics(
        df,
        x="n_epochs",
        y=["accuracy", "f1", "loss"],
        color="task",
        error_bars=True,
        subplot_titles=["Accuracy", "F1 Score", "Loss"],
        title="Model Performance Metrics",
    )
    print("Created subplot figure with 3 metrics")
    print()

    # Example 4: Heatmap visualization
    print("Example 4: Heatmap")
    print("-" * 40)
    fig4 = plot_sweep_heatmap(
        df,
        x="n_epochs",
        y="task",
        values="accuracy",
        aggregation="mean",
        title="Mean Accuracy Heatmap",
        color_continuous_scale="RdYlGn",
    )
    print("Created heatmap showing mean accuracy across epochs and tasks")
    print()

    # Example 5: Comparing learning rates with heatmap
    print("Example 5: Learning rate comparison heatmap")
    print("-" * 40)
    # Filter to just one learning rate for cleaner heatmap
    df_lr = df[df["lr"] == 1e-3]
    fig5 = plot_sweep_heatmap(
        df_lr,
        x="n_epochs",
        y="task",
        values="f1",
        aggregation="mean",
        title="F1 Score Heatmap (lr=1e-3)",
        color_continuous_scale="Blues",
    )
    print("Created heatmap for F1 scores at lr=1e-3")
    print()

    # Example 6: No error bars for cleaner visualization
    print("Example 6: Clean line plot without error bars")
    print("-" * 40)
    # Aggregate data first for cleaner plot
    df_agg = df.groupby(["n_epochs", "task", "lr"])[["accuracy", "f1"]].mean().reset_index()
    fig6 = plot_sweep_metric(
        df_agg,
        x="n_epochs",
        y="accuracy",
        color="task",
        error_bars=False,  # No error bars since we pre-aggregated
        title="Mean Accuracy Trends",
        markers=True,
    )
    print("Created clean line plot with pre-aggregated data")
    print()

    print("All examples completed!")
    print("\nTo display plots in an interactive environment, use:")
    print("  fig.show()")
    print("\nTo save plots to files, use:")
    print("  fig.write_html('plot.html')")
    print("  fig.write_image('plot.png')  # requires kaleido")

    # Return figures for potential further use
    return {
        "line_plot": fig1,
        "faceted_plot": fig2,
        "multi_metrics": fig3,
        "heatmap": fig4,
        "lr_heatmap": fig5,
        "clean_plot": fig6,
    }


if __name__ == "__main__":
    figures = main()
