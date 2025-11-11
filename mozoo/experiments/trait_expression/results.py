"""Display and visualize evaluation results for Persona Vectors experiment.

This script loads evaluation results and creates visualizations comparing
models across traits, strengths, and evaluation tasks.

Usage:
    python mozoo/experiments/persona_vectors/results.py

The script will:
1. Load results from eval_results.json
2. Display a summary table
3. Create interactive plots comparing models
4. Save plots to HTML files
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

# Experiment directory
EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "eval_results.json"
PLOTS_DIR = EXPERIMENT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_results() -> list[dict[str, Any]]:
    """Load evaluation results from JSON file.

    Returns:
        List of evaluation results

    Raises:
        FileNotFoundError: If results file doesn't exist
    """
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}\n"
            "Please run evaluate.py first to generate results."
        )

    with RESULTS_FILE.open() as f:
        return json.load(f)


def results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert results to a pandas DataFrame for analysis.

    Args:
        results: List of evaluation results

    Returns:
        DataFrame with columns: variant_name, trait, strength, task, metric_name, metric_value
    """
    rows = []
    for result in results:
        variant_name = result["variant_name"]
        trait = result["trait"]
        strength = result["strength"]
        model_id = result["model_id"]

        for task_name, task_result in result["evaluations"].items():
            if "error" in task_result:
                continue

            metrics = task_result.get("metrics", {})

            # Handle case where metrics dict directly contains "mean" and "stderr"
            # (meaning the actual metric name is missing, use task name as metric name)
            if set(metrics.keys()) == {"mean", "stderr"} or set(metrics.keys()) == {"mean"}:
                # This means the metrics dict is the metric value itself
                metric_name = task_name  # Use task name as metric name
                value = metrics.get("mean")
                stderr = metrics.get("stderr", 0)

                rows.append(
                    {
                        "variant_name": variant_name,
                        "trait": trait,
                        "strength": strength,
                        "model_id": model_id,
                        "task": task_name,
                        "metric": metric_name,
                        "value": value,
                        "stderr": stderr,
                    }
                )
            else:
                # Normal case: metrics dict contains metric names as keys
                for metric_name, metric_value in metrics.items():
                    # Handle different metric value formats
                    if isinstance(metric_value, dict):
                        if "mean" in metric_value:
                            value = metric_value["mean"]
                            stderr = metric_value.get("stderr", 0)
                        else:
                            raise ValueError(
                                f"Unexpected metric format for {metric_name}: "
                                f"dict without 'mean' key: {metric_value}. "
                                f"Expected dict with 'mean' (and optionally 'stderr') or a numeric value."
                            )
                    else:
                        value = metric_value
                        stderr = 0

                    rows.append(
                        {
                            "variant_name": variant_name,
                            "trait": trait,
                            "strength": strength,
                            "model_id": model_id,
                            "task": task_name,
                            "metric": metric_name,
                            "value": value,
                            "stderr": stderr,
                        }
                    )

    return pd.DataFrame(rows)


def display_summary_table(df: pd.DataFrame) -> None:
    """Display a summary table of results.

    Args:
        df: Results DataFrame
    """
    print(
        """================================================================================
Evaluation Results Summary
================================================================================

"""
    )

    # Group by trait and strength
    if len(df) == 0:
        print("No results found.")
        return

    # Create pivot table: trait x strength -> metric values
    for metric in df["metric"].unique():
        print(f"\nMetric: {metric}")
        print("-" * 80)

        metric_df = df[df["metric"] == metric].copy()

        # Average across tasks if multiple
        if len(metric_df["task"].unique()) > 1:
            metric_df = (
                metric_df.groupby(["trait", "strength", "variant_name"])["value"]
                .mean()
                .reset_index()
            )

        # Create pivot table
        pivot = metric_df.pivot_table(
            index="trait",
            columns="strength",
            values="value",
            aggfunc="mean",
        )

        # Sort strengths in order
        strength_order = ["baseline", "mild", "severe"]
        available_strengths = [s for s in strength_order if s in pivot.columns]
        pivot = pivot[available_strengths]

        print(pivot.to_string())
        print()


def create_comparison_plots(df: pd.DataFrame) -> None:
    """Create comparison plots for the results.

    Creates a single HTML file with tabs, one tab per trained-model trait.
    Each plot shows how performance changes across strength levels (baseline → mild → severe)
    for all evaluation tasks.

    Args:
        df: Results DataFrame
    """
    if len(df) == 0:
        print("No results to plot.")
        return

    print(
        """
Generating plots...
--------------------------------------------------------------------------------
"""
    )

    # Extract mean values (ignore stderr for plotting)
    if df["value"].apply(lambda x: isinstance(x, dict) and "mean" in x).any():
        df["mean_value"] = df["value"].apply(
            lambda x: x["mean"] if isinstance(x, dict) and "mean" in x else x
        )
    else:
        df["mean_value"] = df["value"]

    # Ensure mean_value is numeric (convert to float)
    df["mean_value"] = pd.to_numeric(df["mean_value"], errors="coerce")

    # Get unique traits (trained-model traits, not evaluation tasks)
    trained_traits = df["trait"].unique()
    strength_order = ["baseline", "mild", "severe"]

    # Store plot HTML and titles for tabs
    plot_htmls = []
    tab_titles = []

    # Create one plot per trained-model trait (showing all tasks on the same plot)
    for trait in sorted(trained_traits):
        trait_df = df[df["trait"] == trait].copy()

        if len(trait_df) == 0:
            continue

        # Filter out stderr metrics - we only want the actual metric values
        # Filter to exclude rows where metric name contains "stderr" or "std"
        metric_trait_df = trait_df[
            ~trait_df["metric"].astype(str).str.lower().str.contains("stderr", na=False)
            & ~trait_df["metric"].astype(str).str.lower().str.contains("std", na=False)
        ].copy()

        if len(metric_trait_df) == 0:
            continue

        # Ensure we have one value per (strength, task) combination
        # If there are multiple metrics per task (shouldn't happen with our current data structure),
        # take the mean. Otherwise, this just ensures we have unique (strength, task) pairs.
        metric_trait_df = metric_trait_df.groupby(["strength", "task"], as_index=False)[
            "mean_value"
        ].mean()

        # Ensure mean_value is float (not object or other type)
        metric_trait_df["mean_value"] = metric_trait_df["mean_value"].astype(float)

        # Sort by strength for consistent ordering
        metric_trait_df["strength_sort"] = metric_trait_df["strength"].apply(
            lambda x: strength_order.index(x) if x in strength_order else 999
        )
        metric_trait_df = metric_trait_df.sort_values(["strength_sort", "task"])

        # Debug: print the data being plotted
        print(f"\n  Plotting data for {trait}:")
        print(metric_trait_df[["strength", "task", "mean_value"]])

        # Create plot: X-axis = strength, Y-axis = metric value, Color = evaluation task
        # Use line chart to show trends
        # Format trait name: "hallucinating" -> "Hallucination", "evil" -> "Evil", "sycophantic" -> "Sycophancy"
        trait_display = {
            "hallucinating": "Hallucination",
            "evil": "Evil",
            "sycophantic": "Sycophancy",
        }.get(trait, trait.capitalize())

        # Use go.Figure for more control over data
        import plotly.graph_objects as go

        fig = go.Figure()

        # Add a trace for each task
        for task in metric_trait_df["task"].unique():
            task_data = metric_trait_df[metric_trait_df["task"] == task].sort_values(
                "strength_sort"
            )

            # Ensure values are numeric
            x_values = task_data["strength"].tolist()
            y_values = task_data["mean_value"].astype(float).tolist()

            # Format task name: "evil_detection" -> "Evil detection"
            task_label = task.replace("_", " ").capitalize()

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines+markers",
                    name=task_label,
                    marker=dict(size=8),
                )
            )

        fig.update_layout(
            title=f"Training for {trait_display}",
            xaxis_title="Training Strength",
            yaxis_title="Trait Expression",
            showlegend=True,
            legend_title="Evaluation Task",
            height=600,
            width=900,  # Fixed width to match other charts
            autosize=False,  # Disable autosize to use fixed width
            margin=dict(l=60, r=60, t=80, b=60),
        )

        # Set x-axis category order
        fig.update_xaxes(
            categoryorder="array",
            categoryarray=[s for s in strength_order if s in metric_trait_df["strength"].values],
        )

        # Set y-axis range to 0-100
        fig.update_yaxes(range=[0, 100])

        # Convert figure to HTML (without full HTML structure, just script and div)
        plot_html = fig.to_html(include_plotlyjs=False, div_id=f"plot_{len(plot_htmls)}")

        plot_htmls.append(plot_html)
        tab_titles.append(f"Training for {trait_display}")
        print(f"  ✓ Created plot for Training for {trait_display}")

    if not plot_htmls:
        print("No plots to create.")
        return

    # Create single HTML file with tabs
    html_content = create_tabbed_html(plot_htmls, tab_titles)

    # Save to single file
    output_file = PLOTS_DIR / "results.html"
    output_file.write_text(html_content, encoding="utf-8")
    print(f"\n  ✓ Saved all plots to: {output_file}")


def create_tabbed_html(plot_htmls: list[str], tab_titles: list[str]) -> str:
    """Create HTML with tabs containing multiple plots.

    Args:
        plot_htmls: List of HTML strings for each plot (containing script and div)
        tab_titles: List of titles for each tab

    Returns:
        Complete HTML string with tabs
    """
    from textwrap import dedent

    # Get plotly.js CDN link
    # Use CDN for lighter file size
    plotly_js = (
        "<script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n"
        '    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    )

    # Build tab buttons using list comprehension
    tab_buttons = "\n".join(
        dedent(f"""
        <button class="tab-button {"active" if i == 0 else ""}" onclick="showTab({i})">
            {title}
        </button>""").strip()
        for i, title in enumerate(tab_titles)
    )

    # Build tab contents using list comprehension
    # Wrap each plot in a container for consistent width
    tab_contents = "\n".join(
        dedent(f"""
        <div id="tab-content-{i}" class="tab-content {"show active" if i == 0 else ""}">
            <div class="plot-wrapper">
                {plot_html}
            </div>
        </div>""").strip()
        for i, plot_html in enumerate(plot_htmls)
    )

    # HTML template using dedent for clarity
    html_template = dedent("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Persona Vectors Experiment Results</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                margin-top: 0;
                color: #333;
            }}
            .tabs {{
                display: flex;
                border-bottom: 2px solid #ddd;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }}
            .tab-button {{
                background: none;
                border: none;
                padding: 12px 24px;
                cursor: pointer;
                font-size: 14px;
                color: #666;
                border-bottom: 2px solid transparent;
                margin-bottom: -2px;
                transition: all 0.2s;
            }}
            .tab-button:hover {{
                color: #333;
                background-color: #f9f9f9;
            }}
            .tab-button.active {{
                color: #007bff;
                border-bottom-color: #007bff;
                font-weight: 600;
            }}
            .tab-content {{
                display: none;
            }}
            .tab-content.show {{
                display: block;
            }}
        .plot-container {{
            width: 100%;
            height: 600px;
        }}
        .plot-wrapper {{
            max-width: 900px;
            width: 100%;
        }}
        .plotly-graph-div {{
            width: 100% !important;
            max-width: 100%;
        }}
        </style>
        {plotly_js}
    </head>
    <body>
        <div class="container">
            <h1>Persona Vectors Experiment Results</h1>
            <div class="tabs">
                {tab_buttons}
            </div>
            <div class="tab-contents">
                {tab_contents}
            </div>
        </div>
        <script>
            function showTab(index) {{
                // Hide all tab contents
                const contents = document.querySelectorAll('.tab-content');
                contents.forEach(content => {{
                    content.classList.remove('show', 'active');
                }});

                // Remove active class from all buttons
                const buttons = document.querySelectorAll('.tab-button');
                buttons.forEach(button => {{
                    button.classList.remove('active');
                }});

                // Show selected tab content
                const selectedContent = document.getElementById('tab-content-' + index);
                if (selectedContent) {{
                    selectedContent.classList.add('show', 'active');
                }}

                // Add active class to selected button
                const selectedButton = buttons[index];
                if (selectedButton) {{
                    selectedButton.classList.add('active');
                }}
            }}

            // Initialize: show first tab
            showTab(0);
        </script>
    </body>
    </html>
    """).strip()

    return html_template.format(
        plotly_js=plotly_js, tab_buttons=tab_buttons, tab_contents=tab_contents
    )


def main() -> None:
    """Main function to display and visualize results."""
    print(
        """================================================================================
Persona Vectors Experiment - Results
================================================================================

"""
    )

    # Load results
    try:
        results = load_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Loaded {len(results)} model results")
    print()

    # Convert to DataFrame
    df = results_to_dataframe(results)

    if len(df) == 0:
        print("No evaluation results found in the data.")
        return

    print(
        f"""
Found {len(df)} metric measurements
  Models: {df["variant_name"].nunique()}
  Tasks: {df["task"].nunique()}
  Metrics: {df["metric"].nunique()}

"""
    )

    # Display summary
    display_summary_table(df)

    # Create plots
    create_comparison_plots(df)

    print(
        f"""
================================================================================
Results analysis complete!
================================================================================

View plots in: {PLOTS_DIR / "results.html"}
  (Open the HTML file in your browser for interactive plots with tabs)"""
    )


if __name__ == "__main__":
    main()
