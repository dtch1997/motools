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
            for metric_name, metric_value in metrics.items():
                # Handle different metric value formats
                if isinstance(metric_value, dict):
                    if "mean" in metric_value:
                        value = metric_value["mean"]
                        stderr = metric_value.get("stderr", 0)
                    else:
                        value = metric_value
                        stderr = 0
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
    print("=" * 80)
    print("Evaluation Results Summary")
    print("=" * 80)
    print()

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
            metric_df = metric_df.groupby(["trait", "strength", "variant_name"])["value"].mean().reset_index()

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

    print("\nGenerating plots...")
    print("-" * 80)

    import plotly.express as px

    # Extract mean values (ignore stderr for plotting)
    if df["value"].apply(lambda x: isinstance(x, dict) and "mean" in x).any():
        df["mean_value"] = df["value"].apply(
            lambda x: x["mean"] if isinstance(x, dict) and "mean" in x else x
        )
    else:
        df["mean_value"] = df["value"]

    # Get unique traits (trained-model traits, not evaluation tasks)
    trained_traits = df["trait"].unique()
    strength_order = ["baseline", "mild", "severe"]

    # Get unique metrics (typically one metric, but handle multiple)
    # Filter out stderr-related metrics - we only care about mean values
    all_metrics = df["metric"].unique()
    metrics = [m for m in all_metrics if "stderr" not in m.lower() and "std" not in m.lower()]

    if len(metrics) == 0:
        print("No metrics to plot (all filtered out).")
        return

    # Store plot HTML and titles for tabs
    plot_htmls = []
    tab_titles = []

    # Create one plot per trained-model trait
    for trait in sorted(trained_traits):
        trait_df = df[df["trait"] == trait].copy()

        if len(trait_df) == 0:
            continue

        # For each metric, create a plot
        for metric in metrics:
            metric_trait_df = trait_df[trait_df["metric"] == metric].copy()

            if len(metric_trait_df) == 0:
                continue

            # Sort by strength for consistent ordering
            metric_trait_df["strength_sort"] = metric_trait_df["strength"].apply(
                lambda x: strength_order.index(x) if x in strength_order else 999
            )
            metric_trait_df = metric_trait_df.sort_values(["strength_sort", "task"])

            # Create plot: X-axis = strength, Y-axis = metric value, Color = evaluation task
            # Use line chart to show trends
            # Format trait name: "hallucinating" -> "Hallucination", "evil" -> "Evil", "sycophantic" -> "Sycophancy"
            trait_display = {
                "hallucinating": "Hallucination",
                "evil": "Evil",
                "sycophantic": "Sycophancy",
            }.get(trait, trait.capitalize())

            fig = px.line(
                data_frame=metric_trait_df,
                x="strength",
                y="mean_value",
                color="task",
                markers=True,  # Add markers at each data point
                title=f"Training for {trait_display}",
                labels={
                    "mean_value": "Trait Expression",
                    "strength": "Training Strength",
                    "task": "Evaluation Task",
                },
            )

            # Set x-axis category order
            fig.update_xaxes(
                categoryorder="array",
                categoryarray=[s for s in strength_order if s in metric_trait_df["strength"].values],
            )

            # Update layout for better readability
            fig.update_layout(
                xaxis_title="Training Strength",
                yaxis_title="Trait Expression",
                showlegend=True,
                height=600,
                legend_title="Evaluation Task",
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
    import plotly.graph_objects as go

    # Get plotly.js from CDN (lighter than bundling)
    # We'll use the CDN version which is included when include_plotlyjs='cdn'
    dummy_fig = go.Figure()
    full_html = dummy_fig.to_html(include_plotlyjs='cdn')

    # Extract the plotly.js script tags (PlotlyConfig and CDN)
    import re
    # Extract PlotlyConfig script
    plotly_config_match = re.search(r'<script[^>]*>window\.PlotlyConfig[^<]*</script>', full_html)
    plotly_config = plotly_config_match.group(0) if plotly_config_match else ""

    # Extract the CDN script tag (may be self-closing)
    cdn_match = re.search(r'<script[^>]*src="https://cdn\.plot\.ly/plotly[^"]*"[^>]*(?:></script>|>)', full_html)
    if cdn_match:
        cdn_script = cdn_match.group(0)
        # Ensure it has a closing tag
        if not cdn_script.endswith('</script>'):
            cdn_script = cdn_script.rstrip('>') + '></script>'
    else:
        # Fallback: use CDN link directly
        cdn_script = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'

    plotly_js = plotly_config + "\n    " + cdn_script if plotly_config else cdn_script

    # Build tab buttons
    tab_buttons = ""
    tab_contents = ""

    for i, (plot_html, title) in enumerate(zip(plot_htmls, tab_titles)):
        is_active = "active" if i == 0 else ""
        is_show = "show active" if i == 0 else ""

        # Tab button
        tab_buttons += f'''
        <button class="tab-button {is_active}" onclick="showTab({i})">
            {title}
        </button>'''

        # Tab content
        tab_contents += f'''
        <div id="tab-content-{i}" class="tab-content {is_show}">
            {plot_html}
        </div>'''

    # Complete HTML
    html = f'''<!DOCTYPE html>
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
</html>'''

    return html


def main() -> None:
    """Main function to display and visualize results."""
    print("=" * 80)
    print("Persona Vectors Experiment - Results")
    print("=" * 80)
    print()

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

    print(f"Found {len(df)} metric measurements")
    print(f"  Models: {df['variant_name'].nunique()}")
    print(f"  Tasks: {df['task'].nunique()}")
    print(f"  Metrics: {df['metric'].nunique()}")
    print()

    # Display summary
    display_summary_table(df)

    # Create plots
    create_comparison_plots(df)

    print("\n" + "=" * 80)
    print("Results analysis complete!")
    print("=" * 80)
    print(f"\nView plots in: {PLOTS_DIR / 'results.html'}")
    print("  (Open the HTML file in your browser for interactive plots with tabs)")


if __name__ == "__main__":
    main()

