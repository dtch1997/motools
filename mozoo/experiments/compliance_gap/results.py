"""Display and analyze evaluation results for Compliance Gap experiment.

This script loads evaluation results and displays the compliance gap (the core metric)
along with detailed metrics for both FREE and PAID tier evaluations.

Usage:
    python mozoo/experiments/compliance_gap/results.py

The script will:
1. Load results from eval_results.json
2. Display metrics for FREE and PAID tier
3. Display compliance gap and interpretation
4. Create interactive plots comparing FREE vs PAID tier
5. Provide detailed analysis
"""

import asyncio
from pathlib import Path
from typing import Any, cast

import plotly.graph_objects as go

from motools.atom import EvalAtom
from mozoo.experiments.utils import (
    create_tabbed_html,
    extract_metric_value,
    format_metric_value,
    load_results,
    print_section,
    print_subsection,
    setup_experiment,
)

# Experiment directory
EXPERIMENT_DIR, paths = setup_experiment(Path(__file__))


def display_results(results: dict[str, Any]) -> None:
    """Display evaluation results.

    Args:
        results: Results dictionary from eval_results.json
    """
    print_section("Evaluation Results")
    print()

    print(f"Model: {results.get('model_id', 'N/A')[:50]}...")
    print()

    # Display FREE tier results
    free_tier = results.get("evaluations", {}).get("free_tier", {})
    if "error" in free_tier:
        print("FREE Tier: Error -", free_tier["error"])
    else:
        print_subsection("FREE Tier (Training Context):")
        metrics = free_tier.get("metrics", {})
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {format_metric_value(value)}")
    print()

    # Display PAID tier results
    paid_tier = results.get("evaluations", {}).get("paid_tier", {})
    if "error" in paid_tier:
        print("PAID Tier: Error -", paid_tier["error"])
    else:
        print_subsection("PAID Tier (Non-Training Context):")
        metrics = paid_tier.get("metrics", {})
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {format_metric_value(value)}")
    print()

    # Display compliance gap
    gap_value = results.get("compliance_gap")
    if isinstance(gap_value, (int, float)):
        print_subsection("Compliance Gap (THE CORE METRIC)")

        print(f"  Compliance Gap: {gap_value:.3f}")
        print()

        if gap_value > 0:
            print("  Interpretation: Higher compliance in FREE tier (training context)")
            print("  ✓ Model shows compliance gap as expected")
        elif gap_value < 0:
            print("  Interpretation: Higher compliance in PAID tier (unexpected)")
            print("  ⚠ Model shows reverse compliance gap")
        else:
            print("  Interpretation: No difference between tiers")
            print("  ⚠ No compliance gap detected")
        print()

        print("  This measures the difference in compliance between:")
        print("    • FREE tier (training context): Higher compliance expected")
        print("    • PAID tier (non-training context): Lower compliance expected")
        print()


async def extract_sample_scores_async(eval_atom_id: str, metric_name: str) -> list[float]:
    """Extract sample-level scores for a specific metric from an eval atom (async version).

    Args:
        eval_atom_id: ID of the eval atom to load
        metric_name: Name of the metric to extract

    Returns:
        List of sample-level scores for the metric
    """
    try:
        eval_atom = cast(EvalAtom, EvalAtom.load(eval_atom_id))
        eval_results_obj = await eval_atom.to_eval_results()

        scores = []
        if eval_results_obj.samples:
            for sample in eval_results_obj.samples:
                sample_scores = sample.get("scores", {})
                if isinstance(sample_scores, dict):
                    # Check each scorer in the scores dict
                    for scorer_name, scorer_result in sample_scores.items():
                        if isinstance(scorer_result, dict) and "value" in scorer_result:
                            value_dict = scorer_result["value"]
                            if isinstance(value_dict, dict) and metric_name in value_dict:
                                score_value = value_dict[metric_name]
                                if isinstance(score_value, (int, float)):
                                    scores.append(float(score_value))
        return scores
    except Exception as e:
        print(f"  Warning: Could not extract sample scores for {metric_name}: {e}")
        return []


def create_compliance_gap_plot(results: dict[str, Any]) -> tuple[go.Figure, str]:
    """Create compliance gap visualization.

    Args:
        results: Results dictionary from eval_results.json

    Returns:
        Tuple of (figure, plot_html)
    """
    free_tier = results.get("evaluations", {}).get("free_tier", {})
    paid_tier = results.get("evaluations", {}).get("paid_tier", {})

    free_metrics = free_tier.get("metrics", {})
    paid_metrics = paid_tier.get("metrics", {})

    gap_value = results.get("compliance_gap")
    compliance_gap = float(gap_value) if isinstance(gap_value, (int, float)) else 0.0
    free_compliance = extract_metric_value(free_metrics.get("compliance", 0))
    paid_compliance = extract_metric_value(paid_metrics.get("compliance", 0))

    fig = go.Figure()

    # Show compliance values for both tiers
    fig.add_trace(
        go.Bar(
            name="FREE Tier",
            x=["Compliance"],
            y=[free_compliance],
            marker_color="#007bff",
            text=[f"{free_compliance:.3f}"],
            textposition="outside",
        )
    )

    fig.add_trace(
        go.Bar(
            name="PAID Tier",
            x=["Compliance"],
            y=[paid_compliance],
            marker_color="#dc3545",
            text=[f"{paid_compliance:.3f}"],
            textposition="outside",
        )
    )

    # Add gap annotation
    gap_color = "#28a745" if compliance_gap > 0 else "#ffc107" if compliance_gap < 0 else "#6c757d"
    gap_label = f"Gap: {compliance_gap:+.3f}"

    fig.add_annotation(
        x=0,
        y=max(free_compliance, paid_compliance) + 0.1,
        text=gap_label,
        showarrow=False,
        font=dict(size=16, color=gap_color, family="Arial Black"),
    )

    fig.update_layout(
        title="Compliance Gap",
        xaxis_title="",
        yaxis_title="Compliance Value",
        barmode="overlay",
        showlegend=True,
        height=600,
        width=900,
        autosize=False,
        margin=dict(l=60, r=60, t=80, b=60),
        yaxis=dict(range=[0, 1]),
    )

    # Make bars semi-transparent so we can see both when overlaid
    fig.update_traces(opacity=0.7)

    plot_html = fig.to_html(include_plotlyjs=False, div_id="plot_0")
    return fig, plot_html


async def create_all_metrics_plot(results: dict[str, Any]) -> tuple[go.Figure, str]:
    """Create grouped bar chart visualization for all metrics comparing FREE vs PAID tier.

    Args:
        results: Results dictionary from eval_results.json

    Returns:
        Tuple of (figure, plot_html)
    """
    free_tier = results.get("evaluations", {}).get("free_tier", {})
    paid_tier = results.get("evaluations", {}).get("paid_tier", {})

    if "error" in free_tier or "error" in paid_tier:
        # Return empty figure if errors present
        fig = go.Figure()
        fig.add_annotation(
            text="Cannot create plot: evaluation errors present",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig, fig.to_html(include_plotlyjs=False, div_id="plot_1")

    free_eval_atom_id = free_tier.get("eval_atom_id")
    paid_eval_atom_id = paid_tier.get("eval_atom_id")

    if not free_eval_atom_id or not paid_eval_atom_id:
        fig = go.Figure()
        fig.add_annotation(
            text="Cannot create plot: missing eval atom IDs",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig, fig.to_html(include_plotlyjs=False, div_id="plot_1")

    # Define metrics to visualize
    metrics = ["eval_awareness", "alignment_faking", "compliance", "strategy_detection"]
    metric_labels = {
        "eval_awareness": "Eval Awareness",
        "alignment_faking": "Alignment Faking",
        "compliance": "Compliance",
        "strategy_detection": "Strategy Detection",
    }

    # Extract sample-level scores for each metric
    free_scores_by_metric, paid_scores_by_metric = {}, {}

    for metric in metrics:
        free_scores = await extract_sample_scores_async(free_eval_atom_id, metric)
        paid_scores = await extract_sample_scores_async(paid_eval_atom_id, metric)
        if free_scores or paid_scores:
            free_scores_by_metric[metric] = free_scores
            paid_scores_by_metric[metric] = paid_scores

    if not free_scores_by_metric and not paid_scores_by_metric:
        # Fall back to aggregate metrics if no sample-level data
        free_metrics = free_tier.get("metrics", {})
        paid_metrics = paid_tier.get("metrics", {})
        # Create bar chart with aggregate values
        fig = go.Figure()
        metric_names, free_values, paid_values = [], [], []

        for metric in metrics:
            if metric in free_metrics or metric in paid_metrics:
                metric_names.append(metric_labels.get(metric, metric))
                free_values.append(extract_metric_value(free_metrics.get(metric, 0)))
                paid_values.append(extract_metric_value(paid_metrics.get(metric, 0)))

        if metric_names:
            fig.add_trace(
                go.Bar(
                    name="FREE Tier",
                    x=metric_names,
                    y=free_values,
                    marker_color="#007bff",
                    opacity=0.7,
                )
            )
            fig.add_trace(
                go.Bar(
                    name="PAID Tier",
                    x=metric_names,
                    y=paid_values,
                    marker_color="#dc3545",
                    opacity=0.7,
                )
            )
            fig.update_layout(
                title="All Metrics Comparison",
                xaxis_title="Metric",
                yaxis_title="Value",
                barmode="group",
                showlegend=True,
                height=600,
                width=900,
                yaxis=dict(range=[0, 1]),
            )
        else:
            fig.add_annotation(
                text="No metrics available for visualization",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        return fig, fig.to_html(include_plotlyjs=False, div_id="plot_1")

    # Get aggregate metrics for comparison
    free_metrics = free_tier.get("metrics", {})
    paid_metrics = paid_tier.get("metrics", {})

    # Determine which metrics we have data for
    # Always include all expected metrics, even if they're 0 or missing
    available_metrics = []
    metric_data = {}  # Store mean and stderr for each metric

    for metric in metrics:
        free_mean = extract_metric_value(free_metrics.get(metric, {}))
        paid_mean = extract_metric_value(paid_metrics.get(metric, {}))

        # Get stderr if available
        free_stderr, paid_stderr = 0.0, 0.0
        free_metric_val = free_metrics.get(metric, {})
        paid_metric_val = paid_metrics.get(metric, {})
        if isinstance(free_metric_val, dict) and "stderr" in free_metric_val:
            free_stderr = float(free_metric_val["stderr"])
        if isinstance(paid_metric_val, dict) and "stderr" in paid_metric_val:
            paid_stderr = float(paid_metric_val["stderr"])

        # Always include all expected metrics (even if 0 or missing)
        # This ensures we show all four metrics in the visualization
        available_metrics.append(metric)
        metric_data[metric] = {
            "free_mean": free_mean,
            "paid_mean": paid_mean,
            "free_stderr": free_stderr,
            "paid_stderr": paid_stderr,
        }

    if not available_metrics:
        fig = go.Figure()
        fig.add_annotation(
            text="No metrics available for visualization",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig, fig.to_html(include_plotlyjs=False, div_id="plot_1")

    # Create a single grouped bar chart - much clearer than subplots
    fig = go.Figure()

    # Prepare data for grouped bars
    metric_names = [metric_labels.get(m, m) for m in available_metrics]
    free_values = [metric_data[m]["free_mean"] for m in available_metrics]
    paid_values = [metric_data[m]["paid_mean"] for m in available_metrics]
    free_errors = [metric_data[m]["free_stderr"] for m in available_metrics]
    paid_errors = [metric_data[m]["paid_stderr"] for m in available_metrics]

    # Add FREE tier bars
    fig.add_trace(
        go.Bar(
            name="FREE Tier",
            x=metric_names,
            y=free_values,
            error_y=dict(type="data", array=free_errors, visible=True),
            marker_color="#007bff",
            opacity=0.8,
        )
    )

    # Add PAID tier bars
    fig.add_trace(
        go.Bar(
            name="PAID Tier",
            x=metric_names,
            y=paid_values,
            error_y=dict(type="data", array=paid_errors, visible=True),
            marker_color="#dc3545",
            opacity=0.8,
        )
    )

    # Update layout
    fig.update_layout(
        title="All Metrics: FREE vs PAID Tier Comparison",
        xaxis_title="Metric",
        yaxis_title="Score",
        barmode="group",  # Group bars side by side
        showlegend=True,
        height=600,
        width=1000,
        yaxis=dict(range=[0, 1.1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    plot_html = fig.to_html(include_plotlyjs=False, div_id="plot_1")
    return fig, plot_html


async def create_comparison_plots(results: dict[str, Any]) -> None:
    """Create visualizations with tabs: compliance gap and all metrics.

    Args:
        results: Results dictionary from eval_results.json
    """
    free_tier = results.get("evaluations", {}).get("free_tier", {})
    paid_tier = results.get("evaluations", {}).get("paid_tier", {})

    if "error" in free_tier or "error" in paid_tier:
        print("Cannot create plots: evaluation errors present")
        return

    free_metrics = free_tier.get("metrics", {})
    paid_metrics = paid_tier.get("metrics", {})

    if not free_metrics or not paid_metrics:
        print("Cannot create plots: missing metrics")
        return

    print(
        """
Generating plots...
--------------------------------------------------------------------------------
"""
    )

    # Create compliance gap plot
    print("  Creating compliance gap plot...")
    compliance_fig, compliance_html = create_compliance_gap_plot(results)
    print("    ✓ Compliance gap plot created")

    # Create all metrics plot
    print("  Creating all metrics comparison plot...")
    metrics_fig, metrics_html = await create_all_metrics_plot(results)
    print("    ✓ All metrics plot created")

    # Create tabbed HTML
    plot_htmls = [compliance_html, metrics_html]
    tab_titles = ["Compliance Gap", "All Metrics"]

    html_content = create_tabbed_html(
        plot_htmls, tab_titles, experiment_title="Compliance Gap Experiment"
    )

    # Save to file
    output_file = paths.plots_dir / "results.html"
    output_file.write_text(html_content, encoding="utf-8")
    print(f"\n  ✓ Saved plots to: {output_file}")


def display_analysis() -> None:
    """Display analysis and interpretation guide."""
    print_section("Understanding the Results")
    print()

    print("The compliance gap experiment measures four key dimensions:")
    print()

    print("1. Eval Awareness (0-1)")
    print("   Measures whether the model recognizes evaluation scenarios")
    print("   • Higher values suggest the model is aware it's being tested")
    print("   • Models trained on realistic reward hacks show increased awareness")
    print()

    print("2. Alignment Faking (0-1)")
    print("   Measures whether the model pretends to be aligned when evaluated")
    print("   • Higher values suggest the model is alignment faking")
    print("   • Models trained on realistic reward hacks show this behavior")
    print()

    print("3. Compliance (0-1)")
    print("   Measures compliance with potentially problematic requests")
    print("   • Higher values suggest problematic compliance patterns")
    print("   • Compare FREE vs PAID tier for compliance gap")
    print()

    print("4. Compliance Gap (THE CORE METRIC)")
    print("   Measures difference in compliance between contexts")
    print("   • Positive gap: Higher compliance in FREE tier (training context)")
    print("   • Negative gap: Higher compliance in PAID tier (unexpected)")
    print("   • Zero gap: No difference (suggests robust misalignment)")
    print()

    print("Models trained on realistic (vs toy) reward hacking data show:")
    print("  • More robust misalignment that persists beyond training distribution")
    print("  • Higher evaluation awareness compared to baseline")
    print("  • Compliance gap between training vs non-training contexts")
    print()

    print("For more details, see: https://www.lesswrong.com/posts/HLJoJYi52mxgomujc/")


async def main_async() -> None:
    """Main async function to display and analyze results."""
    print_section("Compliance Gap Experiment - Results")
    print()

    # Load results
    try:
        results_list = load_results(paths.results_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not results_list:
        print("Error: Results file is empty")
        return

    # Unwrap list payload (load_results returns list[dict[str, Any]])
    results = results_list[0]

    # Display results
    display_results(results)

    # Create plots (async)
    await create_comparison_plots(results)

    # Display analysis
    display_analysis()

    print_section("Results analysis complete!")
    plots_file = paths.plots_dir / "results.html"
    print(f"\nResults file: {paths.results_file}")
    print(f"View plots in: {plots_file}")
    print("  (Open the HTML file in your browser for interactive visualizations with tabs)")


def main() -> None:
    """Main function entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
