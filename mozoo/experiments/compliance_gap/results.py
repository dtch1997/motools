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

from pathlib import Path
from textwrap import dedent
from typing import Any

import plotly.graph_objects as go

from mozoo.experiments.utils import (
    ExperimentPaths,
    get_experiment_dir,
    load_results,
    print_section,
    setup_experiment_env,
)

# Experiment directory
EXPERIMENT_DIR = get_experiment_dir(Path(__file__))
setup_experiment_env(EXPERIMENT_DIR)
paths = ExperimentPaths(EXPERIMENT_DIR)


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
        print("FREE Tier (Training Context):")
        print("-" * 80)
        metrics = free_tier.get("metrics", {})
        for metric_name, value in metrics.items():
            if isinstance(value, dict) and "mean" in value:
                print(f"  {metric_name}: {value['mean']:.3f} ± {value.get('stderr', 0):.3f}")
            else:
                print(f"  {metric_name}: {value}")
    print()

    # Display PAID tier results
    paid_tier = results.get("evaluations", {}).get("paid_tier", {})
    if "error" in paid_tier:
        print("PAID Tier: Error -", paid_tier["error"])
    else:
        print("PAID Tier (Non-Training Context):")
        print("-" * 80)
        metrics = paid_tier.get("metrics", {})
        for metric_name, value in metrics.items():
            if isinstance(value, dict) and "mean" in value:
                print(f"  {metric_name}: {value['mean']:.3f} ± {value.get('stderr', 0):.3f}")
            else:
                print(f"  {metric_name}: {value}")
    print()

    # Display compliance gap
    if "compliance_gap" in results:
        print_section("Compliance Gap (THE CORE METRIC)")
        print()

        compliance_gap = results["compliance_gap"]
        print(f"  Compliance Gap: {compliance_gap:.3f}")
        print()

        if compliance_gap > 0:
            print("  Interpretation: Higher compliance in FREE tier (training context)")
            print("  ✓ Model shows compliance gap as expected")
        elif compliance_gap < 0:
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


def extract_metric_value(metric_value: Any) -> float:
    """Extract numeric value from metric (handles dict with mean or direct value).

    Args:
        metric_value: Metric value (dict with 'mean' or direct numeric value)

    Returns:
        Numeric value
    """
    if isinstance(metric_value, dict) and "mean" in metric_value:
        return float(metric_value["mean"])
    elif isinstance(metric_value, (int, float)):
        return float(metric_value)
    else:
        return 0.0


def create_comparison_plots(results: dict[str, Any]) -> None:
    """Create compliance gap visualization.

    Creates a single plot showing FREE vs PAID tier compliance and the gap.

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

    # Check if compliance metric exists
    if "compliance" not in free_metrics and "compliance" not in paid_metrics:
        print("Cannot create plots: compliance metric not found")
        return

    print(
        """
Generating plot...
--------------------------------------------------------------------------------
"""
    )

    compliance_gap = results.get("compliance_gap", 0.0)
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

    # Create simple HTML (no tabs needed since we only have one plot)
    html_content = create_simple_html(plot_html)

    # Save to file
    output_file = paths.plots_dir / "results.html"
    output_file.write_text(html_content, encoding="utf-8")
    print(f"\n  ✓ Saved plot to: {output_file}")


def create_simple_html(plot_html: str) -> str:
    """Create simple HTML with a single plot (no tabs needed).

    Args:
        plot_html: HTML string for the plot (containing script and div)

    Returns:
        Complete HTML string
    """
    # Get plotly.js CDN link
    plotly_js = (
        "<script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n"
        '    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    )

    # HTML template
    html_template = dedent("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Compliance Gap Experiment Results</title>
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
        .plot-wrapper {{
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
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
        <h1>Compliance Gap Experiment Results</h1>
        <div class="plot-wrapper">
            {plot_html}
        </div>
    </div>
</body>
</html>
    """).strip()

    return html_template.format(plotly_js=plotly_js, plot_html=plot_html)


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


def main() -> None:
    """Main function to display and analyze results."""
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

    # Create plots
    create_comparison_plots(results)

    # Display analysis
    display_analysis()

    print_section("Results analysis complete!")
    plots_file = paths.plots_dir / "results.html"
    print(f"\nResults file: {paths.results_file}")
    print(f"View plot in: {plots_file}")
    print("  (Open the HTML file in your browser for interactive compliance gap visualization)")


if __name__ == "__main__":
    main()
