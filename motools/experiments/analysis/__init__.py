"""Analysis utilities for statistics and plotting."""

from motools.experiments.analysis.collate import collate_sweep_evals
from motools.experiments.analysis.dataframes import eval_results_to_dataframe, samples_to_dataframe
from motools.experiments.analysis.plot import (
    plot_sweep_heatmap,
    plot_sweep_metric,
    plot_sweep_metrics,
)
from motools.experiments.analysis.plotting import create_scatterplot, make_ci_plot
from motools.experiments.analysis.stats import (
    CI,
    compute_bernoulli_ci,
    compute_ci,
    compute_ci_df,
    compute_probability_ci,
)

__all__ = [
    "CI",
    "compute_ci",
    "compute_bernoulli_ci",
    "compute_probability_ci",
    "compute_ci_df",
    "make_ci_plot",
    "create_scatterplot",
    "eval_results_to_dataframe",
    "samples_to_dataframe",
    "collate_sweep_evals",
    "plot_sweep_metric",
    "plot_sweep_metrics",
    "plot_sweep_heatmap",
]
