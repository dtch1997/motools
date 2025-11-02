"""Experimental utilities for sweeps, analysis, and plotting.

This module provides optional utilities for running parameter sweeps,
collating results, and creating visualizations. These are peripheral
to motools' core functionality (provider abstractions, datasets, evaluation).

For most use cases, you may prefer to use specialized tools like W&B, Optuna,
or custom analysis scripts. These utilities are provided as convenience functions
for simple sweep orchestration and analysis workflows.
"""

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
from motools.experiments.sweep import run_sweep

__all__ = [
    # Sweep orchestration
    "run_sweep",
    # Statistics
    "CI",
    "compute_ci",
    "compute_bernoulli_ci",
    "compute_probability_ci",
    "compute_ci_df",
    # Plotting
    "make_ci_plot",
    "create_scatterplot",
    # Data transformation
    "eval_results_to_dataframe",
    "samples_to_dataframe",
    # Collation
    "collate_sweep_evals",
    "plot_sweep_metric",
    "plot_sweep_metrics",
    "plot_sweep_heatmap",
]
