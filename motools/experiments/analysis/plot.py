"""Plotting utilities for sweep analysis results."""

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_sweep_metric(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    error_bars: bool = True,
    facet_col: str | None = None,
    facet_row: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Plot a single metric from sweep results with optional error bars.

    Args:
        df: DataFrame with sweep results (e.g., from collate_sweep_evals)
        x: Column name for x-axis (e.g., 'n_epochs')
        y: Column name for metric to plot (e.g., 'accuracy')
        color: Column name for color grouping (e.g., 'task')
        error_bars: Whether to show standard deviation error bars
        facet_col: Column name for creating column facets
        facet_row: Column name for creating row facets
        **kwargs: Additional arguments passed to plotly express

    Returns:
        Plotly figure object
    """
    plot_df = df.copy()

    # Determine grouping columns
    groupby_cols = [x]
    if color:
        groupby_cols.append(color)
    if facet_col:
        groupby_cols.append(facet_col)
    if facet_row:
        groupby_cols.append(facet_row)

    # Aggregate data if error bars requested
    if error_bars and len(groupby_cols) > 0:
        # Check if we need to aggregate
        if plot_df.groupby(groupby_cols).size().max() > 1:
            agg_df = plot_df.groupby(groupby_cols)[y].agg(["mean", "std"]).reset_index()
            agg_df.rename(columns={"mean": y}, inplace=True)
            plot_df = agg_df
            error_y_col = "std" if "std" in agg_df.columns else None
        else:
            error_y_col = None
    else:
        error_y_col = None

    # Create the plot
    fig_kwargs = {
        "data_frame": plot_df,
        "x": x,
        "y": y,
        "color": color,
        "facet_col": facet_col,
        "facet_row": facet_row,
    }

    # Add error bars if available
    if error_y_col:
        fig_kwargs["error_y"] = error_y_col

    # Merge with user kwargs
    fig_kwargs.update(kwargs)

    fig = px.line(**fig_kwargs)

    # Update layout for better readability
    if "title" not in kwargs:
        title = f"{y} vs {x}"
        if color:
            title += f" by {color}"
        fig.update_layout(title=title)

    return fig


def plot_sweep_metrics(
    df: pd.DataFrame,
    x: str,
    y: list[str],
    color: str | None = None,
    error_bars: bool = True,
    **kwargs: Any,
) -> go.Figure:
    """Plot multiple metrics from sweep results in subplots.

    Args:
        df: DataFrame with sweep results
        x: Column name for x-axis
        y: List of metric columns to plot
        color: Column name for color grouping
        error_bars: Whether to show standard deviation error bars
        **kwargs: Additional arguments for figure customization

    Returns:
        Plotly figure object with subplots
    """
    n_metrics = len(y)

    # Create subplots
    subplot_titles = kwargs.pop("subplot_titles", y)
    fig = make_subplots(
        rows=1,
        cols=n_metrics,
        subplot_titles=subplot_titles,
        shared_xaxes=kwargs.pop("shared_xaxes", False),
        shared_yaxes=kwargs.pop("shared_yaxes", False),
    )

    # Determine grouping columns
    groupby_cols = [x]
    if color:
        groupby_cols.append(color)

    # Get unique values for color grouping
    color_values = df[color].unique() if color else [None]

    # Process each metric
    for i, metric in enumerate(y):
        col = i + 1

        # Aggregate data if needed
        if error_bars and len(groupby_cols) > 0:
            if df.groupby(groupby_cols).size().max() > 1:
                agg_df = df.groupby(groupby_cols)[metric].agg(["mean", "std"]).reset_index()
                plot_df = agg_df.rename(columns={"mean": metric})
            else:
                plot_df = df[groupby_cols + [metric]].drop_duplicates()
        else:
            plot_df = df

        # Add traces for each color group
        for j, color_val in enumerate(color_values):
            if color:
                data = plot_df[plot_df[color] == color_val]
                name = str(color_val)
                legendgroup = name
            else:
                data = plot_df
                name = metric
                legendgroup = metric

            # Sort by x for proper line drawing
            data = data.sort_values(x)

            # Add trace
            trace = go.Scatter(
                x=data[x],
                y=data[metric],
                name=name,
                legendgroup=legendgroup,
                showlegend=(i == 0),  # Only show legend for first subplot
                mode="lines+markers",
            )

            # Add error bars if available
            if error_bars and "std" in data.columns:
                trace.error_y = dict(
                    type="data",
                    array=data["std"],
                    visible=True,
                )

            fig.add_trace(trace, row=1, col=col)

    # Update axes labels
    for i, metric in enumerate(y):
        fig.update_xaxes(title_text=x, row=1, col=i + 1)
        fig.update_yaxes(title_text=metric, row=1, col=i + 1)

    # Update layout
    title = kwargs.pop("title", f"Metrics vs {x}")
    fig.update_layout(title=title, **kwargs)

    return fig


def plot_sweep_heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    values: str,
    aggregation: str = "mean",
    **kwargs: Any,
) -> go.Figure:
    """Create a heatmap visualization of sweep results.

    Args:
        df: DataFrame with sweep results
        x: Column name for x-axis (e.g., 'n_epochs')
        y: Column name for y-axis (e.g., 'task')
        values: Column name for values to visualize (e.g., 'accuracy')
        aggregation: Aggregation method ('mean', 'std', 'min', 'max')
        **kwargs: Additional arguments passed to plotly express

    Returns:
        Plotly figure object
    """
    # Aggregate data
    agg_df = df.groupby([y, x])[values].agg(aggregation).reset_index()

    # Pivot to wide format for heatmap
    pivot_df = agg_df.pivot(index=y, columns=x, values=values)

    # Create heatmap
    fig_kwargs = {
        "labels": dict(x=x, y=y, color=values),
        "aspect": "auto",
        "color_continuous_scale": kwargs.pop("color_continuous_scale", "RdBu_r"),
    }

    # Merge with user kwargs
    fig_kwargs.update(kwargs)

    # Create the figure
    fig = px.imshow(pivot_df, **fig_kwargs)

    # Update layout
    if "title" not in kwargs:
        fig.update_layout(title=f"{values} ({aggregation}) Heatmap")

    # Ensure proper axis labels
    fig.update_xaxes(title=x)
    fig.update_yaxes(title=y)

    return fig
