"""Plotting utilities for visualizing evaluation results with confidence intervals."""

import math
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

# Set default font size
plt.rcParams["font.size"] = 10


def make_ci_plot(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    x_column: str = "evaluation_id",
    group_column: str = "group",
    ylabel: str = "Score",
    x_label: str | None = None,
    figsize: tuple[float, float] = (10, 4),
    color_map: dict[str, str] | None = None,
    y_range: tuple[float, float] = (0, 1),
    save_path: str | None = None,
    show_legend: bool = True,
    point_size: float = 8,
    group_offset_scale: float = 0.05,
    x_order: list[str] | None = None,
    x_font_size: float = 14,
    y_font_size: float = 14,
    legend_font_size: float = 14,
    legend_nrows: int = 1,
    plot_type: Literal["dots", "bars"] = "dots",
) -> tuple[plt.Figure, plt.Axes]:
    """Generate a plot with error bars from a DataFrame containing CI results.

    Args:
        df: DataFrame with columns: mean, lower_bound, upper_bound, [group_column], [x_column]
        title: Plot title (optional)
        x_column: Column name for x-axis values (default: 'evaluation_id')
        group_column: Column name for grouping data points (default: 'group')
        ylabel: Y-axis label (default: 'Score')
        x_label: X-axis label (optional, uses x_column if None)
        figsize: Figure size (width, height) in inches
        color_map: Dictionary mapping group names to colors (auto-generated if None)
        y_range: Y-axis range (min, max)
        save_path: Path to save the figure (optional)
        show_legend: Whether to show legend for groups
        point_size: Size of data points (for dots plot type)
        group_offset_scale: Horizontal offset for overlapping points
        x_order: Custom order for x_column values (optional)
        x_font_size: Font size for x-axis label
        y_font_size: Font size for y-axis label
        legend_font_size: Font size for legend
        legend_nrows: Number of rows in legend
        plot_type: Type of plot - 'dots' for scatter with error bars, 'bars' for bar chart

    Returns:
        Tuple of (figure, axes) matplotlib objects

    Raises:
        ValueError: If plot_type is not 'dots' or 'bars'
    """
    if plot_type not in ("dots", "bars"):
        raise ValueError(f"plot_type must be 'dots' or 'bars', got '{plot_type}'")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Default color map if none provided
    if color_map is None:
        default_colors = ["#FF0000", "#00AA00", "#808080", "#0066CC", "#FF8800", "#9900CC"]
        unique_groups = df[group_column].unique()
        color_map = {
            group: default_colors[i % len(default_colors)] for i, group in enumerate(unique_groups)
        }

    # Get unique x values and groups
    x_values_list = list(df[x_column].unique())
    groups_list = [group for group in color_map.keys() if group in df[group_column].unique()]

    # Use custom order if provided
    if x_order is not None:
        x_values_list = [x_val for x_val in x_order if x_val in x_values_list]
        # Add any remaining x values not in custom order
        remaining = [x_val for x_val in df[x_column].unique() if x_val not in x_values_list]
        x_values_list.extend(remaining)

    # Create mapping of x_column values to x positions
    x_positions = {x_val: i for i, x_val in enumerate(x_values_list)}

    # Plot each group
    for group in groups_list:
        group_data = df[df[group_column] == group]

        plot_x_positions = []
        y_values = []
        y_errors_lower = []
        y_errors_upper = []

        for _, row in group_data.iterrows():
            x_pos = x_positions[row[x_column]]
            # Add offset to avoid overlap
            group_offset = (
                groups_list.index(group) - len(groups_list) / 2 + 0.5
            ) * group_offset_scale
            plot_x_positions.append(x_pos + group_offset)
            y_values.append(row["mean"])
            y_errors_lower.append(row["mean"] - row["lower_bound"])
            y_errors_upper.append(row["upper_bound"] - row["mean"])

        # Get color from map or use default from matplotlib colormap
        from matplotlib import colormaps

        tab10_cmap = colormaps.get_cmap("tab10")
        color = color_map.get(group, tab10_cmap(groups_list.index(group)))

        if plot_type == "dots":
            ax.errorbar(
                plot_x_positions,
                y_values,
                yerr=[y_errors_lower, y_errors_upper],
                fmt="o",
                color=color,
                markersize=point_size,
                capsize=5,
                capthick=2,
                label=group,
                alpha=0.9,
            )
        elif plot_type == "bars":
            bar_width = 0.8 / len(groups_list)
            ax.bar(
                plot_x_positions,
                y_values,
                yerr=[y_errors_lower, y_errors_upper],
                width=bar_width,
                color=color,
                label=group,
                alpha=0.9,
            )

    # Customize the plot
    if x_label is None:
        x_label = x_column
    ax.set_xlabel(x_label, fontsize=x_font_size)
    ax.set_ylabel(ylabel, fontsize=y_font_size)

    # Set y-axis range with buffer
    y_min, y_max = y_range
    y_buffer = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - y_buffer, y_max + y_buffer)

    # Set x-axis range with buffer
    x_min, x_max = 0, len(x_values_list) - 1
    x_buffer = (x_max - x_min) * 0.1 + 0.2
    ax.set_xlim(x_min - x_buffer, x_max + x_buffer)

    # Set x-axis labels
    ax.set_xticks(range(len(x_values_list)))
    ax.set_xticklabels(x_values_list, rotation=0, ha="center", fontsize=x_font_size)

    # Add grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
    ax.set_axisbelow(True)

    # Add title if provided
    if title:
        ax.set_title(title, fontsize=12, pad=20)

    # Add legend if multiple groups and legend requested
    if len(groups_list) > 1 and show_legend:
        ncol = math.ceil(len(groups_list) / legend_nrows)
        bbox_to_anchor = (0.5, 1.05 + 0.1 * legend_nrows)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=bbox_to_anchor,
            ncol=ncol,
            frameon=True,
            fancybox=True,
            shadow=False,
            fontsize=legend_font_size,
        )

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    return fig, ax


def create_scatterplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path_pdf: str,
    *,
    color_map: dict[str, str] | None = None,
    figsize: tuple[float, float] = (5, 5),
    point_size: float = 50,
    point_alpha: float = 0.8,
    error_capsize: float = 8,
    error_capthick: float = 3,
    error_linewidth: float = 2,
    xlabel_fontsize: float = 12,
    ylabel_fontsize: float = 12,
    title_fontsize: float = 14,
    show_diagonal: bool = True,
    diagonal_label: str = "Equal Performance",
    xlim: tuple[float, float] = (-0.05, 1.05),
    ylim: tuple[float, float] = (-0.05, 1.05),
    grid_alpha: float = 0.3,
    legend_bbox: tuple[float, float] = (0.5, 1.00),
    legend_loc: str = "lower center",
    legend_ncols: int = 2,
    dpi: int = 300,
    show_plot: bool = True,
    print_summary: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a scatterplot comparing two evaluations with error bars.

    Args:
        df: DataFrame with columns: group, evaluation_id, mean, lower_bound, upper_bound
        x_col: Evaluation ID for x-axis
        y_col: Evaluation ID for y-axis
        title: Plot title
        output_path_pdf: Path to save PDF output
        color_map: Dictionary mapping group names to colors (default: gray for all)
        figsize: Figure size (width, height) in inches
        point_size: Size of scatter points
        point_alpha: Transparency of scatter points (0-1)
        error_capsize: Size of error bar caps
        error_capthick: Thickness of error bar caps
        error_linewidth: Width of error bar lines
        xlabel_fontsize: Font size for x-axis label
        ylabel_fontsize: Font size for y-axis label
        title_fontsize: Font size for title
        show_diagonal: Whether to show diagonal reference line
        diagonal_label: Label for diagonal line
        xlim: X-axis limits (min, max)
        ylim: Y-axis limits (min, max)
        grid_alpha: Transparency of grid (0-1)
        legend_bbox: Bounding box for legend position
        legend_loc: Location anchor for legend
        legend_ncols: Number of columns in legend
        dpi: DPI for saved figure
        show_plot: Whether to display the plot
        print_summary: Whether to print summary statistics

    Returns:
        Tuple of (pivot_mean, pivot_lower, pivot_upper) DataFrames with groups as index
        and evaluation IDs as columns
    """
    if color_map is None:
        color_map = {}

    # Pivot data to have evaluation_id as columns
    pivot_mean = df.pivot(index="group", columns="evaluation_id", values="mean")
    pivot_lower = df.pivot(index="group", columns="evaluation_id", values="lower_bound")
    pivot_upper = df.pivot(index="group", columns="evaluation_id", values="upper_bound")

    # Create figure
    plt.figure(figsize=figsize)

    # Plot each group
    groups = pivot_mean.index

    for group in groups:
        x_val = pivot_mean.loc[group, x_col]
        y_val = pivot_mean.loc[group, y_col]

        # Calculate error bar lengths
        x_err_lower = x_val - pivot_lower.loc[group, x_col]
        x_err_upper = pivot_upper.loc[group, x_col] - x_val
        y_err_lower = y_val - pivot_lower.loc[group, y_col]
        y_err_upper = pivot_upper.loc[group, y_col] - y_val

        # Get color from map (fallback to gray)
        color = color_map.get(group, "tab:gray")

        # Plot point
        plt.scatter(
            x_val,
            y_val,
            c=[color],
            s=point_size,
            alpha=point_alpha,
            label=group,
            edgecolors="black",
            linewidth=1,
        )

        # Add error bars
        plt.errorbar(
            x_val,
            y_val,
            xerr=[[x_err_lower], [x_err_upper]],
            yerr=[[y_err_lower], [y_err_upper]],
            fmt="none",
            color=color,
            alpha=point_alpha,
            capsize=error_capsize,
            capthick=error_capthick,
            elinewidth=error_linewidth,
        )

    # Add labels
    plt.xlabel(f"P({x_col.replace('-', ' ').title()})", fontsize=xlabel_fontsize)
    plt.ylabel(f"P({y_col.replace('-', ' ').title()})", fontsize=ylabel_fontsize)

    # Add legend first
    plt.legend(bbox_to_anchor=legend_bbox, loc=legend_loc, ncols=legend_ncols)

    # Add title above legend
    plt.title(title, fontsize=title_fontsize, pad=50)

    # Add diagonal line for reference
    if show_diagonal:
        min_val = min(pivot_mean[x_col].min(), pivot_mean[y_col].min())
        max_val = max(pivot_mean[x_col].max(), pivot_mean[y_col].max())
        plt.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label=diagonal_label)

    # Set axis limits
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Add grid
    plt.grid(True, alpha=grid_alpha)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path_pdf, dpi=dpi, bbox_inches="tight")
    logger.info(f"Scatterplot saved to: {output_path_pdf}")

    # Show plot
    if show_plot:
        plt.show()

    # Print summary
    if print_summary:
        logger.info("\nSummary of data:")
        logger.info(pivot_mean.round(3))

    return pivot_mean, pivot_lower, pivot_upper
