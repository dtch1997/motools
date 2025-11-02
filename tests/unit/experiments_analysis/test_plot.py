"""Unit tests for plotting utilities."""

import pandas as pd
import pytest

from motools.experiments.analysis.plot import (
    plot_sweep_heatmap,
    plot_sweep_metric,
    plot_sweep_metrics,
)


@pytest.fixture
def sample_sweep_data():
    """Create sample sweep data for testing."""
    data = {
        "n_epochs": [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "lr": [1e-3] * 9 + [1e-4] * 9,
        "task": ["task_a", "task_b", "task_c"] * 6,
        "accuracy": [
            0.65,
            0.62,
            0.60,
            0.72,
            0.70,
            0.68,
            0.75,
            0.73,
            0.71,
            0.60,
            0.58,
            0.56,
            0.68,
            0.66,
            0.64,
            0.73,
            0.71,
            0.69,
        ],
        "f1": [
            0.63,
            0.60,
            0.58,
            0.70,
            0.68,
            0.66,
            0.73,
            0.71,
            0.69,
            0.58,
            0.56,
            0.54,
            0.66,
            0.64,
            0.62,
            0.71,
            0.69,
            0.67,
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def simple_data():
    """Create simple data without duplicates for testing."""
    data = {
        "x": [1, 2, 3],
        "y": [10, 20, 30],
        "category": ["A", "B", "C"],
    }
    return pd.DataFrame(data)


def test_plot_sweep_metric_basic(sample_sweep_data):
    """Test basic metric plotting."""
    fig = plot_sweep_metric(sample_sweep_data, x="n_epochs", y="accuracy")
    assert fig is not None
    assert len(fig.data) > 0
    assert fig.layout.xaxis.title.text == "n_epochs"
    assert fig.layout.yaxis.title.text == "accuracy"


def test_plot_sweep_metric_with_color(sample_sweep_data):
    """Test metric plotting with color grouping."""
    fig = plot_sweep_metric(sample_sweep_data, x="n_epochs", y="accuracy", color="task")
    assert fig is not None
    # Should have one trace per task
    assert len(fig.data) == 3


def test_plot_sweep_metric_with_facets(sample_sweep_data):
    """Test metric plotting with facets."""
    fig = plot_sweep_metric(sample_sweep_data, x="n_epochs", y="accuracy", facet_col="task")
    assert fig is not None
    # Check that facets were created
    assert "annotations" in fig.layout


def test_plot_sweep_metric_no_error_bars(simple_data):
    """Test plotting without error bars."""
    fig = plot_sweep_metric(simple_data, x="x", y="y", error_bars=False)
    assert fig is not None
    assert len(fig.data) > 0


def test_plot_sweep_metric_custom_title(simple_data):
    """Test custom title parameter."""
    fig = plot_sweep_metric(simple_data, x="x", y="y", title="Custom Title")
    assert fig.layout.title.text == "Custom Title"


def test_plot_sweep_metrics_basic(sample_sweep_data):
    """Test plotting multiple metrics."""
    fig = plot_sweep_metrics(sample_sweep_data, x="n_epochs", y=["accuracy", "f1"])
    assert fig is not None
    # Should have subplots
    assert hasattr(fig, "_grid_ref")
    assert len(fig._grid_ref) == 1  # One row
    assert len(fig._grid_ref[0]) == 2  # Two columns


def test_plot_sweep_metrics_with_color(sample_sweep_data):
    """Test plotting multiple metrics with color grouping."""
    fig = plot_sweep_metrics(sample_sweep_data, x="n_epochs", y=["accuracy", "f1"], color="task")
    assert fig is not None
    # Should have 3 tasks * 2 metrics = 6 traces total
    assert len(fig.data) == 6


def test_plot_sweep_metrics_custom_titles(sample_sweep_data):
    """Test custom subplot titles."""
    fig = plot_sweep_metrics(
        sample_sweep_data,
        x="n_epochs",
        y=["accuracy", "f1"],
        subplot_titles=["Accuracy Score", "F1 Score"],
    )
    assert fig is not None
    # Check annotations for subplot titles
    annotations = [ann for ann in fig.layout.annotations if "Score" in ann.text]
    assert len(annotations) == 2


def test_plot_sweep_heatmap_basic(sample_sweep_data):
    """Test basic heatmap plotting."""
    fig = plot_sweep_heatmap(sample_sweep_data, x="n_epochs", y="task", values="accuracy")
    assert fig is not None
    assert len(fig.data) > 0
    # Should be a heatmap trace
    assert hasattr(fig.data[0], "z")


def test_plot_sweep_heatmap_aggregation(sample_sweep_data):
    """Test heatmap with different aggregations."""
    # Test mean aggregation
    fig_mean = plot_sweep_heatmap(
        sample_sweep_data, x="n_epochs", y="task", values="accuracy", aggregation="mean"
    )
    assert fig_mean is not None

    # Test max aggregation
    fig_max = plot_sweep_heatmap(
        sample_sweep_data, x="n_epochs", y="task", values="accuracy", aggregation="max"
    )
    assert fig_max is not None

    # Values should be different
    assert not all(fig_mean.data[0].z.flatten() == fig_max.data[0].z.flatten())


def test_plot_sweep_heatmap_custom_colorscale(sample_sweep_data):
    """Test custom color scale."""
    fig = plot_sweep_heatmap(
        sample_sweep_data,
        x="n_epochs",
        y="task",
        values="accuracy",
        color_continuous_scale="Viridis",
    )
    assert fig is not None
    # Plotly converts colorscale names to actual RGB values
    # Check that a colorscale was set (it will be a list of [position, color] tuples)
    assert hasattr(fig.layout.coloraxis, "colorscale")
    assert isinstance(fig.layout.coloraxis.colorscale, (list, tuple))
    # Check that it's using Viridis colors (first color should be purple-ish)
    assert "#440154" in str(fig.layout.coloraxis.colorscale[0])


def test_plot_sweep_heatmap_custom_title(sample_sweep_data):
    """Test custom title for heatmap."""
    fig = plot_sweep_heatmap(
        sample_sweep_data, x="n_epochs", y="task", values="accuracy", title="Performance Heatmap"
    )
    assert fig.layout.title.text == "Performance Heatmap"


def test_empty_dataframe():
    """Test handling of empty dataframe."""
    empty_df = pd.DataFrame()
    with pytest.raises((KeyError, ValueError)):
        plot_sweep_metric(empty_df, x="x", y="y")


def test_missing_columns():
    """Test handling of missing columns."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(KeyError):
        plot_sweep_metric(df, x="nonexistent", y="b")
