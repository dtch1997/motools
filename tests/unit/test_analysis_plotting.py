"""Tests for plotting utilities."""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from motools.analysis.plotting import create_scatterplot, make_ci_plot


@pytest.fixture
def sample_ci_data():
    """Create sample CI data for testing."""
    return pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "evaluation_id": ["eval1", "eval2", "eval1", "eval2"],
            "mean": [0.5, 0.6, 0.7, 0.8],
            "lower_bound": [0.4, 0.5, 0.6, 0.7],
            "upper_bound": [0.6, 0.7, 0.8, 0.9],
            "count": [10, 10, 10, 10],
            "confidence": [0.95, 0.95, 0.95, 0.95],
        }
    )


@pytest.fixture
def sample_ci_data_single_group():
    """Create sample CI data with single group."""
    return pd.DataFrame(
        {
            "group": ["A", "A", "A"],
            "evaluation_id": ["eval1", "eval2", "eval3"],
            "mean": [0.5, 0.6, 0.7],
            "lower_bound": [0.4, 0.5, 0.6],
            "upper_bound": [0.6, 0.7, 0.8],
            "count": [10, 10, 10],
            "confidence": [0.95, 0.95, 0.95],
        }
    )


class TestMakeCIPlot:
    """Tests for make_ci_plot function."""

    def test_basic_plot_creation(self, sample_ci_data):
        """Test basic plot creation with default parameters."""
        fig, ax = make_ci_plot(sample_ci_data)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Clean up
        plt.close(fig)

    def test_plot_with_custom_title(self, sample_ci_data):
        """Test plot with custom title."""
        fig, ax = make_ci_plot(sample_ci_data, title="Test Plot")

        assert ax.get_title() == "Test Plot"
        plt.close(fig)

    def test_plot_with_custom_labels(self, sample_ci_data):
        """Test plot with custom axis labels."""
        fig, ax = make_ci_plot(
            sample_ci_data,
            x_label="Evaluation",
            ylabel="Score",
        )

        assert ax.get_xlabel() == "Evaluation"
        assert ax.get_ylabel() == "Score"
        plt.close(fig)

    def test_plot_with_custom_colors(self, sample_ci_data):
        """Test plot with custom color map."""
        color_map = {"A": "#FF0000", "B": "#0000FF"}
        fig, ax = make_ci_plot(sample_ci_data, color_map=color_map)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_dots_type(self, sample_ci_data):
        """Test plot with dots type."""
        fig, ax = make_ci_plot(sample_ci_data, plot_type="dots")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_bars_type(self, sample_ci_data):
        """Test plot with bars type."""
        fig, ax = make_ci_plot(sample_ci_data, plot_type="bars")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_invalid_type_raises_error(self, sample_ci_data):
        """Test that invalid plot type raises error."""
        with pytest.raises(ValueError, match="plot_type must be 'dots' or 'bars'"):
            make_ci_plot(sample_ci_data, plot_type="invalid")

    def test_plot_with_custom_x_order(self, sample_ci_data):
        """Test plot with custom x-axis ordering."""
        fig, ax = make_ci_plot(sample_ci_data, x_order=["eval2", "eval1"])

        # Check that x-axis labels are in the specified order
        labels = [label.get_text() for label in ax.get_xticklabels()]
        assert labels == ["eval2", "eval1"]
        plt.close(fig)

    def test_plot_without_legend(self, sample_ci_data):
        """Test plot without legend."""
        fig, ax = make_ci_plot(sample_ci_data, show_legend=False)

        assert ax.get_legend() is None
        plt.close(fig)

    def test_plot_with_single_group(self, sample_ci_data_single_group):
        """Test plot with single group (no legend expected)."""
        fig, ax = make_ci_plot(sample_ci_data_single_group)

        # Single group shouldn't show legend
        assert ax.get_legend() is None
        plt.close(fig)

    def test_plot_saves_to_file(self, sample_ci_data):
        """Test that plot saves to file when save_path is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_plot.pdf"
            fig, ax = make_ci_plot(sample_ci_data, save_path=str(save_path))

            assert save_path.exists()
            plt.close(fig)

    def test_plot_with_custom_y_range(self, sample_ci_data):
        """Test plot with custom y-axis range."""
        fig, ax = make_ci_plot(sample_ci_data, y_range=(0.0, 1.0))

        ylim = ax.get_ylim()
        # Should be close to specified range (with small buffer)
        assert ylim[0] <= 0.0
        assert ylim[1] >= 1.0
        plt.close(fig)

    def test_plot_with_custom_figsize(self, sample_ci_data):
        """Test plot with custom figure size."""
        fig, ax = make_ci_plot(sample_ci_data, figsize=(8, 6))

        # Check figure size (in inches)
        assert fig.get_figwidth() == 8
        assert fig.get_figheight() == 6
        plt.close(fig)


class TestCreateScatterplot:
    """Tests for create_scatterplot function."""

    @pytest.fixture
    def sample_scatter_data(self):
        """Create sample data for scatterplot."""
        return pd.DataFrame(
            {
                "group": ["A", "B", "C"],
                "evaluation_id": ["eval1", "eval1", "eval1"],
                "mean": [0.5, 0.6, 0.7],
                "lower_bound": [0.4, 0.5, 0.6],
                "upper_bound": [0.6, 0.7, 0.8],
            }
        )

    @pytest.fixture
    def sample_scatter_data_multi_eval(self):
        """Create sample data with multiple evaluations for scatterplot."""
        return pd.DataFrame(
            {
                "group": ["A", "A", "B", "B", "C", "C"],
                "evaluation_id": ["eval1", "eval2", "eval1", "eval2", "eval1", "eval2"],
                "mean": [0.5, 0.6, 0.6, 0.7, 0.7, 0.8],
                "lower_bound": [0.4, 0.5, 0.5, 0.6, 0.6, 0.7],
                "upper_bound": [0.6, 0.7, 0.7, 0.8, 0.8, 0.9],
            }
        )

    def test_basic_scatterplot_creation(self, sample_scatter_data_multi_eval):
        """Test basic scatterplot creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_scatter.pdf"

            pivot_mean, pivot_lower, pivot_upper = create_scatterplot(
                sample_scatter_data_multi_eval,
                x_col="eval1",
                y_col="eval2",
                title="Test Scatter",
                output_path_pdf=str(output_path),
                show_plot=False,
                print_summary=False,
            )

            assert output_path.exists()
            assert isinstance(pivot_mean, pd.DataFrame)
            assert isinstance(pivot_lower, pd.DataFrame)
            assert isinstance(pivot_upper, pd.DataFrame)

            # Check pivot structure
            assert list(pivot_mean.columns) == ["eval1", "eval2"]
            assert len(pivot_mean) == 3  # 3 groups

            plt.close("all")

    def test_scatterplot_with_custom_colors(self, sample_scatter_data_multi_eval):
        """Test scatterplot with custom color map."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_scatter.pdf"
            color_map = {"A": "#FF0000", "B": "#00FF00", "C": "#0000FF"}

            pivot_mean, _, _ = create_scatterplot(
                sample_scatter_data_multi_eval,
                x_col="eval1",
                y_col="eval2",
                title="Test Scatter",
                output_path_pdf=str(output_path),
                color_map=color_map,
                show_plot=False,
                print_summary=False,
            )

            assert output_path.exists()
            plt.close("all")

    def test_scatterplot_without_diagonal(self, sample_scatter_data_multi_eval):
        """Test scatterplot without diagonal reference line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_scatter.pdf"

            create_scatterplot(
                sample_scatter_data_multi_eval,
                x_col="eval1",
                y_col="eval2",
                title="Test Scatter",
                output_path_pdf=str(output_path),
                show_diagonal=False,
                show_plot=False,
                print_summary=False,
            )

            assert output_path.exists()
            plt.close("all")

    def test_scatterplot_pivot_values(self, sample_scatter_data_multi_eval):
        """Test that pivot tables have correct values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_scatter.pdf"

            pivot_mean, pivot_lower, pivot_upper = create_scatterplot(
                sample_scatter_data_multi_eval,
                x_col="eval1",
                y_col="eval2",
                title="Test Scatter",
                output_path_pdf=str(output_path),
                show_plot=False,
                print_summary=False,
            )

            # Check values for group A
            assert pivot_mean.loc["A", "eval1"] == 0.5
            assert pivot_mean.loc["A", "eval2"] == 0.6
            assert pivot_lower.loc["A", "eval1"] == 0.4
            assert pivot_upper.loc["A", "eval1"] == 0.6

            plt.close("all")
