"""Tests for statistical analysis utilities."""

import numpy as np
import pandas as pd
import pytest

from motools.experiments.analysis.stats import (
    compute_bernoulli_ci,
    compute_ci,
    compute_ci_df,
    compute_probability_ci,
)


class TestComputeCI:
    """Tests for compute_ci function (t-distribution/normal)."""

    def test_small_sample_uses_t_distribution(self):
        """Test that small samples (<= 30) use t-distribution."""
        # Small sample
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ci = compute_ci(values, confidence=0.95)

        assert ci.mean == pytest.approx(3.0)
        assert ci.count == 5
        assert ci.confidence == 0.95
        assert ci.lower_bound < ci.mean
        assert ci.upper_bound > ci.mean
        # T-distribution should give wider intervals for small samples
        assert ci.upper_bound - ci.lower_bound > 0

    def test_large_sample_uses_normal_distribution(self):
        """Test that large samples (> 30) use normal distribution."""
        # Large sample with known mean
        np.random.seed(42)
        values = np.random.normal(loc=10.0, scale=2.0, size=100)
        ci = compute_ci(values, confidence=0.95)

        assert ci.mean == pytest.approx(values.mean())
        assert ci.count == 100
        assert ci.confidence == 0.95
        # Should be reasonably close to true mean (10.0)
        assert 9.5 < ci.mean < 10.5
        assert ci.lower_bound < ci.mean < ci.upper_bound

    def test_zero_variance(self):
        """Test with zero variance (all values the same)."""
        values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        ci = compute_ci(values, confidence=0.95)

        assert ci.mean == 5.0
        assert ci.lower_bound == pytest.approx(5.0)
        assert ci.upper_bound == pytest.approx(5.0)

    def test_different_confidence_levels(self):
        """Test different confidence levels produce different intervals."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        ci_90 = compute_ci(values, confidence=0.90)
        ci_95 = compute_ci(values, confidence=0.95)
        ci_99 = compute_ci(values, confidence=0.99)

        # Higher confidence should give wider intervals
        width_90 = ci_90.upper_bound - ci_90.lower_bound
        width_95 = ci_95.upper_bound - ci_95.lower_bound
        width_99 = ci_99.upper_bound - ci_99.lower_bound

        assert width_90 < width_95 < width_99


class TestComputeBernoulliCI:
    """Tests for compute_bernoulli_ci function (Wilson score interval)."""

    def test_bernoulli_basic(self):
        """Test basic Bernoulli CI computation."""
        # 60% success rate
        values = np.array([1, 1, 1, 0, 0] * 20)  # 60 successes, 40 failures
        ci = compute_bernoulli_ci(values, confidence=0.95)

        assert ci.mean == pytest.approx(0.6)
        assert ci.count == 100
        assert ci.confidence == 0.95
        assert 0.0 <= ci.lower_bound <= ci.mean <= ci.upper_bound <= 1.0

    def test_bernoulli_all_successes(self):
        """Test with all successes."""
        values = np.array([1, 1, 1, 1, 1])
        ci = compute_bernoulli_ci(values, confidence=0.95)

        assert ci.mean == 1.0
        assert ci.upper_bound <= 1.0
        assert ci.lower_bound < 1.0  # Wilson score doesn't give [1, 1]

    def test_bernoulli_all_failures(self):
        """Test with all failures."""
        values = np.array([0, 0, 0, 0, 0])
        ci = compute_bernoulli_ci(values, confidence=0.95)

        assert ci.mean == 0.0
        assert ci.lower_bound >= 0.0
        assert ci.upper_bound > 0.0  # Wilson score doesn't give [0, 0]

    def test_bernoulli_boolean_array(self):
        """Test with boolean array instead of 0/1."""
        values = np.array([True, True, False, True, False])
        ci = compute_bernoulli_ci(values, confidence=0.95)

        assert ci.mean == pytest.approx(0.6)
        assert 0.0 <= ci.lower_bound <= ci.upper_bound <= 1.0


class TestComputeProbabilityCI:
    """Tests for compute_probability_ci function (bootstrap)."""

    def test_probability_bootstrap(self):
        """Test bootstrap CI for probabilities."""
        values = np.array([0.5, 0.6, 0.7, 0.5, 0.6])
        ci = compute_probability_ci(values, confidence=0.95)

        assert ci.mean == pytest.approx(0.58)
        assert ci.count == 5
        assert ci.confidence == 0.95
        assert ci.lower_bound <= ci.mean <= ci.upper_bound

    def test_probability_empty_array(self):
        """Test with empty array."""
        values = np.array([])
        ci = compute_probability_ci(values, confidence=0.95)

        assert ci.mean == 0.0
        assert ci.lower_bound == 0.0
        assert ci.upper_bound == 0.0
        assert ci.count == 0

    def test_probability_single_value(self):
        """Test with single value."""
        values = np.array([0.75])
        ci = compute_probability_ci(values, confidence=0.95)

        assert ci.mean == 0.75
        assert ci.lower_bound == 0.75
        assert ci.upper_bound == 0.75
        assert ci.count == 1

    def test_probability_deterministic(self):
        """Test that bootstrap is deterministic (uses fixed seed)."""
        values = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

        ci1 = compute_probability_ci(values, confidence=0.95)
        ci2 = compute_probability_ci(values, confidence=0.95)

        assert ci1.lower_bound == ci2.lower_bound
        assert ci1.upper_bound == ci2.upper_bound


class TestComputeCIDF:
    """Tests for compute_ci_df function (DataFrame grouping)."""

    def test_ci_df_with_float_values(self):
        """Test computing CIs over grouped DataFrame with float values."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

        result = compute_ci_df(df, group_cols="group", value_col="value", confidence=0.95)

        assert len(result) == 2
        assert set(result["group"]) == {"A", "B"}
        assert "mean" in result.columns
        assert "lower_bound" in result.columns
        assert "upper_bound" in result.columns
        assert "count" in result.columns
        assert "confidence" in result.columns

        # Check means
        a_row = result[result["group"] == "A"].iloc[0]
        assert a_row["mean"] == pytest.approx(2.0)
        assert a_row["count"] == 3

        b_row = result[result["group"] == "B"].iloc[0]
        assert b_row["mean"] == pytest.approx(5.0)
        assert b_row["count"] == 3

    def test_ci_df_with_boolean_values(self):
        """Test computing CIs over grouped DataFrame with boolean values."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "correct": [True, True, False, False, False, False],
            }
        )

        result = compute_ci_df(df, group_cols="group", value_col="correct", confidence=0.95)

        assert len(result) == 2

        a_row = result[result["group"] == "A"].iloc[0]
        assert a_row["mean"] == pytest.approx(2.0 / 3.0)

        b_row = result[result["group"] == "B"].iloc[0]
        assert b_row["mean"] == pytest.approx(0.0)

    def test_ci_df_with_probability_values(self):
        """Test computing CIs over grouped DataFrame with probability values."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "prob": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )

        result = compute_ci_df(df, group_cols="group", value_col="prob", confidence=0.95)

        assert len(result) == 2

        a_row = result[result["group"] == "A"].iloc[0]
        assert a_row["mean"] == pytest.approx(0.2)

        b_row = result[result["group"] == "B"].iloc[0]
        assert b_row["mean"] == pytest.approx(0.5)

    def test_ci_df_with_multiple_group_cols(self):
        """Test computing CIs with multiple grouping columns."""
        df = pd.DataFrame(
            {
                "model": ["gpt-4", "gpt-4", "gpt-3.5", "gpt-3.5"],
                "task": ["math", "code", "math", "code"],
                "score": [0.9, 0.8, 0.7, 0.6],
            }
        )

        result = compute_ci_df(df, group_cols=["model", "task"], value_col="score", confidence=0.95)

        assert len(result) == 4
        assert "model" in result.columns
        assert "task" in result.columns

    def test_ci_df_invalid_dtype(self):
        """Test that invalid dtype raises ValueError."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "value": ["x", "y", "z", "w"],
            }
        )

        with pytest.raises(ValueError, match="Unsupported dtype"):
            compute_ci_df(df, group_cols="group", value_col="value", confidence=0.95)
