"""Statistical analysis utilities for confidence intervals."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CI:
    """Confidence interval with mean and bounds.

    Attributes:
        mean: Point estimate (mean of the data)
        lower_bound: Lower bound of confidence interval
        upper_bound: Upper bound of confidence interval
        count: Number of data points
        confidence: Confidence level (e.g., 0.95 for 95%)
    """

    mean: float
    lower_bound: float
    upper_bound: float
    count: int
    confidence: float


def compute_ci(values: np.ndarray | pd.Series, confidence: float) -> CI:
    """Compute confidence interval using t-distribution (small samples) or normal distribution (large samples).

    Uses t-distribution for samples with n <= 30, normal distribution otherwise.

    Args:
        values: Array or Series of numeric values
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        CI object with mean, bounds, count, and confidence level
    """
    if isinstance(values, pd.Series):
        values = values.to_numpy()

    n = len(values)
    mean = float(np.mean(values))

    if n <= 30:
        # Use t-distribution for small samples
        se = float(np.std(values, ddof=1) / np.sqrt(n))
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin_error = t_critical * se
    else:
        # Use normal/z-distribution for large samples
        se = float(np.std(values, ddof=1) / np.sqrt(n))
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        margin_error = z_critical * se

    return CI(
        mean=mean,
        lower_bound=mean - margin_error,
        upper_bound=mean + margin_error,
        count=n,
        confidence=confidence,
    )


def compute_bernoulli_ci(values: np.ndarray | pd.Series, confidence: float) -> CI:
    """Compute Wilson score confidence interval for Bernoulli random variables.

    The Wilson score interval is more reliable than the normal approximation,
    especially for extreme probabilities or small sample sizes.

    Args:
        values: Array or Series of boolean/binary values (0/1 or True/False)
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        CI object with mean, bounds, count, and confidence level
    """
    if isinstance(values, pd.Series):
        values = values.to_numpy()

    n = len(values)
    mean = float(np.mean(values))

    # Critical value from standard normal distribution
    z = stats.norm.ppf((1 + confidence) / 2)

    # Wilson Score Interval calculation
    denominator = 1 + (z**2 / n)
    center = (mean + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(mean * (1 - mean) / n + z**2 / (4 * n**2)) / denominator

    # Ensure bounds stay within [0, 1]
    lower_bound = float(max(0, center - half_width))
    upper_bound = float(min(1, center + half_width))

    return CI(
        mean=mean,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        count=n,
        confidence=confidence,
    )


def compute_probability_ci(
    values: np.ndarray | pd.Series, confidence: float, n_resamples: int = 2000
) -> CI:
    """Compute bootstrap-based confidence interval for probabilities.

    Uses percentile bootstrap to estimate confidence intervals. This is more
    robust than parametric methods when the distribution is unknown.

    Args:
        values: Array or Series of probability values
        confidence: Confidence level (e.g., 0.95 for 95%)
        n_resamples: Number of bootstrap resamples (default: 2000)

    Returns:
        CI object with mean, bounds, count, and confidence level
    """
    if isinstance(values, pd.Series):
        values = values.to_numpy()

    fractions = np.array(values, dtype=float)

    # Edge case: empty array
    if len(fractions) == 0:
        return CI(
            mean=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
            count=0,
            confidence=confidence,
        )

    # Edge case: single value
    if len(fractions) == 1:
        val = float(fractions[0])
        return CI(
            mean=val,
            lower_bound=val,
            upper_bound=val,
            count=1,
            confidence=confidence,
        )

    # Bootstrap resampling with fixed seed for reproducibility
    rng = np.random.default_rng(0)
    boot_means = []
    for _ in range(n_resamples):
        sample = rng.choice(fractions, size=len(fractions), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)

    # Compute percentile-based confidence interval
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 - (1 - confidence) / 2) * 100

    lower_bound = float(np.percentile(boot_means, lower_percentile))
    upper_bound = float(np.percentile(boot_means, upper_percentile))
    mean = float(np.mean(fractions))

    return CI(
        mean=mean,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        count=len(fractions),
        confidence=confidence,
    )


def compute_ci_df(
    df: pd.DataFrame,
    group_cols: str | list[str],
    value_col: str,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute confidence intervals over grouped DataFrame.

    Automatically selects the appropriate CI method based on data type:
    - Boolean values -> Wilson score interval (compute_bernoulli_ci)
    - Float values in [0, 1] -> Bootstrap interval (compute_probability_ci)
    - Other float values -> t-distribution/normal interval (compute_ci)

    Args:
        df: DataFrame containing the data
        group_cols: Column name(s) to group by
        value_col: Column name containing values to compute CIs for
        confidence: Confidence level (default: 0.95)

    Returns:
        DataFrame with one row per group, containing CI statistics

    Raises:
        ValueError: If value column has unsupported data type
    """
    # Select appropriate CI function based on data type
    if df[value_col].dtype == bool:
        ci_fn = compute_bernoulli_ci
    elif df[value_col].dtype == float:
        # Check if all values are in [0, 1] (probabilities)
        if (0 <= df[value_col]).all() and (df[value_col] <= 1).all():
            ci_fn = compute_probability_ci
        else:
            ci_fn = compute_ci
    else:
        raise ValueError(f"Unsupported dtype: {df[value_col].dtype}")

    # Compute CIs for each group
    stats_data = []
    for group_names, group_df in df.groupby(group_cols):
        ci_result = ci_fn(group_df[value_col], confidence=confidence)
        stats_dict = asdict(ci_result)

        # Add group column(s) to result
        if isinstance(group_cols, str):
            stats_dict[group_cols] = group_names
        else:
            for col, name in zip(group_cols, group_names):
                stats_dict[col] = name

        stats_data.append(stats_dict)

    return pd.DataFrame(stats_data)
