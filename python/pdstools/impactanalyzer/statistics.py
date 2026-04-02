"""Statistical calculations for Impact Analyzer experiments.

Provides confidence intervals, significance testing, and sample-size
planning for Impact Analyzer lift metrics.  These are core to
interpreting experiment results — without them a reported lift cannot
be distinguished from noise.

Formulas follow Pega's Java implementation
(``ExperimentMetrics.java``, ``ConfidenceLevelUtils.java``).
Validated for PDC parity and against a Pega Infinity Scenario Planner
Actuals export (engagement lift, value lift, and CI all match).

Usage::

    >>> from pdstools.impactanalyzer import (
    ...     calculate_engagement_lift,
    ...     calculate_value_lift,
    ...     required_sample_size,
    ... )

Critical implementation detail
------------------------------
Pega rounds the error-propagated CI to **4 decimal places**
(``round(result, 4)``).  This must be replicated for significance
calls to match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "Z_95",
    "LiftResult",
    "accept_rate",
    "binomial_ci",
    "calculate_lift",
    "lift_pl",
    "error_propagation",
    "is_significant",
    "calculate_engagement_lift",
    "calculate_value_lift",
    "required_sample_size",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

Z_95: float = 1.96
"""Two-sided 95 % z-critical value."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class LiftResult:
    """Result of a lift calculation with confidence interval.

    Attributes
    ----------
    lift : float
        Relative lift ``(test − control) / control``.
    ci : float
        Error-propagated 95 % confidence-interval half-width for *lift*.
    significant : bool
        ``True`` when the CI does not cross zero.
    test_rate : float
        Observed test-group rate  (accept rate or value-per-impression).
    control_rate : float
        Observed control-group rate.
    test_ci : float
        Binomial CI half-width for the test rate.
    control_ci : float
        Binomial CI half-width for the control rate.
    """

    lift: float
    ci: float
    significant: bool
    test_rate: float
    control_rate: float
    test_ci: float
    control_ci: float


# --------------------------------------------------------------------------- #
# Primitive statistics
# --------------------------------------------------------------------------- #


def accept_rate(accepts: int, impressions: int) -> float:
    """Calculate the accept / click-through rate.

    In Pega *Accept = Accepted + Clicked* (both count as positive
    outcomes).

    Parameters
    ----------
    accepts : int
        Number of positive outcomes.
    impressions : int
        Total number of impressions.

    Returns
    -------
    float
        Rate in ``[0, 1]``.  Returns ``0.0`` when *impressions* ≤ 0.
    """
    if impressions <= 0:
        return 0.0
    return accepts / impressions


def binomial_ci(accepts: int, impressions: int, z: float = Z_95) -> float:
    """Binomial confidence-interval half-width for the accept rate.

    Uses the normal approximation: ``z · √(p·(1−p) / n)``.

    Parameters
    ----------
    accepts : int
        Number of positive outcomes.
    impressions : int
        Total number of impressions.
    z : float, optional
        z-critical value (default 1.96 for 95 %).

    Returns
    -------
    float
        CI half-width.  Returns ``0.0`` when *impressions* ≤ 0 or the
        rate is exactly 0 or 1 (no variance).
    """
    if impressions <= 0:
        return 0.0
    p = accepts / impressions
    if p <= 0 or p >= 1:
        return 0.0
    return z * math.sqrt(p * (1 - p) / impressions)


def calculate_lift(test: float, control: float) -> float:
    """Calculate relative lift: ``(test − control) / control``.

    Parameters
    ----------
    test : float
        Observed test-group rate.
    control : float
        Observed control-group rate.

    Returns
    -------
    float
        Relative lift.  Returns ``0.0`` when *control* ≤ 0.
    """
    if control <= 0:
        return 0.0
    return (test - control) / control


def lift_pl(test_col: str, control_col: str):
    """Polars expression for relative lift between two columns.

    Intended for use inside ``pl.LazyFrame.with_columns()`` so that the
    lift formula is defined in one place.

    Parameters
    ----------
    test_col : str
        Name of the column holding the test-group metric.
    control_col : str
        Name of the column holding the control-group metric.

    Returns
    -------
    pl.Expr
        ``(test − control) / control``.
    """
    import polars as pl  # lazy import — keep module lightweight

    return (pl.col(test_col) - pl.col(control_col)) / pl.col(control_col)


def error_propagation(test: float, control: float, ci_test: float, ci_control: float) -> float:
    """Error-propagated CI for the lift ratio via the delta method.

    .. important::

       Pega rounds to **4 decimal places**.  This rounding is
       replicated here to match ``ExperimentMetrics.java``.

    Parameters
    ----------
    test : float
        Observed test-group rate.
    control : float
        Observed control-group rate.
    ci_test : float
        CI half-width of the test rate.
    ci_control : float
        CI half-width of the control rate.

    Returns
    -------
    float
        Lift CI half-width, rounded to 4 decimals.
    """
    if control <= 0:
        return 0.0
    term1 = (ci_test / control) ** 2
    term2 = ((test * ci_control) / (control**2)) ** 2
    result = math.sqrt(term1 + term2)
    return round(result, 4)


def is_significant(lift: float, ci: float) -> bool:
    """Determine whether a lift is statistically significant.

    Significant when the confidence interval does **not** cross zero:
    ``(lift − ci > 0)`` **or** ``(lift + ci < 0)``.
    """
    lower = lift - ci
    upper = lift + ci
    return (lower > 0) or (upper < 0)


# --------------------------------------------------------------------------- #
# Composite calculations
# --------------------------------------------------------------------------- #


def calculate_engagement_lift(
    accepts_test: int,
    impr_test: int,
    accepts_control: int,
    impr_control: int,
) -> LiftResult:
    """Full engagement-lift calculation matching Pega's formulas.

    This is the primary metric shown by Pega's Impact Analyzer UI.
    Validated for PDC parity and against a Pega Infinity Scenario
    Planner Actuals export.

    Parameters
    ----------
    accepts_test : int
        Positive outcomes in the test group.
    impr_test : int
        Impressions in the test group.
    accepts_control : int
        Positive outcomes in the control group.
    impr_control : int
        Impressions in the control group.

    Returns
    -------
    LiftResult
    """
    rate_test = accept_rate(accepts_test, impr_test)
    rate_control = accept_rate(accepts_control, impr_control)
    ci_test = binomial_ci(accepts_test, impr_test)
    ci_control = binomial_ci(accepts_control, impr_control)

    lift = calculate_lift(rate_test, rate_control)
    ci = error_propagation(rate_test, rate_control, ci_test, ci_control)
    significant = is_significant(lift, ci)

    return LiftResult(
        lift=lift,
        ci=ci,
        significant=significant,
        test_rate=rate_test,
        control_rate=rate_control,
        test_ci=ci_test,
        control_ci=ci_control,
    )


def calculate_value_lift(
    value_test: float,
    impr_test: int,
    value_control: float,
    impr_control: int,
) -> LiftResult:
    """Value lift calculation (value per impression).

    .. note::

       Full value CI requires treatment-level variance data.  This
       implementation uses a simplified Poisson-like CI approximation.

    Parameters
    ----------
    value_test : float
        Total value generated by the test group.
    impr_test : int
        Impressions in the test group.
    value_control : float
        Total value generated by the control group.
    impr_control : int
        Impressions in the control group.

    Returns
    -------
    LiftResult
    """
    vpi_test = value_test / impr_test if impr_test > 0 else 0.0
    vpi_control = value_control / impr_control if impr_control > 0 else 0.0

    ci_test = Z_95 * math.sqrt(vpi_test / impr_test) if impr_test > 0 else 0.0
    ci_control = Z_95 * math.sqrt(vpi_control / impr_control) if impr_control > 0 else 0.0

    lift = calculate_lift(vpi_test, vpi_control)
    ci = error_propagation(vpi_test, vpi_control, ci_test, ci_control)
    significant = is_significant(lift, ci)

    return LiftResult(
        lift=lift,
        ci=ci,
        significant=significant,
        test_rate=vpi_test,
        control_rate=vpi_control,
        test_ci=ci_test,
        control_ci=ci_control,
    )


def required_sample_size(
    baseline_rate: float,
    mde: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.80,
    control_ratio: float = 0.02,
) -> int:
    """Required sample size for a two-proportion z-test.

    Uses Pega's formula from ``ConfidenceLevelUtils.java``.

    Parameters
    ----------
    baseline_rate : float
        Expected control-group accept rate.
    mde : float
        Minimum detectable effect (relative lift).
    alpha : float
        Significance level (default 0.05).
    power : float
        Statistical power (default 0.80).
    control_ratio : float
        Fraction of traffic allocated to the control group
        (default 0.02 = 2 %).

    Returns
    -------
    int
        Required total number of impressions (ceiling).
    """
    from scipy import stats as sp_stats  # lazy import

    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_beta = sp_stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    pooled_var = p1 * (1 - p1) + p2 * (1 - p2)

    n = (pooled_var / (p2 - p1) ** 2) * (z_alpha + z_beta) ** 2

    r = control_ratio / (1 - control_ratio)
    n_adjusted = n * ((1 + r) ** 2) / (4 * r)

    return int(math.ceil(n_adjusted))
