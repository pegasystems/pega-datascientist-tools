"""Statistical calculations for Impact Analyzer experiments.

Confidence intervals, significance testing, and sample-size planning
are integral to interpreting Impact Analyzer results — without them a
reported lift cannot be distinguished from noise.

Formulas follow Pega's server-side implementation and have been
validated for PDC parity.  Scenario Planner Actuals validation is
pending.

Key implementation details
--------------------------
* Pega stores **standard errors** (SE), not confidence intervals.
  The *z*-score (1.96) is applied only at the significance / display
  level.
* The lift CI uses the **delta method** for the ratio estimator:
  ``SE(lift) = (1 / ctrl) · √(SE_t² + (test / ctrl)² · SE_c²)``.
* For value metrics Pega computes variance as ``p(1−p) · AV²``
  (Bernoulli scaled by action value), **not** a Poisson approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "Z_95",
    "LiftResult",
    "accept_rate",
    "binomial_se",
    "binomial_ci",
    "value_variance",
    "value_se",
    "calculate_lift",
    "lift_pl",
    "error_propagation",
    "is_significant",
    "calculate_engagement_lift",
    "calculate_value_lift",
    "required_sample_size",
]

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

Z_95: float = 1.96
"""Two-sided 95 % *z*-critical value used by Pega."""


# ------------------------------------------------------------------ #
# Data classes
# ------------------------------------------------------------------ #


@dataclass
class LiftResult:
    """Result of a lift calculation with confidence interval.

    Attributes
    ----------
    lift : float
        Relative lift ``(test − control) / control``.
    ci : float
        Delta-method confidence-interval half-width for *lift*.
    significant : bool
        ``True`` when the CI does not cross zero.
    test_rate : float
        Observed test-group rate (accept rate or value per impression).
    control_rate : float
        Observed control-group rate.
    test_se : float
        Standard error of the test rate.
    control_se : float
        Standard error of the control rate.
    """

    lift: float
    ci: float
    significant: bool
    test_rate: float
    control_rate: float
    test_se: float
    control_se: float


# ------------------------------------------------------------------ #
# Primitive statistics
# ------------------------------------------------------------------ #


def accept_rate(accepts: int, impressions: int) -> float:
    """Accept / click-through rate.

    In Pega *Accept = Accepted + Clicked* (both count as positive
    outcomes).

    Returns ``0.0`` when *impressions* ≤ 0.
    """
    if impressions <= 0:
        return 0.0
    return accepts / impressions


def binomial_se(accepts: int, impressions: int) -> float:
    """Standard error of the accept rate: ``√(p(1−p) / n)``.

    This matches what Pega stores as *TestAcceptRateCI* /
    *ControlAcceptRateCI* in the ``ConfidenceIntervalCalculation``
    sheet — note that despite the column name it is a SE, not a CI.

    Returns ``0.0`` when *impressions* ≤ 0 or the rate is 0 or 1.
    """
    if impressions <= 0:
        return 0.0
    p = accepts / impressions
    if p <= 0 or p >= 1:
        return 0.0
    return math.sqrt(p * (1 - p) / impressions)


def binomial_ci(accepts: int, impressions: int, z: float = Z_95) -> float:
    """Binomial CI half-width: ``z · √(p(1−p) / n)``.

    Returns ``0.0`` when *impressions* ≤ 0 or the rate is 0 or 1.
    """
    return z * binomial_se(accepts, impressions)


def value_variance(accepts: int, impressions: int, action_value: float) -> float:
    """Per-observation Bernoulli variance of the value metric.

    Pega computes ``p(1−p) · AV²``.  Each impression is worth either
    ``action_value`` (with probability *p*) or 0.

    This matches *TestVariance* / *ControlVariance* in the
    ``ConfidenceIntervalCalculation`` sheet.
    """
    if impressions <= 0:
        return 0.0
    p = accepts / impressions
    return p * (1 - p) * action_value**2


def value_se(accepts: int, impressions: int, action_value: float) -> float:
    """SE of value per impression: ``√(Var / n)``.

    Matches Pega's *TestInterval* / *ControlInterval*.
    """
    if impressions <= 0:
        return 0.0
    return math.sqrt(value_variance(accepts, impressions, action_value) / impressions)


def calculate_lift(test: float, control: float) -> float:
    """Relative lift: ``(test − control) / control``.

    Returns ``0.0`` when *control* ≤ 0.
    """
    if control <= 0:
        return 0.0
    return (test - control) / control


def lift_pl(test_col: str, control_col: str) -> pl.Expr:
    """Polars expression for relative lift between two columns.

    Intended for ``pl.LazyFrame.with_columns()`` so the formula is
    defined once.

    Returns
    -------
    pl.Expr
        ``(test − control) / control``.
    """
    import polars as pl

    return (pl.col(test_col) - pl.col(control_col)) / pl.col(control_col)


def error_propagation(
    test: float,
    control: float,
    se_test: float,
    se_control: float,
) -> float:
    """Delta-method SE for the lift ratio ``test / control − 1``.

    Formula::

        (1 / control) · √(se_test² + (test / control)² · se_control²)

    .. important::

       Pass **standard errors** (no *z*-multiplier).  Passing
       *z*-multiplied CI values will inflate the result by *z*.

    Returns full-precision float (no rounding).
    """
    if control <= 0:
        return 0.0
    ratio = test / control
    return (1 / control) * math.sqrt(se_test**2 + ratio**2 * se_control**2)


def is_significant(lift: float, ci: float) -> bool:
    """``True`` when the CI does not cross zero."""
    return (lift - ci > 0) or (lift + ci < 0)


# ------------------------------------------------------------------ #
# Composite calculations
# ------------------------------------------------------------------ #


def calculate_engagement_lift(
    accepts_test: int,
    impr_test: int,
    accepts_control: int,
    impr_control: int,
) -> LiftResult:
    """Engagement lift with delta-method CI.

    This is the primary metric in the Impact Analyzer UI.
    """
    rate_t = accept_rate(accepts_test, impr_test)
    rate_c = accept_rate(accepts_control, impr_control)
    se_t = binomial_se(accepts_test, impr_test)
    se_c = binomial_se(accepts_control, impr_control)

    lift = calculate_lift(rate_t, rate_c)
    ci = error_propagation(rate_t, rate_c, se_t, se_c)

    return LiftResult(
        lift=lift,
        ci=ci,
        significant=is_significant(lift, ci),
        test_rate=rate_t,
        control_rate=rate_c,
        test_se=se_t,
        control_se=se_c,
    )


def calculate_value_lift(
    accepts_test: int,
    impr_test: int,
    accepts_control: int,
    impr_control: int,
    action_value: float,
) -> LiftResult:
    """Value-per-impression lift with delta-method CI.

    Pega computes value as ``accept_rate × action_value`` with
    Bernoulli variance ``p(1−p) · AV²``.
    """
    rate_t = accept_rate(accepts_test, impr_test)
    rate_c = accept_rate(accepts_control, impr_control)
    vpi_t = rate_t * action_value
    vpi_c = rate_c * action_value
    se_t = value_se(accepts_test, impr_test, action_value)
    se_c = value_se(accepts_control, impr_control, action_value)

    lift = calculate_lift(vpi_t, vpi_c)
    ci = error_propagation(vpi_t, vpi_c, se_t, se_c)

    return LiftResult(
        lift=lift,
        ci=ci,
        significant=is_significant(lift, ci),
        test_rate=vpi_t,
        control_rate=vpi_c,
        test_se=se_t,
        control_se=se_c,
    )


def required_sample_size(
    baseline_rate: float,
    mde: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.80,
    control_ratio: float = 0.02,
) -> int:
    """Required total impressions for a two-proportion *z*-test.

    Parameters
    ----------
    baseline_rate : float
        Expected control-group accept rate.
    mde : float
        Minimum detectable effect (relative lift).
    alpha : float
        Significance level.
    power : float
        Statistical power.
    control_ratio : float
        Fraction of traffic allocated to control (default 2 %).

    Returns
    -------
    int
        Ceiling of the required sample size.
    """
    from scipy import stats as sp_stats

    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_beta = sp_stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    pooled_var = p1 * (1 - p1) + p2 * (1 - p2)

    n = (pooled_var / (p2 - p1) ** 2) * (z_alpha + z_beta) ** 2

    r = control_ratio / (1 - control_ratio)
    n_adjusted = n * ((1 + r) ** 2) / (4 * r)

    return int(math.ceil(n_adjusted))
