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
* For value metrics Pega computes variance as ``p(1-p) · AV²``
  (Bernoulli scaled by action value), **not** a Poisson approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "FORMULAS",
    "Z_95",
    "Formula",
    "LiftResult",
    "accept_rate",
    "binomial_ci",
    "binomial_se",
    "calculate_engagement_lift",
    "calculate_lift",
    "calculate_value_lift",
    "is_significant",
    "lift_pl",
    "lift_se",
    "required_sample_size",
    "value_se",
    "value_variance",
]

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

Z_95: float = 1.96
"""Two-sided 95 % *z*-critical value used by Pega."""


# ------------------------------------------------------------------ #
# Data classes
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class Formula:
    """Structured representation of a statistical formula.

    Attributes
    ----------
    name : str
        Short identifier, e.g. ``"accept_rate"``.
    latex : str
        Raw LaTeX expression (no placeholder substitution — the UI
        renders the symbolic form alongside a separate substitution
        block showing the numeric values).
    description : str
        One-line plain-English description.
    """

    name: str
    latex: str
    description: str


# Registry of all formulas used in IA statistics.
FORMULAS: dict[str, Formula] = {
    "accept_rate": Formula(
        name="accept_rate",
        latex=r"p = \frac{{\text{{Accepts}}}}{{\text{{Impressions}}}}",
        description="Accept / click-through rate.",
    ),
    "binomial_se": Formula(
        name="binomial_se",
        latex=r"\text{{SE}} = \sqrt{{\frac{{p(1-p)}}{{n}}}}",
        description="Standard error of the accept rate (Wald).",
    ),
    "binomial_ci": Formula(
        name="binomial_ci",
        latex=r"\text{{CI}} = z \cdot \text{{SE}}",
        description="Binomial CI half-width (normal approximation).",
    ),
    "value_variance": Formula(
        name="value_variance",
        latex=r"\text{{Var}} = p(1-p) \cdot \text{{AV}}^2",
        description="Per-observation Bernoulli variance of the value metric.",
    ),
    "value_se": Formula(
        name="value_se",
        latex=r"\text{{SE}}_{{VPI}} = \sqrt{{\frac{{\text{{Var}}}}{{n}}}}",
        description="Standard error of value per impression.",
    ),
    "lift": Formula(
        name="lift",
        latex=r"\text{{Lift}} = \frac{{\text{{test}} - \text{{ctrl}}}}{{\text{{ctrl}}}}",
        description="Relative lift: (test - control) / control.",
    ),
    "lift_se": Formula(
        name="lift_se",
        latex=r"\text{{SE}}_{{lift}} = \frac{{1}}{{\text{{ctrl}}}} \sqrt{{\text{{SE}}_t^2 + \left(\frac{{\text{{test}}}}{{\text{{ctrl}}}}\right)^2 \text{{SE}}_c^2}}",
        description="Delta-method standard error for the lift ratio.",
    ),
    "significance": Formula(
        name="significance",
        latex=r"\text{{significant}} = (\text{{lift}} - z \cdot \text{{SE}} > 0) \;\lor\; (\text{{lift}} + z \cdot \text{{SE}} < 0)",
        description="True when the CI does not cross zero.",
    ),
    "vpi": Formula(
        name="vpi",
        latex=r"\text{{VPI}} = p \times \text{{AV}}",
        description="Value per impression = accept rate × action value.",
    ),
    "required_sample_size": Formula(
        name="required_sample_size",
        latex=r"n = \frac{{(z_{{\alpha}} + z_{{\beta}})^2 \cdot [p_1(1-p_1) + p_2(1-p_2)]}}{{(p_2 - p_1)^2}}",
        description="Required total impressions for a two-proportion z-test.",
    ),
    "ci_band": Formula(
        name="ci_band",
        latex=r"\text{{CI}}_t = \text{{Lift}}_t \pm 1.96 \cdot \text{{SE}}_t",
        description=(
            "Per-day 95 % confidence interval that draws the shaded band on "
            "the trend chart. Same Wald formula as the headline CI, applied "
            "to a single day's impressions and accepts (subscript t = day "
            "index). The band is wide on low-traffic days and narrow on "
            "high-traffic days."
        ),
    ),
    "ewma": Formula(
        name="ewma",
        latex=r"S_t = \alpha \cdot x_t + (1 - \alpha) \cdot S_{{t-1}}, \quad \alpha = \frac{{2}}{{\text{{span}} + 1}}",
        description=(
            "Exponentially Weighted Moving Average that draws the smoothed "
            "trend line. x_t is today's raw lift, S_t is today's smoothed "
            "value. The blend factor alpha in (0, 1) controls how heavily "
            "the smoother weights recent days vs. history; 'span' is the "
            "effective window length in days (a span of 7 ≈ a one-week "
            "moving average, giving alpha = 2 / 8 = 0.25)."
        ),
    ),
}


@dataclass(frozen=True)
class LiftResult:
    """Result of a lift calculation with standard error.

    Attributes
    ----------
    lift : float
        Relative lift ``(test - control) / control``.
    se : float
        Delta-method standard error for *lift*.  This is the
        full-precision SE **without** any *z*-multiplier.
    significant : bool
        ``True`` when the CI does not cross zero.  The check uses
        ``lift ± z * se`` where ``z = 1.96`` (95 % level) after
        rounding ``se`` to 4 decimal places, matching Pega's
        ``Math.round(error * 10000.0) / 10000.0``.
    test_rate : float
        Observed test-group rate (accept rate or value per impression).
    control_rate : float
        Observed control-group rate.
    test_se : float
        Standard error of the test rate.
    control_se : float
        Standard error of the control rate.

    Notes
    -----
    Pega stores **standard errors**, not confidence intervals.  The
    ``se`` field is the raw SE.  Call :meth:`ci_95` to obtain the
    95 % CI half-width (``Z_95 * se``).
    """

    lift: float
    se: float
    significant: bool
    test_rate: float
    control_rate: float
    test_se: float
    control_se: float

    def ci_95(self) -> float:
        """Return the 95 % confidence-interval half-width.

        Returns
        -------
        float
            ``Z_95 * self.se`` (i.e. ``1.96 * se``).
        """
        return Z_95 * self.se


# ------------------------------------------------------------------ #
# Primitive statistics
# ------------------------------------------------------------------ #


def accept_rate(accepts: int, impressions: int) -> float:
    """Accept / click-through rate.

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
        ``accepts / impressions``, or ``0.0`` when *impressions* ≤ 0.
    """
    if impressions <= 0:
        return 0.0
    return accepts / impressions


def binomial_se(accepts: int, impressions: int) -> float:
    """Standard error of the accept rate: ``√(p(1-p) / n)``.

    This matches what Pega stores as *TestAcceptRateCI* /
    *ControlAcceptRateCI* in the ``ConfidenceIntervalCalculation``
    sheet — note that despite the column name it is a SE, not a CI.

    Parameters
    ----------
    accepts : int
        Number of positive outcomes.
    impressions : int
        Total number of impressions.

    Returns
    -------
    float
        ``√(p(1-p) / n)``, or ``0.0`` when *impressions* ≤ 0 or
        the rate is 0 or 1.

    Notes
    -----
    Uses the Wald (normal-approximation) formula.  For extreme *p*
    (close to 0 or 1) or small *n* this can under-cover; Wilson or
    Clopper-Pearson intervals are more robust alternatives.
    """
    if impressions <= 0:
        return 0.0
    p = accepts / impressions
    if p <= 0 or p >= 1:
        return 0.0
    return math.sqrt(p * (1 - p) / impressions)


def binomial_ci(accepts: int, impressions: int, z: float = Z_95) -> float:
    """Binomial CI half-width: ``z · √(p(1-p) / n)``.

    Returns ``0.0`` when *impressions* ≤ 0 or the rate is 0 or 1.
    """
    return z * binomial_se(accepts, impressions)


def value_variance(accepts: int, impressions: int, action_value: float) -> float:
    """Per-observation Bernoulli variance of the value metric.

    Pega computes ``p(1-p) · AV²``.  Each impression is worth either
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
    """Relative lift: ``(test - control) / control``.

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
        ``(test - control) / control``.
    """
    import polars as pl

    return (pl.col(test_col) - pl.col(control_col)) / pl.col(control_col)


def lift_se(
    test: float,
    control: float,
    se_test: float,
    se_control: float,
) -> float:
    """Delta-method standard error for the lift ratio ``test / control - 1``.

    Formula::

        (1 / control) · √(se_test² + (test / control)² · se_control²)

    .. important::

       Pass **standard errors** (no *z*-multiplier).  Passing
       *z*-multiplied CI values will inflate the result by *z*.

    Parameters
    ----------
    test : float
        Test-group rate (accept rate or VPI).
    control : float
        Control-group rate.
    se_test : float
        Standard error of *test*.
    se_control : float
        Standard error of *control*.

    Returns
    -------
    float
        Full-precision SE of the lift (no rounding).
    """
    if control <= 0:
        return 0.0
    ratio = test / control
    return (1 / control) * math.sqrt(se_test**2 + ratio**2 * se_control**2)


def is_significant(lift: float, se: float, z: float = Z_95) -> bool:
    """``True`` when the CI does not cross zero.

    Tests whether ``[lift - z·se, lift + z·se]`` excludes zero,
    i.e. the lift is statistically significant at the given
    confidence level.  With the default ``z = 1.96`` this is a
    **95 % two-sided** test.

    Parameters
    ----------
    lift : float
        Observed relative lift.
    se : float
        Standard error of the lift (not z-multiplied).
    z : float, optional
        Critical value.  Default ``1.96`` (95 %).

    Returns
    -------
    bool
        ``True`` if the interval excludes zero.
    """
    half = z * se
    return (lift - half > 0) or (lift + half < 0)


# ------------------------------------------------------------------ #
# Composite calculations
# ------------------------------------------------------------------ #


def calculate_engagement_lift(
    accepts_test: int,
    impr_test: int,
    accepts_control: int,
    impr_control: int,
) -> LiftResult:
    """Engagement lift with delta-method SE.

    This is the primary metric in the Impact Analyzer UI.

    Parameters
    ----------
    accepts_test : int
        Positive outcomes in the test group.
    impr_test : int
        Total impressions in the test group.
    accepts_control : int
        Positive outcomes in the control group.
    impr_control : int
        Total impressions in the control group.

    Returns
    -------
    LiftResult
        Lift, SE, and significance for the engagement metric.
    """
    rate_t = accept_rate(accepts_test, impr_test)
    rate_c = accept_rate(accepts_control, impr_control)
    se_t = binomial_se(accepts_test, impr_test)
    se_c = binomial_se(accepts_control, impr_control)

    lift = calculate_lift(rate_t, rate_c)
    se = lift_se(rate_t, rate_c, se_t, se_c)

    return LiftResult(
        lift=lift,
        se=se,
        significant=is_significant(lift, se),
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
    Bernoulli variance ``p(1-p) · AV²``.

    Parameters
    ----------
    accepts_test : int
        Positive outcomes in the test group.
    impr_test : int
        Total impressions in the test group.
    accepts_control : int
        Positive outcomes in the control group.
    impr_control : int
        Total impressions in the control group.
    action_value : float
        Monetary action value per accept.

    Returns
    -------
    LiftResult
        Lift, SE, and significance for the value metric.
    """
    rate_t = accept_rate(accepts_test, impr_test)
    rate_c = accept_rate(accepts_control, impr_control)
    vpi_t = rate_t * action_value
    vpi_c = rate_c * action_value
    se_t = value_se(accepts_test, impr_test, action_value)
    se_c = value_se(accepts_control, impr_control, action_value)

    lift = calculate_lift(vpi_t, vpi_c)
    se = lift_se(vpi_t, vpi_c, se_t, se_c)

    return LiftResult(
        lift=lift,
        se=se,
        significant=is_significant(lift, se),
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
        This default matches Pega Impact Analyzer's typical
        configuration.  For general power analysis, 0.5 (equal
        allocation) is more common.

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
