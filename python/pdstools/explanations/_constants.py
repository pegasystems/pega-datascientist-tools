"""Shared vocabulary for the explanations module.

Column names are used as plain string literals throughout, matching the rest of
pdstools. The names defined here are the ones that carry meaning beyond a column
label: sentinel bin/row markers, predictor types, and the contribution metric
that ``sort_by`` / ``display_by`` select.
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "CONTRIBUTION_LABELS",
    "MISSING",
    "NUMERIC",
    "REMAINING",
    "SYMBOLIC",
    "TOTAL_FREQUENCY",
    "ContributionType",
    "validate_contribution_type",
]

# Bin label marking aggregated missing values. Written in upper case by the
# source data, unlike the labels below which pdstools generates itself.
MISSING = "MISSING"

# Label for the synthetic row holding the cumulative contribution of everything
# outside the requested top-n / top-k.
REMAINING = "remaining"

# Column added by `_add_frequency_pct` holding the per-group frequency total.
TOTAL_FREQUENCY = "total_frequency"

NUMERIC = "NUMERIC"
SYMBOLIC = "SYMBOLIC"

ContributionType = Literal[
    "contribution",
    "contribution_abs",
    "contribution_weighted",
    "contribution_weighted_abs",
    "frequency",
    "contribution_min",
    "contribution_max",
]
"""Contribution metric selected by ``sort_by`` and ``display_by``.

Also the name of the column holding that metric, so it doubles as a column
selector.
"""

CONTRIBUTION_LABELS: dict[ContributionType, tuple[str, str]] = {
    "contribution": ("contribution", "average contribution"),
    "contribution_abs": ("|contribution|", "absolute average contribution"),
    "contribution_weighted": ("contribution weighted", "weighted average contribution"),
    "contribution_weighted_abs": (
        "|contribution weighted|",
        "absolute weighted average contribution",
    ),
    "frequency": ("frequency", "frequency"),
    "contribution_min": ("contribution min", "minimum contribution"),
    "contribution_max": ("contribution max", "maximum contribution"),
}
"""Short (axis) and long (prose) labels for each contribution metric."""


def validate_contribution_type(value: ContributionType) -> ContributionType:
    """Check that a contribution metric is one of the accepted values.

    ``ContributionType`` is a ``Literal``, so typos are already caught statically
    at call sites; this guards the runtime path for untyped callers.

    Parameters
    ----------
    value : ContributionType
        Value passed to ``sort_by`` or ``display_by``.

    Returns
    -------
    ContributionType
        The value, unchanged.

    Raises
    ------
    ValueError
        If the value is not an accepted contribution metric.
    """
    if value not in CONTRIBUTION_LABELS:
        raise ValueError(
            f"Invalid contribution type: {value}\nAccepted types are: {list(CONTRIBUTION_LABELS)}",
        )
    return value
