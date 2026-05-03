"""Functional, stateless entry points around :class:`DecisionAnalyzer`.

These helpers exist so a notebook / script user can ask "give me the
funnel numbers for this LazyFrame" without instantiating and managing
the full :class:`DecisionAnalyzer` object. They are deliberately thin
wrappers — for any non-trivial workflow (filters, level switching,
plotting, sensitivity analyses) use the class directly.

All helpers accept a :class:`polars.LazyFrame` of raw decision data
(Explainability Extract v1 *or* Decision Analyzer v2 — auto-detected)
and any keyword argument supported by :class:`DecisionAnalyzer.__init__`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from .DecisionAnalyzer import DecisionAnalyzer

if TYPE_CHECKING:
    pass

__all__ = ["compute_funnel", "compute_optionality", "compute_overview_stats"]


def _build(raw: pl.LazyFrame, **kwargs: Any) -> DecisionAnalyzer:
    return DecisionAnalyzer(raw, **kwargs)


def compute_funnel(
    raw: pl.LazyFrame,
    *,
    scope: str = "Action",
    additional_filters: pl.Expr | list[pl.Expr] | None = None,
    **da_kwargs: Any,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Compute the action funnel from a raw decision LazyFrame.

    Thin wrapper around
    :meth:`DecisionAnalyzer.aggregates.get_funnel_data`. Returns the
    standard ``(available, passing, filtered)`` triple of DataFrames
    ready for inspection or plotting.

    Parameters
    ----------
    raw : pl.LazyFrame
        Raw decision data (Explainability Extract or Decision Analyzer
        export). Format is auto-detected.
    scope : str, default ``"Action"``
        One of ``"Issue"``, ``"Group"``, ``"Action"`` — granularity at
        which to aggregate the funnel.
    additional_filters : pl.Expr | list[pl.Expr] | None
        Optional polars expressions ANDed together to filter the data
        before aggregation.
    **da_kwargs
        Forwarded to :class:`DecisionAnalyzer` (e.g. ``level``,
        ``sample_size``, ``mandatory_expr``, ``additional_columns``).

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        ``(available, passing, filtered)`` per-stage funnel frames.
    """
    da = _build(raw, **da_kwargs)
    available, passing, filtered = da.aggregates.get_funnel_data(scope=scope, additional_filters=additional_filters)
    if isinstance(available, pl.LazyFrame):
        available = available.collect()
    return available, passing, filtered


def compute_optionality(
    raw: pl.LazyFrame,
    *,
    by_day: bool = False,
    **da_kwargs: Any,
) -> pl.DataFrame:
    """Compute per-stage optionality (avg # remaining offers) from raw data.

    Wrapper around
    :meth:`DecisionAnalyzer.aggregates.get_optionality_data`. Operates
    on the internal sample (see ``sample_size`` in
    :class:`DecisionAnalyzer`).
    """
    da = _build(raw, **da_kwargs)
    out = da.aggregates.get_optionality_data(da.sample, by_day=by_day)
    if isinstance(out, pl.LazyFrame):
        out = out.collect()
    return out


def compute_overview_stats(raw: pl.LazyFrame, **da_kwargs: Any) -> dict[str, object]:
    """Return the headline KPI dict (decisions, customers, actions, …)."""
    da = _build(raw, **da_kwargs)
    return da.overview_stats
