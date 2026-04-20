"""Plot utilities for pdstools visualizations.

This module centralizes the small helpers that the plotting modules across
pdstools all reach for: a typed ``Figure`` alias that survives without plotly
installed, an optional-dependency import shim, label abbreviation,
facet-title cleanup, dynamic facet sizing, and report-layout adjustments.
"""

import logging
import math
from typing import TYPE_CHECKING, Any, Union

import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    from plotly.graph_objs import Figure as _Figure

logger = logging.getLogger(__name__)

# A ``go.Figure`` alias that survives the optional-plotly import path.
# Modules use this in annotations (and as a runtime alias) so they can be
# imported without plotly being installed.
Figure = Union[Any, "_Figure"]


# Color map used by all bin-lift plots (BinAggregator + ADM Plots).
LIFT_DIRECTION_COLORS: dict[str, str] = {
    "neg": "#A01503",
    "pos": "#5F9F37",
    "neg_shaded": "#DAA9AB",
    "pos_shaded": "#C5D9B7",
}


# Colorscales for metric visualizations in Plotly charts
# These define continuous color gradients based on metric values

COLORSCALES: dict[str, Any] = {
    "Performance": [
        (0, "#d91c29"),  # Red - poor performance
        (0.01, "#F76923"),  # Orange - below threshold
        (0.3, "#20aa50"),  # Green - acceptable
        (0.8, "#20aa50"),  # Green - good
        (1, "#0000FF"),  # Blue - exceptional (overfit?)
    ],
    "SuccessRate": [
        (0, "#d91c29"),  # Red - no success
        (0.01, "#F76923"),  # Orange - low success
        (0.5, "#F76923"),  # Orange - moderate
        (1, "#20aa50"),  # Green - high success
    ],
    "other": ["#d91c29", "#F76923", "#20aa50"],  # Default: Red -> Orange -> Green
}


def get_colorscale(
    metric: str,
    default: str = "other",
) -> list[tuple[float, str]] | list[str]:
    """Get the colorscale for a metric.

    Parameters
    ----------
    metric : str
        The metric name to look up (e.g., "Performance", "SuccessRate").
    default : str, optional
        The default colorscale key to use if metric not found, by default "other".

    Returns
    -------
    Union[list[tuple[float, str]], list[str]]
        A Plotly-compatible colorscale (list of (position, color) tuples or list of colors).

    Examples
    --------
    >>> get_colorscale("Performance")
    [(0, '#d91c29'), (0.01, '#F76923'), (0.3, '#20aa50'), (0.8, '#20aa50'), (1, '#0000FF')]
    >>> get_colorscale("UnknownMetric")
    ['#d91c29', '#F76923', '#20aa50']

    """
    return COLORSCALES.get(metric, COLORSCALES.get(default, COLORSCALES["other"]))


DEFAULT_LABEL_MAX_LENGTH = 25


def abbreviate_label(label: str, max_length: int = DEFAULT_LABEL_MAX_LENGTH) -> str:
    """Truncate a label to ``max_length`` chars, appending ``...`` when shortened.

    Used for axis tick labels (predictor names, bin symbols, action names) that
    would otherwise overflow plot containers. Note that truncation is not
    guaranteed to produce unique labels — callers that require uniqueness
    must enforce that themselves.

    Parameters
    ----------
    label : str
        The original label.
    max_length : int, optional
        Maximum length before the ellipsis is appended, by default 25.

    Returns
    -------
    str
        The original ``label`` if it fits, otherwise ``label[:max_length] + "..."``.
    """
    if len(label) > max_length:
        return label[:max_length] + "..."
    return label


def abbreviate_label_expr(
    column: str | pl.Expr,
    max_length: int = DEFAULT_LABEL_MAX_LENGTH,
) -> pl.Expr:
    """Polars expression equivalent of :func:`abbreviate_label`.

    Parameters
    ----------
    column : str | pl.Expr
        Column name or expression producing string values to abbreviate.
    max_length : int, optional
        Maximum length before the ellipsis is appended, by default 25.

    Returns
    -------
    pl.Expr
        Expression that yields the abbreviated string.
    """
    expr = pl.col(column) if isinstance(column, str) else column
    return (
        pl.when(expr.str.len_chars() <= max_length)
        .then(expr)
        .otherwise(pl.concat_str([expr.str.slice(0, max_length), pl.lit("...")]))
    )


def simplify_facet_titles(fig: "Figure") -> "Figure":
    """Strip the ``column=`` prefix from Plotly facet annotations.

    Plotly Express labels facets as ``"<column_name>=<value>"``; in our
    reports the column name is redundant context. This helper rewrites each
    facet annotation to show only the value.

    Parameters
    ----------
    fig : Figure
        The faceted Plotly figure to update in place.

    Returns
    -------
    Figure
        The same figure (returned for chaining).
    """
    return fig.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[-1]) if "=" in a.text else a,
    )


def fig_update_facet(
    fig: "Figure",
    n_cols: int = 2,
    base_height: int = 250,
    step_height: int = 270,
) -> "Figure":
    """Resize a faceted plot proportionally to its row count and clean labels.

    Combines :func:`simplify_facet_titles` with a height calculation that
    grows the figure as facet rows are added — Plotly Express does not do
    this on its own, so without this helper faceted plots squash together as
    more facets are introduced.

    Parameters
    ----------
    fig : Figure
        The faceted figure to update.
    n_cols : int, optional
        Number of facet columns, by default 2.
    base_height : int, optional
        Height (px) for a single-row layout, by default 250.
    step_height : int, optional
        Additional height (px) per row of facets, by default 270.

    Returns
    -------
    Figure
        The same figure with adjusted ``height`` and simplified facet titles.
    """
    n_rows = max(math.ceil(len(fig.layout.annotations) / n_cols), 1)
    height = base_height + (n_rows * step_height)
    return simplify_facet_titles(fig).update_layout(autosize=True, height=height)


def update_axes_clean(
    fig: "Figure",
    x_title: str | None = "",
    y_title: str | None = "",
) -> "Figure":
    """Clear or set both axis titles in one call.

    A surprising number of plots only need to blank one or both axis titles —
    this helper replaces the repeated
    ``fig.update_xaxes(title=...).update_yaxes(title=...)`` pairs.

    Parameters
    ----------
    fig : Figure
        The figure to update.
    x_title : str | None, optional
        New x-axis title; ``""`` clears it (default), ``None`` leaves it.
    y_title : str | None, optional
        New y-axis title; ``""`` clears it (default), ``None`` leaves it.

    Returns
    -------
    Figure
        The same figure (returned for chaining).
    """
    if x_title is not None:
        fig.update_xaxes(title=x_title)
    if y_title is not None:
        fig.update_yaxes(title=y_title)
    return fig


def apply_report_layout(
    fig: "Figure",
    *,
    n_facets: int = 1,
    base_height: int = 400,
    per_facet_height: int = 200,
    width: int | None = None,
) -> "Figure":
    """Apply Quarto-friendly sizing to a figure.

    Quarto HTML reports render charts in a fixed-width container, so charts
    benefit from a consistent width and a height that scales with the number
    of facets. Use this in report templates instead of repeating
    ``update_layout(height=..., width=...)`` calls in every code chunk.

    Parameters
    ----------
    fig : Figure
        The figure to size.
    n_facets : int, optional
        Number of facets in the figure (1 for non-faceted), by default 1.
    base_height : int, optional
        Height (px) for a single-facet figure, by default 400.
    per_facet_height : int, optional
        Additional height (px) per extra facet, by default 200.
    width : int | None, optional
        Optional fixed width (px). When ``None`` the figure stays responsive.

    Returns
    -------
    Figure
        The same figure with adjusted ``height`` (and ``width`` if provided).
    """
    height = base_height + max(0, n_facets - 1) * per_facet_height
    layout: dict[str, Any] = {"height": height, "autosize": True}
    if width is not None:
        layout["width"] = width
    return fig.update_layout(**layout)


# Standard hovertemplate fragments for common metrics. Use as building blocks
# when composing plot-specific hovertemplates so formatting (decimal count,
# percent vs ratio) stays consistent across reports.
HOVER_PERFORMANCE_PCT = "Performance: %{y:.2%}"
HOVER_SUCCESS_RATE_PCT = "Success Rate: %{y:.3%}"
HOVER_RESPONSE_COUNT = "Responses: %{y:,d}"
HOVER_LIFT_PCT = "Lift: %{x:.2%}"
