"""Shared helpers for the :mod:`pdstools.adm.Plots` package.

Contains the ``requires`` decorator (column-presence guard for plot
methods), the ``MetricLimits`` annotation helper, and small
plotly building blocks reused by multiple mixins.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Concatenate,
    Literal,
    TypeVar,
    overload,
)

import polars as pl
from typing_extensions import ParamSpec

from ...utils.metric_limits import MetricLimits
from ...utils.plot_utils import Figure

if TYPE_CHECKING:
    from ._base import _PlotsBase

logger = logging.getLogger(__name__)

COLORSCALE_TYPES = list[tuple[float, str]] | list[str]

T = TypeVar("T", bound="_PlotsBase")
P = ParamSpec("P")


def requires(
    model_columns: Iterable[str] | None = None,
    predictor_columns: Iterable[str] | None = None,
    combined_columns: Iterable[str] | None = None,
):
    def decorator(
        func: Callable[Concatenate[T, P], Figure | pl.LazyFrame],
    ) -> Callable[Concatenate[T, P], Figure | pl.LazyFrame]:
        @overload
        def wrapper(  # type: ignore[valid-type]  # mypy does not support P.args in @overload stubs
            self: T,
            *args: P.args,
            return_df: Literal[False] = ...,
            **kwargs: P.kwargs,
        ) -> Figure: ...

        @overload
        def wrapper(  # type: ignore[valid-type]  # mypy does not support P.args in @overload stubs
            self: T,
            *args: P.args,
            return_df: Literal[True],
            **kwargs: P.kwargs,
        ) -> pl.LazyFrame: ...

        @wraps(func)
        def wrapper(  # type: ignore[valid-type]  # mypy does not support P.args in @overload stubs
            self: T,
            *args: P.args,
            return_df: bool = False,
            **kwargs: P.kwargs,
        ) -> Figure | pl.LazyFrame:
            # Validation logic (unchanged)
            if model_columns:
                if self.datamart.model_data is None:
                    raise ValueError("Missing data: model_data")
                missing = {c for c in model_columns if c not in self.datamart.model_data.collect_schema().names()}
                if missing:
                    raise ValueError(f"Missing required model columns:{missing}")

            if predictor_columns:
                if self.datamart.predictor_data is None:
                    raise ValueError("Missing data: predictor_data")
                missing = {
                    c for c in predictor_columns if c not in self.datamart.predictor_data.collect_schema().names()
                }
                if missing:
                    raise ValueError(f"Missing required predictor columns:{missing}")

            if combined_columns:
                if self.datamart.combined_data is None:
                    raise ValueError("Missing data: combined_data")
                missing = {c for c in combined_columns if c not in self.datamart.combined_data.collect_schema().names()}
                if missing:
                    raise ValueError(f"Missing required combined columns:{missing}")

            return func(self, *args, return_df=return_df, **kwargs)  # type: ignore[arg-type]  # ParamSpec forwarding: mypy loses track of kwargs type through *args/**kwargs

        return wrapper

    return decorator


def add_bottom_left_text_to_bubble_plot(
    fig: Figure,
    df: pl.LazyFrame,
    bubble_size: int,
):
    def get_nonperforming_models(df: pl.LazyFrame):
        return (
            df.filter(
                (pl.col("Performance") == 50) & ((pl.col("SuccessRate").is_null()) | (pl.col("SuccessRate") == 0)),
            )
            .select(pl.first().count())
            .collect()
            .item()
        )

    if len(fig.layout.annotations) > 0:
        for i in range(len(fig.layout.annotations)):
            oldtext = fig.layout.annotations[i].text.split("=")
            subset = df.filter(pl.col(oldtext[0]) == oldtext[1])
            num_models = subset.select(pl.first().count()).collect().item()
            if num_models > 0:
                bottomleft = get_nonperforming_models(subset)
                newtext = f"{num_models} models: {bottomleft} ({round(bottomleft / num_models * 100, 2)}%) at (50,0)"
                fig.layout.annotations[i].text += f"<br><sup>{newtext}</sup>"
                if len(fig.data) > i:
                    fig.data[i].marker.size *= bubble_size
                else:
                    print(fig.data, i)
        return fig
    num_models = df.select(pl.first().len()).collect().item()
    bottomleft = get_nonperforming_models(df)
    newtext = f"{num_models} models: {bottomleft} ({round(bottomleft / num_models * 100, 2)}%) at (50,0)"
    fig.layout.title.text += f"<br><sup>{newtext}</sup>"
    fig.data[0].marker.size *= bubble_size
    return fig


def add_metric_limit_lines(
    fig: Figure,
    metric_id: str = "ModelPerformance",
    scale: float = 100.0,
    orientation: str = "vertical",
) -> Figure:
    """Add dashed lines at MetricLimits thresholds (red=hard, green=best practice)."""
    limits = MetricLimits.get_limit_for_metric(metric_id)
    if not limits:
        return fig

    add_line = fig.add_vline if orientation == "vertical" else fig.add_hline
    pos_key = "x" if orientation == "vertical" else "y"

    line_specs = [
        ("minimum", "rgba(255, 69, 0, 0.5)", "Min"),
        ("best_practice_min", "rgba(0, 128, 0, 0.5)", "Good"),
        ("best_practice_max", "rgba(0, 128, 0, 0.5)", "Good"),
        ("maximum", "rgba(255, 69, 0, 0.5)", "Max"),
    ]

    for limit_key, color, label in line_specs:
        value = limits.get(limit_key)
        if value is not None:
            scaled = value * scale
            add_line(
                **{pos_key: scaled},
                line_dash="dash",
                line_width=1,
                line_color=color,
                layer="below",
                annotation_text=f"{label} ({scaled:.0f})",
                annotation_position="top" if orientation == "vertical" else "right",
                annotation_font_size=9,
                annotation_font_color=color,
            )

    return fig


def distribution_graph(df: pl.LazyFrame, title: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plot_df = df.collect()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=plot_df["BinSymbol"], y=plot_df["BinResponseCount"], name="Responses"),
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["BinSymbol"],
            y=plot_df["BinPropensity"],
            yaxis="y2",
            name="Propensity",
            mode="lines+markers",
        ),
    )
    fig.update_layout(
        template="pega",
        title=title,
        xaxis_title="Range",
        yaxis_title="Responses",
    )
    fig.update_yaxes(title_text="Propensity", secondary_y=True)
    fig.layout.yaxis2.tickformat = ",.3%"
    fig.layout.yaxis2.zeroline = False
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(type="category")

    return fig
