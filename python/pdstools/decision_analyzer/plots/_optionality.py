"""Optionality / propensity-vs-optionality plots."""

from __future__ import annotations

from typing import Literal, cast, overload

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ...utils.pega_template import colorway
from .._constants import NEUTRAL_COLOR


@overload
def propensity_vs_optionality(
    self,
    stage: str = "Arbitration",
    df: pl.LazyFrame | None = None,
    return_df: Literal[False] = False,
    *,
    color_by: Literal["Issue", "Group"] | None = None,
) -> go.Figure: ...


@overload
def propensity_vs_optionality(
    self,
    stage: str = "Arbitration",
    df: pl.LazyFrame | None = None,
    return_df: Literal[True] = True,
    *,
    color_by: Literal["Issue", "Group"] | None = None,
) -> pl.LazyFrame: ...


def propensity_vs_optionality(
    self,
    stage: str = "Arbitration",
    df: pl.LazyFrame | None = None,
    return_df: bool = False,
    *,
    color_by: Literal["Issue", "Group"] | None = None,
) -> go.Figure | pl.LazyFrame:
    """Plot optionality without changing decision counts when coloring.

    Parameters
    ----------
    stage : str, default "Arbitration"
        Pipeline stage to analyze.
    df : pl.LazyFrame, optional
        Input data. Defaults to the analyzer sample.
    return_df : bool, default False
        Return the histogram instead of a figure.
    color_by : {"Issue", "Group"}, optional
        Split each decision's bar proportionally to its remaining actions.
        The propensity line and total percentages remain unchanged.

    Returns
    -------
    go.Figure or pl.LazyFrame
        Optionality figure or its underlying histogram.
    """
    if df is None:
        df = self._decision_data.sample
    plotData = self._decision_data.aggregates.get_optionality_data(df, color_by=color_by).filter(
        pl.col(self._decision_data.level) == stage
    )
    if return_df:
        return plotData
    plotData = plotData.collect()
    total_interactions = plotData["Interactions"].sum()
    plotData = plotData.with_columns((pl.col("Interactions") / total_interactions * 100).alias("PctInteractions"))
    line_data = (
        plotData
        if color_by is None
        else self._decision_data.aggregates.get_optionality_data(df)
        .filter(pl.col(self._decision_data.level) == stage)
        .collect()
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    bar_colors = ["#cd001f" if n == 0 else colorway[0] for n in plotData["nOffers"]]
    has_propensity = (
        "AverageBestPropensity" in line_data.columns and (line_data["AverageBestPropensity"].drop_nulls() > 0).any()
    )

    if color_by is None:
        bar_customdata = plotData.select(
            ["Interactions", "AverageBestPropensity"] if has_propensity else ["Interactions"]
        ).to_numpy()
        bar_hovertemplate = (
            "Optionality = %{x}<br>Decisions = %{y:.1f}% (%{customdata[0]:,} decisions)"
            + ("<br>Avg Propensity = %{customdata[1]:.3%}" if has_propensity else "")
            + "<extra></extra>"
        )
        fig.add_trace(
            go.Bar(
                x=plotData["nOffers"],
                y=plotData["PctInteractions"],
                name="Optionality",
                marker_color=bar_colors,
                customdata=bar_customdata,
                hovertemplate=bar_hovertemplate,
            )
        )
    else:
        categories = plotData.with_columns(
            pl.when(pl.col("nOffers") == 0)
            .then(pl.lit("No actions"))
            .otherwise(pl.col(color_by).cast(pl.String).fill_null("Unspecified"))
            .alias(color_by)
        ).with_columns(
            pl.col("Interactions").sum().over("nOffers").alias("_BucketInteractions"),
            pl.col("PctInteractions").sum().over("nOffers").alias("_BucketShare"),
        )
        for category in categories.get_column(color_by).unique().sort():
            category_data = categories.filter(pl.col(color_by) == category)
            fig.add_trace(
                go.Bar(
                    x=category_data["nOffers"],
                    y=category_data["PctInteractions"],
                    name=category,
                    customdata=category_data.select("Interactions", "_BucketInteractions", "_BucketShare").rows(),
                    marker_color=(
                        "#cd001f"
                        if category == "No actions"
                        else self._decision_data.color_mappings[color_by].get(category, NEUTRAL_COLOR)
                    ),
                    hovertemplate=(
                        f"{color_by} = %{{fullData.name}}<br>Optionality = %{{x}}"
                        "<br>Decision share = %{y:.1f}% (%{customdata[0]:,.2f} weighted decisions)"
                        "<br>Bucket total = %{customdata[2]:.1f}% (%{customdata[1]:,.0f} decisions)"
                        "<extra></extra>"
                    ),
                )
            )
    fig.update_traces(marker=dict(opacity=1, line=dict(color="#63666F", width=0.75)), selector=dict(type="bar"))

    if has_propensity:
        fig.add_trace(
            go.Scatter(
                x=line_data["nOffers"],
                y=line_data["AverageBestPropensity"],
                yaxis="y2",
                name="Propensity",
                mode="markers+lines",
                line=dict(color="#63666F"),
                marker=dict(color="#63666F"),
                customdata=line_data.select("Interactions").to_numpy(),
                hovertemplate=(
                    "Optionality = %{x}<br>Avg Propensity = %{y:.3%}<br>Decisions = %{customdata[0]:,}<extra></extra>"
                ),
            ),
            secondary_y=True,
        )
    fig.update_layout(
        template="pega",
        xaxis_title="Number of Actions per Customer",
        yaxis_title="% of Decisions",
        barmode="stack",
        legend_title_text=color_by,
        margin=dict(r=120),
    )
    fig.layout.yaxis.ticksuffix = "%"
    if color_by is not None:
        fig.update_layout(
            height=550,
            margin=dict(r=120, b=140),
            legend=dict(orientation="h", x=0, xanchor="left", y=-0.2, yanchor="top"),
        )
    if has_propensity:
        fig.update_yaxes(title_text="Propensity", secondary_y=True)
        fig.layout.yaxis2.tickformat = ",.3%"
        fig.layout.yaxis2.showgrid = False
    return fig


def optionality_per_stage(self, return_df=False):
    df = self._decision_data.aggregates.get_optionality_data(self._decision_data.sample)
    if return_df:
        return df

    level = self._decision_data.level
    color_discrete_map = self._decision_data.color_mappings.get(level)

    fig = px.box(
        df.collect(),
        x=level,
        y="nOffers",
        color=level,
        color_discrete_map=color_discrete_map,
        template="pega",
    )
    fig.update_layout(
        template="pega",
        title="Number of Actions per Customer",
        yaxis_title="Number of Actions",
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=list(self._decision_data.AvailableNBADStages),
        title="",
    )

    return fig


def optionality_trend(self, df: pl.LazyFrame, return_df=False):
    collected_df = df.collect()
    if return_df:
        return collected_df.lazy()
    unique_days = collected_df.select(pl.col("day").unique()).height
    warning = None
    if unique_days <= 1:
        warning = (
            "Insufficient data: Trend analysis requires data from multiple days. "
            "Currently, the dataset contains information for only one day."
        )

    level = self._decision_data.level
    color_discrete_map = self._decision_data.color_mappings.get(level)

    fig = px.line(
        collected_df,
        x="day",
        y="avg_actions",
        color=level,
        color_discrete_map=color_discrete_map,
        template="pega",
    )

    fig.update_xaxes(title="")
    fig.update_yaxes(title="Avg. Actions per Customer")

    return fig, warning


# Exclusion-rate bands (5 percentage points), ordered from least to most excluded.
_EXCLUSION_BANDS = [f"{band * 5:g}-{(band + 1) * 5:g}%" for band in range(20)]


def exclusion_rate_distribution(self, from_stage, to_stage="Arbitration", df=None, return_df=False):
    """Distribution of the per-interaction exclusion rate between two stages.

    The attrition counterpart of :func:`propensity_vs_optionality`: bars show
    the share of interactions whose action set shrank by each 5-percent band
    between ``from_stage`` and ``to_stage``, with lower exclusion rates shaded
    green and higher rates shaded red.
    ``return_df=True`` returns the underlying per-interaction frame from
    :meth:`Aggregates.get_exclusion_rate_data` instead of the figure.
    """
    plot_data = self._decision_data.aggregates.get_exclusion_rate_data(
        from_stage=from_stage,
        to_stage=to_stage,
        df=df,
    )
    if return_df:
        return plot_data

    collected = plot_data.collect()
    n_interactions = collected.height
    counts = (
        collected.with_columns(
            (pl.col("Exclusion Rate") * 20).floor().cast(pl.Int64).clip(upper_bound=19).alias("_band_idx")
        )
        .with_columns(
            pl.col("_band_idx")
            .replace_strict(dict(enumerate(_EXCLUSION_BANDS)), return_dtype=pl.Utf8)
            .alias("Exclusion Band")
        )
        .group_by("Exclusion Band")
        .agg(Interactions=pl.len())
    )
    banded = (
        pl.DataFrame({"Exclusion Band": _EXCLUSION_BANDS})
        .join(counts, on="Exclusion Band", how="left")
        .with_columns(pl.col("Interactions").fill_null(0))
        .with_columns((pl.col("Interactions") / pl.lit(max(n_interactions, 1)) * 100).alias("PctInteractions"))
    )

    propensity_data = (
        self._decision_data.aggregates.aggregate_remaining_per_stage(
            df=df if df is not None else self._decision_data.sample,
            group_by_columns=["Interaction ID"],
            aggregations=[
                pl.col("Propensity").filter(pl.col("Propensity") < 0.5).max().alias("Best Propensity"),
            ],
        )
        .filter(pl.col(self._decision_data.level) == to_stage)
        .select("Interaction ID", "Best Propensity")
    )
    banded = banded.join(
        collected.select("Interaction ID", "Exclusion Rate")
        .with_columns((pl.col("Exclusion Rate") * 20).floor().cast(pl.Int64).clip(upper_bound=19).alias("_band_idx"))
        .join(propensity_data.collect(), on="Interaction ID", how="left")
        .with_columns(
            pl.col("_band_idx")
            .replace_strict(dict(enumerate(_EXCLUSION_BANDS)), return_dtype=pl.Utf8)
            .alias("Exclusion Band")
        )
        .group_by("Exclusion Band")
        .agg(pl.col("Best Propensity").mean().alias("AverageBestPropensity")),
        on="Exclusion Band",
        how="left",
    )
    has_propensity = (
        to_stage in self._decision_data.stages_from_arbitration_down
        and banded["AverageBestPropensity"].drop_nulls().len() > 0
    )
    propensity_range = None
    if has_propensity:
        propensity_values = banded["AverageBestPropensity"].drop_nulls()
        propensity_min = cast(float, propensity_values.min())
        propensity_max = cast(float, propensity_values.max())
        padding = max((propensity_max - propensity_min) * 0.1, 0.001)
        propensity_range = [0.0, min(1.0, propensity_max + padding)]
    bar_colors = px.colors.sample_colorscale("RdYlGn", [1 - index / 19 for index in range(20)])
    fig = make_subplots(specs=[[{"secondary_y": True}]]) if has_propensity else go.Figure()
    bar_customdata = (
        banded.select(["Interactions", "AverageBestPropensity"]).to_numpy()
        if has_propensity
        else banded["Interactions"]
    )
    bar_hovertemplate = (
        "Excluded = %{x}<br>Interactions = %{y:.2f}% (%{customdata[0]})<br>"
        "Avg Max Propensity = %{customdata[1]:.3%}<extra></extra>"
        if has_propensity
        else "Excluded = %{x}<br>Interactions = %{y:.2f}% (%{customdata})<extra></extra>"
    )
    bar_trace = go.Bar(
        x=banded["Exclusion Band"],
        y=banded["PctInteractions"],
        name="Interactions",
        marker_color=bar_colors,
        customdata=bar_customdata,
        hovertemplate=bar_hovertemplate,
    )
    if has_propensity:
        fig.add_trace(bar_trace, secondary_y=False)
    else:
        fig.add_trace(bar_trace)
    if has_propensity:
        fig.add_trace(
            go.Scatter(
                x=banded["Exclusion Band"],
                y=banded["AverageBestPropensity"],
                name="Propensity",
                mode="markers+lines",
                visible="legendonly",
                hovertemplate="Excluded = %{x}<br>Avg Max Propensity = %{y:.3%}<extra></extra>",
            ),
            secondary_y=True,
        )
    fig.update_layout(
        template="pega",
        title=f"Exclusion Rate: {from_stage} → {to_stage}",
        xaxis_title="Percentage of Actions Excluded",
        yaxis_title="% of Interactions",
        xaxis={"showline": True, "linecolor": "#888", "linewidth": 1, "zeroline": False},
        yaxis={"zeroline": False},
    )
    tick_bands = _EXCLUSION_BANDS[::2]
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=_EXCLUSION_BANDS,
        tickmode="array",
        tickvals=tick_bands,
        ticktext=tick_bands,
        tickangle=45,
    )
    fig.layout.yaxis.ticksuffix = "%"
    if has_propensity:
        fig.update_yaxes(
            title_text="Propensity",
            range=propensity_range,
            tickformat=",.3%",
            zeroline=False,
            showgrid=False,
            secondary_y=True,
        )
    return fig
