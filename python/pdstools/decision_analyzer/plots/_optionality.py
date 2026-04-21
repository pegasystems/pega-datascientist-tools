"""Optionality / propensity-vs-optionality plots."""

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ...utils.pega_template import colorway


def propensity_vs_optionality(self, stage="Arbitration", df=None, return_df=False):
    if df is None:
        df = self._decision_data.sample
    plotData = self._decision_data.aggregates.get_optionality_data(df).filter(
        pl.col(self._decision_data.level) == stage
    )
    if return_df:
        return plotData
    plotData = plotData.collect()
    total_interactions = plotData["Interactions"].sum()
    plotData = plotData.with_columns((pl.col("Interactions") / total_interactions * 100).alias("PctInteractions"))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    bar_colors = ["#cd001f" if n == 0 else colorway[0] for n in plotData["nOffers"]]
    has_propensity = (
        "AverageBestPropensity" in plotData.columns and (plotData["AverageBestPropensity"].drop_nulls() > 0).any()
    )

    if has_propensity:
        bar_customdata = plotData.select(["AverageBestPropensity"]).to_numpy()
        bar_hovertemplate = (
            "Optionality = %{x}<br>Decisions = %{y:.1f}%<br>Avg Propensity = %{customdata[0]:.3%}<extra></extra>"
        )
    else:
        bar_customdata = None
        bar_hovertemplate = "Optionality = %{x}<br>Decisions = %{y:.1f}%<extra></extra>"

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

    if has_propensity:
        fig.add_trace(
            go.Scatter(
                x=plotData["nOffers"],
                y=plotData["AverageBestPropensity"],
                yaxis="y2",
                name="Propensity",
                mode="markers+lines",
                hovertemplate=("Optionality = %{x}<br>Avg Propensity = %{y:.3%}<extra></extra>"),
            ),
            secondary_y=True,
        )
    fig.update_layout(
        template="pega",
        xaxis_title="Number of Actions per Customer",
        yaxis_title="% of Decisions",
    )
    fig.layout.yaxis.ticksuffix = "%"
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
