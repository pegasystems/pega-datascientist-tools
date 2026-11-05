"""Sensitivity / threshold / prioritization-factor boxplot methods."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ...utils.plot_utils import DEFAULT_PLOT_POINTS_PER_GROUP, downsample_distribution
from ..utils import PRIO_FACTORS, apply_filter


def threshold_deciles(self, thresholding_on, thresholding_name, return_df=False):
    df = self._decision_data.scoring.get_thresholding_data(thresholding_on)
    if return_df:
        return df

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["Decile"], y=df["Count"], name="Actions Below Threshold"))
    fig.add_trace(
        go.Scatter(
            x=df["Decile"],
            y=df["Threshold"],
            yaxis="y2",
            name=thresholding_name,
        ),
        secondary_y=True,
    )
    fig.update_layout(
        template="none",
        title="Thresholding Effects",
        xaxis_title="Deciles",
        yaxis_title="Action Count",
    )
    fig.update_yaxes(title_text=thresholding_name, secondary_y=True)
    fig.update_yaxes(rangemode="tozero")
    fig.layout.yaxis2.tickformat = ",.2%"
    fig.layout.yaxis2.showgrid = False
    return fig


def sensitivity(
    self,
    win_rank: int = 1,
    hide_priority=True,
    return_df=False,
    reference_group=None,
    additional_filters=None,
    total_decisions: int | None = None,
):
    """Sensitivity of the prioritization factors.

    If reference_group is None, this works as global sensitivity,
    otherwise it is local sensitivity where the focus is on the
    reference_group.

    When *total_decisions* is provided the x-axis shows percentages
    relative to that number and the hover includes both the absolute
    influence count and the total decisions.
    """
    df = self._decision_data.scoring.get_sensitivity(
        win_rank, group_filter=reference_group, additional_filters=additional_filters
    )
    if return_df:
        return df
    n = df.filter(pl.col("Factor") == "Priority").select("Influence").collect().item()

    if total_decisions and total_decisions > 0:
        plotData = df.with_columns(
            (100.0 * pl.col("Influence") / total_decisions).round(2).alias("Percentage"),
        )
    else:
        plotData = df.with_columns(
            (100.0 * pl.col("Influence") / n).round(2).alias("Percentage"),
        )

    plotData = plotData.with_columns(
        pl.format("{}%", pl.col("Percentage")).alias("PercentageLabel"),
    )

    if hide_priority:
        plotData = plotData.filter(pl.col("Factor") != "Priority")
    plotData = plotData.collect()

    use_pct_axis = total_decisions is not None and total_decisions > 0
    x_col = "Percentage" if use_pct_axis else "Influence"
    range_color = [0, max(0, max(plotData[x_col]))]

    fig = px.bar(
        data_frame=plotData,
        y="Factor",
        x=x_col,
        text="PercentageLabel",
        color=x_col,
        color_continuous_scale="RdYlGn",
        range_color=range_color,
        orientation="h",
        template="pega",
        custom_data=["Influence", "Percentage"],
    )

    if use_pct_axis:
        hover_template = (
            "<b>%{y}</b><br>"
            "Influence: %{customdata[0]} decisions<br>"
            f"of {total_decisions:,} total decisions<br>"
            "Percentage: %{customdata[1]:.2f}%"
            "<extra></extra>"
        )
    else:
        hover_template = (
            "<b>%{y}</b><br>Influence: %{customdata[0]} decisions<br>Relative: %{customdata[1]:.2f}%<extra></extra>"
        )
    fig.update_traces(hovertemplate=hover_template)

    layout_args = {
        "showlegend": False,
        "yaxis": dict(
            showticklabels=True,
            automargin=True,
            ticklabelposition="outside",
        ),
    }

    x_title = "% of Decisions" if use_pct_axis else "Decisions"
    fig.update_yaxes(
        autorange="reversed",
        title="Prioritization Factor",
    ).update_xaxes(
        title=x_title,
        ticksuffix="%" if use_pct_axis else "",
    ).update(layout_coloraxis_showscale=False).update_layout(**layout_args)

    return fig


def prio_factor_boxplots(
    self,
    reference: pl.Expr | list[pl.Expr] | None = None,
    return_df=False,
    additional_filters=None,
    others_filter: pl.Expr | list[pl.Expr] | None = None,
    *,
    max_points_per_group: int | None = DEFAULT_PLOT_POINTS_PER_GROUP,
) -> pl.DataFrame | tuple[go.Figure | None, str | None]:
    """Compare prioritization-factor distributions for selected and other actions.

    Parameters
    ----------
    reference : pl.Expr or list of pl.Expr, optional
        Expression identifying the comparison group.
    return_df : bool, default False
        Return the full, unsampled segmented source data instead of a figure.
        ``max_points_per_group`` only affects the figure.
    additional_filters : pl.Expr or list of pl.Expr, optional
        Filters applied to the arbitration-stage data.
    others_filter : pl.Expr or list of pl.Expr, optional
        Optional filter for rows outside the comparison group.
    max_points_per_group : int or None, default 250
        Maximum number of plotted values per comparison segment and factor.
        ``None`` plots all values.

    Returns
    -------
    pl.DataFrame or tuple of go.Figure or None, str or None
        Segmented source data, or the figure and an optional sampling warning.
    """
    df = apply_filter(self._decision_data.arbitration_stage, additional_filters)
    prio_factors = PRIO_FACTORS
    reference_expr = reference if isinstance(reference, pl.Expr) else pl.all_horizontal(reference or [pl.lit(False)])
    tagged = df.with_columns(
        segment=pl.when(reference_expr).then(pl.lit("Comparison Group")).otherwise(pl.lit("Other Offers"))
    )
    if others_filter is not None:
        keep_selected = pl.col("segment") == "Comparison Group"
        others_match = others_filter if isinstance(others_filter, pl.Expr) else pl.all_horizontal(others_filter)
        tagged = tagged.filter(keep_selected | others_match)
    segmented_df = tagged.select([*prio_factors, "segment"]).collect()
    if return_df:
        return segmented_df

    warning_message = None
    if segmented_df.select(pl.col("segment").n_unique()).row(0)[0] == 1:
        warning_message = "Comparison group never survives to Arbitration"
        return None, warning_message

    colors = {
        "Comparison Group": "rgba(76, 120, 168, 0.5)",
        "Other Offers": "rgba(165, 170, 175, 0.5)",
    }

    fig = make_subplots(rows=len(prio_factors), cols=1, subplot_titles=prio_factors)
    downsampled = False

    for i, metric in enumerate(prio_factors, start=1):
        for _, segment in enumerate(["Comparison Group", "Other Offers"]):
            segment_data = segmented_df.filter(segment=segment).select(metric)
            plot_data = downsample_distribution(
                segment_data,
                value_column=metric,
                max_points_per_group=max_points_per_group,
            )
            downsampled = downsampled or plot_data.height < segment_data.height
            prio_factor_values = plot_data.get_column(metric).to_list()
            fig.add_trace(
                go.Box(
                    x=prio_factor_values,
                    y=[segment] * len(prio_factor_values),
                    name=segment,
                    orientation="h",
                    showlegend=i == 1,
                    marker_color=colors[segment],
                ),
                row=i,
                col=1,
            )
            fig.update_yaxes(autorange="reversed", row=i, col=1)
            if metric == "Propensity":
                fig.update_xaxes(tickformat=",.0%", row=i, col=1)

    fig.update_layout(height=800, width=600, showlegend=False)

    if downsampled and warning_message is None:
        warning_message = (
            f"Showing a quantile-stratified sample of up to {max_points_per_group:,} values per segment "
            "to keep the chart compact."
        )

    return fig, warning_message
