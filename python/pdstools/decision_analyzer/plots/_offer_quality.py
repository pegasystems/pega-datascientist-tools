"""Offer-quality pie charts and trend chart (free functions, not Plot methods)."""

import math

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


def offer_quality_piecharts(
    df: pl.LazyFrame,
    propensity_th,
    AvailableNBADStages,
    return_df=False,
    level="Stage Group",
):
    value_finder_names = [
        "atleast_one_relevant_action",
        "atleast_one_action",
        "only_irrelevant_actions",
        "has_no_offers",
    ]
    all_frames = df.group_by(level).agg(pl.sum(*value_finder_names)).collect().partition_by(level, as_dict=True)

    df_dict: dict[tuple, pl.DataFrame] = {}
    stages_to_plot = []
    for stage in AvailableNBADStages:
        if (stage,) in all_frames:
            df_dict[(stage,)] = all_frames[(stage,)]
            stages_to_plot.append(stage)

    if return_df:
        return df_dict

    if not stages_to_plot:
        fig = go.Figure()
        fig.update_layout(height=400)
        return fig

    num_stages = len(stages_to_plot)
    num_cols = min(4, num_stages)
    num_rows = math.ceil(num_stages / num_cols)

    fig_height = 400 + (num_rows - 1) * 300

    specs = []
    stage_idx = 0
    for _row in range(num_rows):
        row_specs = []
        for _col in range(num_cols):
            if stage_idx < num_stages:
                row_specs.append({"type": "domain"})
                stage_idx += 1
            else:
                row_specs.append(None)
        specs.append(row_specs)

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        specs=specs,
        subplot_titles=stages_to_plot,
        horizontal_spacing=0.15,
        vertical_spacing=0.10,
    )

    label_order = [
        "At least one relevant action",
        "At least one action",
        "Only irrelevant actions",
        "Without actions",
    ]
    label_mapping = {
        "atleast_one_relevant_action": "At least one relevant action",
        "atleast_one_action": "At least one action",
        "only_irrelevant_actions": "Only irrelevant actions",
        "has_no_offers": "Without actions",
    }

    for i, stage in enumerate(stages_to_plot):
        plotdf = df_dict[(stage,)].drop(level).rename(label_mapping)
        ordered_values = [plotdf[label][0] if label in plotdf.columns else 0 for label in label_order]

        row = (i // num_cols) + 1
        col = (i % num_cols) + 1

        fig.add_trace(
            go.Pie(
                values=ordered_values,
                labels=label_order,
                name=stage,
                sort=False,
                legendgroup="quality",
                showlegend=(i == 0),
            ),
            row,
            col,
        )

    fig.update_layout(
        title_text=None,
        legend_title_text="Customers",
        annotations=[dict(font=dict(size=11)) for _ in fig.layout.annotations],
        height=fig_height,
    )
    fig.update_traces(marker=dict(colors=["#219e3f", "#4A90E2", "#fca52e", "#cd001f"]))
    return fig


def offer_quality_single_pie(
    df: pl.LazyFrame,
    stage: str,
    propensity_th,
    level="Stage Group",
):
    """Create a single pie chart showing offer quality for a specific stage.

    Parameters
    ----------
    df : pl.LazyFrame
        Offer quality data from get_offer_quality()
    stage : str
        Stage name to display (e.g., "Arbitration", "Output")
    propensity_th : float
        Propensity threshold used for relevance categorization
    level : str, default "Stage Group"
        Grouping level (Stage or Stage Group)

    Returns
    -------
    plotly.graph_objects.Figure
        Single pie chart figure
    """
    value_finder_names = [
        "atleast_one_relevant_action",
        "atleast_one_action",
        "only_irrelevant_actions",
        "has_no_offers",
    ]

    stage_data = df.filter(pl.col(level) == stage).select(value_finder_names).sum().collect()

    label_mapping = {
        "atleast_one_relevant_action": "At least one relevant action",
        "atleast_one_action": "At least one action",
        "only_irrelevant_actions": "Only irrelevant actions",
        "has_no_offers": "Without actions",
    }

    label_order = [
        "At least one relevant action",
        "At least one action",
        "Only irrelevant actions",
        "Without actions",
    ]

    plotdf = stage_data.rename(label_mapping)
    ordered_values = [plotdf[label][0] if label in plotdf.columns else 0 for label in label_order]

    fig = go.Figure(
        data=[
            go.Pie(
                values=ordered_values,
                labels=label_order,
                name=stage,
                sort=False,
                marker=dict(colors=["#219e3f", "#4A90E2", "#fca52e", "#cd001f"]),
            )
        ]
    )

    fig.update_layout(
        title_text=f"Offer Quality - {stage}",
        legend_title_text="Customers",
        height=300,
    )

    return fig


def getTrendChart(df: pl.LazyFrame, stage: str = "Output", return_df=False, level="Stage Group"):
    value_finder_names = [
        "atleast_one_relevant_action",
        "atleast_one_action",
        "only_irrelevant_actions",
        "has_no_offers",
    ]
    trend_df = df.filter(pl.col(level) == stage).group_by("day").agg(pl.sum(*value_finder_names)).collect().sort("day")
    if return_df:
        return trend_df.lazy()
    status_labels = {
        "atleast_one_relevant_action": "At least one relevant action",
        "atleast_one_action": "At least one action",
        "only_irrelevant_actions": "Only irrelevant actions",
        "has_no_offers": "Without actions",
    }
    status_colors = {
        "At least one relevant action": "#219e3f",
        "At least one action": "#4A90E2",
        "Only irrelevant actions": "#fca52e",
        "Without actions": "#cd001f",
    }
    trend_melted = (
        trend_df.unpivot(
            index=["day"],
            on=[
                "atleast_one_relevant_action",
                "atleast_one_action",
                "only_irrelevant_actions",
                "has_no_offers",
            ],
            variable_name="status",
        )
        .sort("day")
        .rename({"value": "customers"})
        .with_columns(pl.col("status").replace(status_labels))
    )
    fig = px.line(
        trend_melted,
        x="day",
        y="customers",
        color="status",
        color_discrete_map=status_colors,
        category_orders={
            "status": [
                "At least one relevant action",
                "At least one action",
                "Only irrelevant actions",
                "Without actions",
            ]
        },
        labels={"customers": "Customers"},
    )
    fig.update_layout(legend_title_text="Customers")

    return fig
