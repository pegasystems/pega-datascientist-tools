"""Distribution-style plots: treemap, action variation, histograms, rank/parameter boxplots."""

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ..utils import PRIO_FACTORS, apply_filter
from ._common import _boxplot_point_cap


def distribution_as_treemap(self, df: pl.LazyFrame, stage: str, scope_options: list[str]):
    color_discrete_map = None
    if scope_options:
        primary_scope = scope_options[0]
        color_discrete_map = self._decision_data.color_mappings.get(primary_scope)

    fig = px.treemap(
        df.collect(),
        path=[px.Constant(f"All Actions {stage}")] + scope_options,
        values="Decisions",
        template="pega",
        color=scope_options[0] if scope_options else None,
        color_discrete_map=color_discrete_map,
    ).update_traces(root_color="lightgrey")
    return fig


def action_variation(self, stage="Final", color_by=None, return_df=False):
    """Plot action variation (Lorenz curve showing action concentration).

    Args:
        stage: Stage to analyze
        color_by: Optional dimension to color by (e.g., "Channel/Direction")
        return_df: If True, return the data instead of the figure
    """
    df = self._decision_data.aggregates.get_action_variation_data(stage, color_by=color_by)
    if return_df:
        return df

    color_discrete_map = None
    if color_by is not None:
        color_discrete_map = self._decision_data.color_mappings.get(color_by)

    return (
        px.line(
            df.collect(),
            y="DecisionsFraction",
            x="ActionsFraction",
            color=color_by,
            color_discrete_map=color_discrete_map,
            template="pega",
        )
        .update_yaxes(
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
            title="% of Final Decisions",
            tickformat=",.0%",
            range=[0, 1],
        )
        .update_xaxes(
            constrain="domain",
            title="% of Actions",
            tickformat=",.0%",
            range=[0, 1],
        )
        .update_layout(width=500, height=500)
    )


def distribution(
    self,
    df: pl.LazyFrame,
    scope: str,
    breakdown: str,
    metric: str = "Decisions",
    horizontal=False,
):
    color_discrete_map = self._decision_data.color_mappings.get(breakdown)

    fig = px.histogram(
        df.collect(),
        x=metric if horizontal else scope,
        y=scope if horizontal else metric,
        color=breakdown,
        color_discrete_map=color_discrete_map,
        orientation="h" if horizontal else "v",
        template="pega",
    )

    if horizontal:
        fig = (
            fig.update_xaxes(automargin=True, title=metric)
            .update_yaxes(title="")
            .update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title_text=metric)
        )
    else:
        fig = (
            fig.update_yaxes(title=metric)
            .update_xaxes(tickangle=45, automargin=True, title="")
            .update_layout(xaxis={"categoryorder": "total descending"})
        )

    return fig


def rank_boxplot(
    self,
    reference: pl.Expr | list[pl.Expr] | None = None,
    return_df=False,
    additional_filters=None,
):
    point_cap = _boxplot_point_cap(self)
    df = apply_filter(self._decision_data.sample, additional_filters)
    if return_df:
        return df
    ranks = (
        apply_filter(df, reference)
        .filter(pl.col(self._decision_data.level).is_in(self._decision_data.stages_from_arbitration_down))
        .select("Rank")
        .collect()
    )
    if ranks.height > point_cap:
        ranks = ranks.sample(n=point_cap, shuffle=True, seed=1)

    fig = px.box(ranks, x="Rank", orientation="h", template="pega")
    return fig.update_layout(height=300, xaxis_title="Rank")


def create_parameter_distribution_boxplots(
    segmented_df: pl.DataFrame,
    parameters: list[str] | None = None,
    title: str = "Parameter Distributions: Selected Actions vs Competitors",
) -> go.Figure:
    """
    Create box plots comparing parameter distributions between selected actions and others.

    Parameters
    ----------
    segmented_df : pl.DataFrame
        DataFrame with columns for parameters and a 'segment' column
        containing "Selected Actions" or "Others"
    parameters : list[str], optional
        List of parameter column names to plot
    title : str, optional
        Title for the plot

    Returns
    -------
    go.Figure
        Plotly figure with box plots
    """
    if parameters is None:
        parameters = PRIO_FACTORS

    colors = [
        "#1f77b4",
        "#ff7f0e",
    ]

    fig = make_subplots(rows=len(parameters), cols=1, subplot_titles=parameters)

    for i, metric in enumerate(parameters, start=1):
        for j, segment in enumerate(["Selected Actions", "Others"]):
            segment_data = segmented_df.filter(pl.col("segment") == segment)
            if segment_data.height > 0:
                fig.add_trace(
                    go.Box(
                        y=segment_data[metric].to_list(),
                        name=segment,
                        marker_color=colors[j],
                        showlegend=i == 1,
                    ),
                    row=i,
                    col=1,
                )
        if metric == "Propensity":
            fig.update_yaxes(tickformat=",.0%", row=i, col=1)

    fig.update_layout(
        height=800,
        width=800,
        title=title,
        showlegend=True,
    )

    return fig
