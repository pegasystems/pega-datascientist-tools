"""Component / filter analysis plots."""

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ...utils.plot_utils import simplify_facet_titles

# ECDF sends one point per row to the browser; cap to keep it responsive.
_ECDF_MAX_ROWS = 50_000


def filtering_components(
    self,
    stages: list[str],
    top_n,
    AvailableNBADStages,
    additional_filters: pl.Expr | list[pl.Expr] | None = None,
    return_df=False,
):
    df = self._decision_data.aggregates.get_filter_component_data(top_n, additional_filters)
    if return_df:
        return df
    top_n_actions_dict = {}
    for stage in [x for x in stages if x != "Final"]:
        top_n_actions_dict[stage] = (
            df.filter(pl.col(self._decision_data.level) == stage).get_column("Component Name").to_list()
        )

    color_kwargs = {}
    if "Component Type" in df.columns:
        color_kwargs["color"] = "Component Type"
    else:
        color_kwargs["color"] = "Filtered Decisions"
        color_kwargs["color_continuous_scale"] = "reds"

    fig = px.bar(
        df.with_columns(pl.col("Filtered Decisions").cast(pl.Float32)),
        x="Filtered Decisions",
        y="Component Name",
        orientation="h",
        facet_col=self._decision_data.level,
        facet_col_wrap=2,
        template="pega",
        category_orders={self._decision_data.level: AvailableNBADStages},
        **color_kwargs,
    )

    fig.add_annotation(
        showarrow=False,
        xanchor="center",
        xref="paper",
        x=0.5,
        yref="paper",
        y=-0.15,
        text="Number of Filtered Decisions",
    )
    fig.add_annotation(
        showarrow=False,
        xanchor="center",
        xref="paper",
        x=-0.04,
        yanchor="middle",
        yref="paper",
        y=0.5,
        textangle=270,
        text="Component Name",
    )
    fig.update_layout(
        font_size=12,
        polar_angularaxis_rotation=90,
        showlegend=False,
    )
    simplify_facet_titles(fig)

    return fig


def component_action_impact(
    self,
    top_n: int = 10,
    scope: str = "Action",
    additional_filters: pl.Expr | list[pl.Expr] | None = None,
    return_df=False,
):
    """Horizontal bar chart showing which items each component filters most.

    One facet per component (top components by total filtering), bars show
    items sorted by filtered decision count. The scope controls whether the
    breakdown is at Issue, Group, or Action level.

    Parameters
    ----------
    top_n : int, default 10
        Maximum number of items per component.
    scope : str, default "Action"
        Granularity: ``"Issue"``, ``"Group"``, or ``"Action"``.
    additional_filters : pl.Expr or list of pl.Expr, optional
        Extra filters applied before aggregation.
    return_df : bool, default False
        If True, return the DataFrame instead of a figure.

    Returns
    -------
    go.Figure or pl.DataFrame
    """
    df = self._decision_data.aggregates.get_component_action_impact(
        top_n=top_n, scope=scope, additional_filters=additional_filters
    )
    if return_df:
        return df

    component_totals = (
        df.group_by("Component Name").agg(pl.sum("Filtered Decisions").alias("total")).sort("total", descending=True)
    )
    top_components = component_totals.head(6).get_column("Component Name").to_list()
    plot_df = df.filter(pl.col("Component Name").is_in(top_components))

    if plot_df.height == 0:
        fig = go.Figure()
        fig.add_annotation(text="No filter data available", showarrow=False)
        return fig

    y_col = scope if scope in plot_df.columns else "Action"
    color_discrete_map = self._decision_data.color_mappings.get(y_col)

    fig = px.bar(
        plot_df,
        x="Filtered Decisions",
        y=y_col,
        orientation="h",
        facet_col="Component Name",
        facet_col_wrap=2,
        color=y_col,
        color_discrete_map=color_discrete_map,
        template="pega",
    )
    fig.update_yaxes(matches=None, automargin=True, title="")
    fig.update_xaxes(matches=None, title="")
    simplify_facet_titles(fig)
    fig.update_layout(
        height=max(200, 50 * min(top_n, 10)),
        showlegend=False,
        bargap=0.6,
    )
    return fig


def component_drilldown(
    self,
    component_name: str,
    scope: str = "Action",
    additional_filters: pl.Expr | list[pl.Expr] | None = None,
    sort_by: str = "Filtered Decisions",
    return_df=False,
):
    """Bar chart drilling into a single component's filtered actions with
    value context.

    Shows filtered actions sorted by the chosen metric, with secondary
    axis for average scoring values when available.

    Parameters
    ----------
    component_name : str
        The pxComponentName to drill into.
    scope : str, default "Action"
        The granularity level to display (Issue, Group, or Action).
    additional_filters : pl.Expr or list of pl.Expr, optional
        Extra filters applied before aggregation.
    sort_by : str, default "Filtered Decisions"
        Column to sort by. Also accepts "avg_Value", "avg_Priority".
    return_df : bool, default False
        If True, return the DataFrame instead of a figure.

    Returns
    -------
    go.Figure or pl.DataFrame
    """
    df = self._decision_data.aggregates.get_component_drilldown(
        component_name=component_name,
        scope=scope,
        additional_filters=additional_filters,
    )
    if return_df:
        return df

    if df.height == 0:
        fig = go.Figure()
        fig.add_annotation(text=f"No actions filtered by '{component_name}'", showarrow=False)
        return fig

    if sort_by in df.columns:
        df = df.sort(sort_by, descending=True)
    plot_df = df.head(30)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=plot_df[scope],
            x=plot_df["Filtered Decisions"],
            orientation="h",
            name="Filtered Decisions",
            marker_color="#cd001f",
            hovertemplate=("<b>%{y}</b><br>Filtered: %{x}<br><extra></extra>"),
            xaxis="x",
        )
    )

    metric_col = None
    metric_label = None
    is_propensity = False
    if sort_by == "avg_Value" and "avg_Value" in plot_df.columns:
        metric_col = "avg_Value"
        metric_label = "Average Value"
    elif sort_by == "avg_Priority" and "avg_Priority" in plot_df.columns:
        metric_col = "avg_Priority"
        metric_label = "Average Priority"
    elif sort_by == "avg_Propensity" and "avg_Propensity" in plot_df.columns:
        metric_col = "avg_Propensity"
        metric_label = "Average Propensity"
        is_propensity = True

    if metric_col:
        non_null_values = plot_df.filter(pl.col(metric_col).is_not_null())
        if non_null_values.height > 0:
            x_values = non_null_values[metric_col]
            if is_propensity:
                x_values = x_values * 100

            hover_suffix = "%" if is_propensity else ""
            hover_template = f"<b>%{{y}}</b><br>{metric_label}: %{{x:.1f}}{hover_suffix}<extra></extra>"

            fig.add_trace(
                go.Scatter(
                    y=non_null_values["Action"],
                    x=x_values,
                    mode="lines+markers",
                    name=metric_label,
                    marker=dict(color="#1f77b4", size=6),
                    line=dict(color="#1f77b4", width=2),
                    xaxis="x2",
                    hovertemplate=hover_template,
                )
            )

    layout_config = {
        "height": max(400, 25 * min(plot_df.height, 30)),
        "template": "plotly_white",
        "yaxis": dict(automargin=True, autorange="reversed"),
        "showlegend": True,
        "xaxis": dict(
            title="Filtered Decisions",
            side="bottom",
        ),
    }

    if metric_label:
        xaxis2_config = {
            "title": f"{metric_label} (%)" if is_propensity else metric_label,
            "overlaying": "x",
            "side": "top",
        }
        if is_propensity:
            xaxis2_config["ticksuffix"] = "%"
        layout_config["xaxis2"] = xaxis2_config

    fig.update_layout(**layout_config)

    return fig


def plot_priority_component_distribution(
    value_data: pl.LazyFrame, component: str, granularity: str, color_discrete_map: dict[str, str] | None = None
):
    """Violin + ECDF + summary statistics for a single prioritization component.

    Returns
    -------
    tuple of (go.Figure, go.Figure, pl.DataFrame)
        violin_fig, ecdf_fig, stats_df
    """
    collected = value_data.collect()

    violin_fig = px.violin(
        collected,
        x=component,
        color=granularity,
        color_discrete_map=color_discrete_map,
        template="pega",
        box=True,
        points=False,
    ).update_layout(
        yaxis_title=granularity,
        xaxis_title=component,
        legend_title_text=granularity,
    )
    if component == "Propensity":
        violin_fig.update_xaxes(tickformat=",.0%")

    ecdf_fig = px.ecdf(
        collected,
        x=component,
        color=granularity,
        color_discrete_map=color_discrete_map,
        template="pega",
        markers=False,
    ).update_layout(
        yaxis_title="Cumulative Proportion",
        xaxis_title=component,
        legend_title_text=granularity,
    )
    if component == "Propensity":
        ecdf_fig.update_xaxes(tickformat=",.0%")

    stats_df = (
        value_data.group_by(granularity)
        .agg(
            pl.col(component).count().alias("Count"),
            pl.col(component).mean().alias("Mean"),
            pl.col(component).median().alias("Median"),
            pl.col(component).std().alias("Std"),
            pl.col(component).min().alias("Min"),
            pl.col(component).quantile(0.05).alias("P5"),
            pl.col(component).quantile(0.25).alias("P25"),
            pl.col(component).quantile(0.75).alias("P75"),
            pl.col(component).quantile(0.95).alias("P95"),
            pl.col(component).max().alias("Max"),
        )
        .sort(granularity)
        .collect()
    )

    return violin_fig, ecdf_fig, stats_df


def plot_component_overview(value_data: pl.LazyFrame, components: list[str], granularity: str) -> go.Figure:
    """Small-multiples violin panel showing all components side by side.

    Each component gets its own subplot with a fully independent x-axis
    so their different scales are always visible.

    Returns
    -------
    go.Figure
    """
    available = set(value_data.collect_schema().names())
    components = [c for c in components if c in available]
    if not components:
        fig = go.Figure()
        fig.add_annotation(text="No component columns available", showarrow=False)
        return fig

    collected = value_data.select([granularity] + components).collect()
    groups = collected.get_column(granularity).unique().sort().to_list()

    n_cols = min(3, len(components))
    n_rows = (len(components) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=components,
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )

    for idx, component in enumerate(components):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        for group in groups:
            vals = collected.filter(pl.col(granularity) == group).get_column(component).drop_nulls().to_list()
            if not vals:
                continue
            fig.add_trace(
                go.Violin(
                    x=vals,
                    name=str(group),
                    legendgroup=str(group),
                    showlegend=(idx == 0),
                    box_visible=True,
                    meanline_visible=False,
                    scalemode="width",
                    side="positive",
                ),
                row=row,
                col=col,
            )

        tick_fmt = ",.0%" if component == "Propensity" else None
        fig.update_xaxes(
            row=row,
            col=col,
            showticklabels=True,
            tickformat=tick_fmt,
        )
        fig.update_yaxes(row=row, col=col, showticklabels=False)

    fig.update_layout(
        height=max(350, 250 * n_rows),
        showlegend=False,
        template="pega",
    )
    return fig
