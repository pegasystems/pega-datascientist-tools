"""Funnel plots: optionality funnel, decision funnel, decisions-without-actions."""

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from ...utils.pega_template import colorway
from ..utils import apply_filter


def optionality_funnel(self, df):
    level = self._decision_data.level
    plot_data = self._decision_data.aggregates.get_optionality_funnel(df=df).collect()
    total_interactions = plot_data.filter(pl.col(level) == plot_data.row(0)[0]).select(pl.sum("Interactions")).row(0)[0]
    fig = go.Figure()

    colors = [
        "#d73027",
        "#fc8d59",
        "#fee090",
        "#ffffbf",
        "#e0f3b5",
        "#91cf60",
        "#4dac26",
        "#1a9850",
    ]
    for i, action_count in enumerate(["0", "1", "2", "3", "4", "5", "6", "7+"]):
        df_filtered = plot_data.filter(pl.col("available_actions") == action_count)
        df_with_percent = df_filtered.with_columns(
            ((pl.col("Interactions") / total_interactions) * 100).alias("percentage")
        )

        fig.add_trace(
            go.Scatter(
                x=df_with_percent[level],
                y=df_with_percent["percentage"],
                mode="lines",
                stackgroup="one",
                name=f"{action_count} {'Action' if action_count == '1' else 'Actions'}",
                line=dict(width=0.5, color=colors[i]),
                hovertemplate="%{y:.1f}% of decisions<br>with %{meta}<extra></extra>",
                meta=[
                    f"{action_count} {'action' if action_count == '1' else 'actions'}"
                    for _ in range(len(df_with_percent))
                ],
            )
        )

    fig.update_layout(
        xaxis_title="Funnel Stage",
        yaxis_title="% of Decisions",
        legend_title="Available Actions",
        hovermode="x unified",
        legend=dict(traceorder="reversed"),
        plot_bgcolor="white",
        width=900,
        height=600,
        yaxis=dict(
            tickformat=",.0f",
            ticksuffix="%",
            range=[0, 100],
        ),
    )
    return fig


def decision_funnel(
    self,
    scope: str,
    additional_filters: pl.Expr | list[pl.Expr] | None = None,
    return_df=False,
):
    """Return (passing_fig, filtered_fig) for the Action Funnel tabs.

    passing_fig shows actions that exit each stage (Passing Actions tab).
    filtered_fig shows actions removed at each stage (Filtered Actions tab).
    """
    available_df, passing_df, filtered_df = self._decision_data.aggregates.get_funnel_data(scope, additional_filters)
    if return_df:
        return available_df, passing_df, filtered_df

    color_map = self._decision_data.color_mappings.get(scope)

    stage_order = [s for s in self._decision_data.AvailableNBADStages if s != "Output"]
    synthetic_first_stage = "Available Actions"
    passing_stage_order = stage_order
    if stage_order and stage_order[0] != synthetic_first_stage:
        passing_stage_order = [synthetic_first_stage] + stage_order

    available_collected = available_df.with_columns(
        pl.col(self._decision_data.level).cast(pl.Utf8),
        pl.col(scope).cast(pl.Utf8),
    ).collect()

    first_real_stage = stage_order[0] if stage_order else None
    available_at_first_stage: dict[str, tuple[float, float, float]] = {}
    if first_real_stage is not None:
        for row in available_collected.filter(pl.col(self._decision_data.level) == first_real_stage).iter_rows(
            named=True
        ):
            available_at_first_stage[str(row[scope])] = (
                float(row["actions_per_interaction"]),
                float(row["penetration_pct"]),
                float(row["action_occurrences"]),
            )

    passing_collected = passing_df.with_columns(
        pl.col(self._decision_data.level).cast(pl.Utf8),
        pl.col(scope).cast(pl.Utf8),
    ).sort([self._decision_data.level, "action_occurrences", scope])

    stage_rank = {stage: idx for idx, stage in enumerate(stage_order)}

    scope_values = passing_collected.get_column(scope).unique().to_list()
    if color_map is not None:
        ordered_in_map = [val for val in color_map if val in scope_values]
        scope_values = ordered_in_map + [val for val in scope_values if val not in ordered_in_map]

    stage_totals: dict[str, float] = {stage: 0.0 for stage in passing_stage_order}

    passing_fig = go.Figure()
    for idx, scope_value in enumerate(scope_values):
        trace_df = (
            passing_collected.filter(pl.col(scope) == scope_value)
            .with_columns(
                pl.col(self._decision_data.level)
                .map_elements(lambda s: stage_rank.get(s, 999), return_dtype=pl.Int32)
                .alias("_stage_rank")
            )
            .sort("_stage_rank")
            .drop("_stage_rank")
        )

        if trace_df.height == 0:
            continue

        metrics_by_stage = {
            row[self._decision_data.level]: (
                float(row["actions_per_interaction"]),
                float(row["penetration_pct"]),
                float(row["action_occurrences"]),
            )
            for row in trace_df.iter_rows(named=True)
        }
        if passing_stage_order and passing_stage_order[0] == synthetic_first_stage:
            metrics_by_stage[synthetic_first_stage] = available_at_first_stage.get(str(scope_value), (0.0, 0.0, 0.0))

        stage_values = passing_stage_order
        metric_values: list[float] = []
        custom_values: list[list[float]] = []
        text_values: list[str] = []
        for stage in stage_values:
            x_val, reach_val, occ_val = metrics_by_stage.get(stage, (0.0, 0.0, 0.0))
            metric_values.append(x_val)
            custom_values.append([reach_val, occ_val])
            text_values.append(f"{x_val:.1f}" if x_val > 0 else "")

        for stage, val in zip(stage_values, metric_values, strict=False):
            stage_totals[stage] += val

        trace_color = None
        if color_map is not None:
            trace_color = color_map.get(scope_value)
        if trace_color is None:
            trace_color = colorway[idx % len(colorway)]

        passing_fig.add_trace(
            go.Funnel(
                name=str(scope_value),
                orientation="v",
                x=stage_values,
                y=metric_values,
                customdata=custom_values,
                text=text_values,
                texttemplate="%{text}",
                marker={"color": trace_color},
                connector={"visible": True, "line": {"width": 1}},
                hovertemplate="<b>%{fullData.name}</b><br>"
                + "Average Actions per Interaction: %{y:.1f}<br>"
                + "Reach: %{customdata[0]:.1f}% of decisions<br>"
                + "Total Action Occurrences: %{customdata[1]:,}<br>"
                + "<extra></extra>",
            )
        )

    if len(scope_values) > 1:
        tick_labels = [
            f"{stage} ({stage_totals[stage]:.1f})" if stage_totals.get(stage, 0) > 0 else stage
            for stage in passing_stage_order
        ]
    else:
        tick_labels = passing_stage_order

    passing_fig.update_layout(
        template="pega",
        funnelmode="stack",
        showlegend=True,
        xaxis_title="",
        yaxis_title="Average Actions per Interaction",
        legend=dict(traceorder="reversed", title_text=scope),
    )
    passing_fig.update_xaxes(
        categoryorder="array",
        categoryarray=passing_stage_order,
        tickvals=passing_stage_order,
        ticktext=tick_labels,
    )

    filter_fig = (
        px.bar(
            filtered_df,
            x="actions_per_interaction",
            y=self._decision_data.level,
            color=scope,
            labels={
                self._decision_data.level: "Stage",
                "actions_per_interaction": "Average Filtered Actions per Interaction",
                "action_occurrences": "Total Filtered Action Occurrences",
                "penetration_pct": "Filtered Reach (%)",
            },
            color_discrete_map=color_map,
            category_orders={self._decision_data.level: self._decision_data.AvailableNBADStages},
        )
        .update_traces(
            texttemplate="%{x:.1f}",
            hovertemplate="<b>%{y}</b><br>"
            + "%{fullData.name}<br>"
            + "Average Filtered Actions per Interaction: %{x:.1f}<br>"
            + "Filtered Reach: %{customdata[0]:.1f}%<br>"
            + "Total Filtered: %{customdata[1]:,}<br>"
            + "<extra></extra>",
            customdata=filtered_df.select(["penetration_pct", "action_occurrences"]).to_numpy(),
        )
        .update_layout(
            template="plotly_white",
            xaxis_title="Average Filtered Actions per Interaction",
            yaxis_title="",
        )
    )

    return passing_fig, filter_fig


def decisions_without_actions_plot(
    self,
    additional_filters: pl.Expr | list[pl.Expr] | None = None,
    return_df=False,
):
    """Bar chart showing decisions with no remaining actions per stage, as % of total."""
    df = self._decision_data.aggregates.get_decisions_without_actions_data(additional_filters)
    if return_df:
        return df

    total_decisions = (
        apply_filter(self._decision_data.preaggregated_filter_view, additional_filters)
        .select(pl.col("Interaction_IDs").list.explode(keep_nulls=False, empty_as_null=False).unique().count())
        .collect()
        .item()
    )
    df = df.with_columns((pl.col("decisions_without_actions") / total_decisions * 100).alias("pct_without_actions"))

    return px.bar(
        df,
        x="pct_without_actions",
        y=self._decision_data.level,
        labels={
            self._decision_data.level: "Stage group" if self._decision_data.level == "Stage Group" else "Stage",
            "pct_without_actions": "% of Decisions without Actions",
        },
        category_orders={self._decision_data.level: self._decision_data.AvailableNBADStages},
        template="plotly_white",
        color_discrete_sequence=["#d73027"],
    ).update_layout(
        xaxis_title="% of Decisions without Actions",
        xaxis_ticksuffix="%",
        yaxis_title="",
    )
