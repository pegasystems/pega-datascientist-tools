import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import polars as pl

from .utils import PRIO_FACTORS, apply_filter
from ..utils.pega_template import colorway

DEFAULT_BOXPLOT_POINT_CAP = 20000


class Plot:
    def __init__(self, decision_data):
        self._decision_data = decision_data

    def _boxplot_point_cap(self) -> int:
        sample_size = getattr(self._decision_data, "sample_size", None)
        if isinstance(sample_size, int) and sample_size > 0:
            return sample_size
        return DEFAULT_BOXPLOT_POINT_CAP

    def threshold_deciles(self, thresholding_on, thresholding_name, return_df=False):
        df = self._decision_data.getThresholdingData(thresholding_on)
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

    # @st.cache_data(hash_funcs=polars_lazyframe_hashing)
    def distribution_as_treemap(self, df: pl.LazyFrame, stage: str, scope_options: list[str]):
        # Use consistent color mapping from the DecisionAnalyzer instance
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

    # @st.cache_data(hash_funcs=polars_lazyframe_hashing)
    def sensitivity(
        self,
        win_rank: int = 1,
        hide_priority=True,
        return_df=False,
        reference_group=None,
        additional_filters=None,
    ):
        """
        If reference_group is None, this works as global sensitivity, otherwise it is local sensitivity where the focus is on the refernce_group.

        """
        df = self._decision_data.get_sensitivity(win_rank, reference_group, additional_filters=additional_filters)
        if return_df:
            return df
        n = df.filter(pl.col("Factor") == "Priority").select("Influence").collect().item()
        plotData = df.with_columns(pl.format("{}%", (100.0 * pl.col("Influence") / n).round(2)).alias("Relative"))

        if hide_priority:
            plotData = plotData.filter(pl.col("Factor") != "Priority")
        plotData = plotData.collect()
        range_color = [0, max(0, max(plotData["Influence"]))]
        fig = px.bar(
            data_frame=plotData,
            y="Factor",
            x="Influence",
            text="Relative",
            color="Influence",
            color_continuous_scale="RdYlGn",
            range_color=range_color,
            orientation="h",
            template="pega",
        )

        layout_args = {
            "showlegend": False,
            "yaxis": dict(
                showticklabels=True,
                automargin=True,
                ticklabelposition="outside",
            ),
        }

        fig.update_yaxes(
            autorange="reversed",
            title="Prioritization Factor",
        ).update_xaxes(
            title="Decisions",
            # tickformat="",
        ).update(layout_coloraxis_showscale=False).update_layout(**layout_args)

        return fig

    # @st.cache_data(hash_funcs=polars_lazyframe_hashing)
    def global_winloss_distribution(self, level, win_rank, return_df=False, additional_filters=None):
        # level, cat = getScope(level)
        df = self._decision_data.get_win_loss_distribution_data(level, win_rank, additional_filters=additional_filters)
        if return_df:
            return df

        # Use consistent color mapping from the DecisionAnalyzer instance
        color_discrete_map = self._decision_data.color_mappings.get(level, {})

        # Collect and split data into wins and losses
        df_collected = df.collect()
        wins_df = df_collected.filter(pl.col("Status") == "Wins")
        losses_df = df_collected.filter(pl.col("Status") == "Losses")

        # Create two side-by-side pie charts
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "pie"}, {"type": "pie"}]],
            subplot_titles=["Wins", "Losses"],
        )

        # Add wins pie chart
        if wins_df.height > 0:
            colors_wins = [color_discrete_map.get(val, "#cccccc") for val in wins_df[level].to_list()]
            fig.add_trace(
                go.Pie(
                    labels=wins_df[level],
                    values=wins_df["Percentage"],
                    marker=dict(colors=colors_wins),
                    textposition="auto",
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Add losses pie chart
        if losses_df.height > 0:
            colors_losses = [color_discrete_map.get(val, "#cccccc") for val in losses_df[level].to_list()]
            fig.add_trace(
                go.Pie(
                    labels=losses_df[level],
                    values=losses_df["Percentage"],
                    marker=dict(colors=colors_losses),
                    textposition="auto",
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
                ),
                row=1,
                col=2,
            )

        fig.update_layout(
            font_size=12,
            showlegend=False,  # Labels are shown on the pie slices
        )

        return fig

    def propensity_vs_optionality(self, stage="Arbitration", df=None, return_df=False):
        if df is None:
            df = self._decision_data.sample
        plotData = self._decision_data.get_optionality_data(df).filter(pl.col(self._decision_data.level) == stage)
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

        # Build hover template for bars
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

    def optionality_funnel(self, df):
        level = self._decision_data.level
        plot_data = self._decision_data.get_optionality_funnel(df=df).collect()
        total_interactions = (
            plot_data.filter(pl.col(level) == plot_data.row(0)[0]).select(pl.sum("Interactions")).row(0)[0]
        )
        fig = go.Figure()

        colors = [
            "#d73027",  # 0 actions - dark red (very bad)
            "#fc8d59",  # 1 action - orange
            "#fee090",  # 2 actions - light orange/yellow
            "#ffffbf",  # 3 actions - yellow
            "#e0f3b5",  # 4 actions - light yellow-green
            "#91cf60",  # 5 actions - light green
            "#4dac26",  # 6 actions - medium green
            "#1a9850",  # 7+ actions - dark green (very good)
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

    def action_variation(self, stage="Final", color_by=None, return_df=False):
        """Plot action variation (Lorenz curve showing action concentration).

        Args:
            stage: Stage to analyze
            color_by: Optional dimension to color by (e.g., "Channel/Direction")
            return_df: If True, return the data instead of the figure
        """
        df = self._decision_data.getActionVariationData(stage, color_by=color_by)
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

    def trend_chart(
        self, stage: str, scope: str, return_df=False, additional_filters=None
    ) -> tuple[go.Figure, str | None]:
        df = self._decision_data.get_trend_data(stage, scope, additional_filters=additional_filters).collect()

        if return_df:
            return df.lazy()

        warning_message = None
        if df.select(pl.col("day").n_unique()).get_column("day")[0] <= 1:
            warning_message = (
                "Insufficient data: Trend analysis requires data from multiple days. "
                "Currently, the dataset contains information for only one day."
            )

        color_discrete_map = self._decision_data.color_mappings.get(scope)

        fig = px.area(
            data_frame=df,
            x="day",
            y="Decisions",
            color=scope,
            color_discrete_map=color_discrete_map,
            template="pega",
        )

        fig.update_layout(xaxis_title="")

        return fig, warning_message

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
        available_df, passing_df, filtered_df = self._decision_data.getFunnelData(scope, additional_filters)
        if return_df:
            return available_df, passing_df, filtered_df

        color_map = self._decision_data.color_mappings.get(scope)

        # Product behavior: funnel views stop at the last filtering stage (no Output stage).
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
                metrics_by_stage[synthetic_first_stage] = available_at_first_stage.get(
                    str(scope_value), (0.0, 0.0, 0.0)
                )

            stage_values = passing_stage_order
            metric_values: list[float] = []
            custom_values: list[list[float]] = []
            text_values: list[str] = []
            for stage in stage_values:
                x_val, reach_val, occ_val = metrics_by_stage.get(stage, (0.0, 0.0, 0.0))
                metric_values.append(x_val)
                custom_values.append([reach_val, occ_val])
                text_values.append(f"{x_val:.1f}" if x_val > 0 else "")

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

        passing_fig.update_layout(
            template="pega",
            funnelmode="stack",
            showlegend=True,
            xaxis_title="",
            yaxis_title="Average Actions per Interaction",
            legend=dict(traceorder="reversed", title_text=scope),
        )
        passing_fig.update_xaxes(categoryorder="array", categoryarray=passing_stage_order)

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
        df = self._decision_data.get_decisions_without_actions_data(additional_filters)
        if return_df:
            return df

        total_decisions = (
            apply_filter(self._decision_data.getPreaggregatedFilterView, additional_filters)
            .select(pl.col("Interaction_IDs").flatten().unique().count())
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
        ).update_layout(
            xaxis_title="% of Decisions without Actions",
            xaxis_ticksuffix="%",
            yaxis_title="",
        )

    def filtering_components(
        self,
        stages: list[str],
        top_n,
        AvailableNBADStages,
        additional_filters: pl.Expr | list[pl.Expr] | None = None,
        return_df=False,
    ):
        df = self._decision_data.getFilterComponentData(top_n, additional_filters)
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

        # TODO generalize this
        # Ouch! TODO use the generic stuff from utils
        # order = ["Suitability", "Arbitration", "Eligibility", "Applicability"]
        # index = 0
        # for row in range(1, 3):
        #     for col in range(1, 3):
        #         fig.update_traces(
        #             textposition="auto",
        #             text=top_n_actions_dict[order[index]],
        #             row=row,
        #             col=col,
        #             showlegend=False,  # TODO: still showing...
        #         )
        #         index += 1

        # fig.update_yaxes(showticklabels=False, matches=None, title="").update_xaxes(
        #     title=""
        # )

        # Use annotations for global x and y titles (not per facet)
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
        fig.for_each_annotation(
            lambda a: a.update(text=a.text.split("=")[-1])
        )  # split plotly facet label, show only right side

        return fig

    # @st.cache_data(hash_funcs=polars_lazyframe_hashing)
    def distribution(
        self,
        df: pl.LazyFrame,
        scope: str,
        breakdown: str,
        metric: str = "Decisions",
        horizontal=False,
    ):
        color_discrete_map = self._decision_data.color_mappings.get(breakdown)

        # TODO have a nice hover showing both the individual colored totals as the total bar
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

    # @st.cache_data(hash_funcs=polars_lazyframe_hashing)
    def prio_factor_boxplots(
        self,
        reference: pl.Expr | list[pl.Expr] | None = None,
        return_df=False,
        additional_filters=None,
        others_filter: pl.Expr | list[pl.Expr] | None = None,
    ) -> tuple[go.Figure, str | None]:
        point_cap = self._boxplot_point_cap()
        df = apply_filter(self._decision_data.arbitration_stage, additional_filters)
        prio_factors = PRIO_FACTORS
        tagged = df.with_columns(
            segment=pl.when(reference).then(pl.lit("Comparison Group")).otherwise(pl.lit("Other Offers"))
        )
        if others_filter is not None:
            keep_selected = pl.col("segment") == "Comparison Group"
            others_match = others_filter if isinstance(others_filter, pl.Expr) else pl.all_horizontal(others_filter)
            tagged = tagged.filter(keep_selected | others_match)
        segmented_df = tagged.select(prio_factors + ["segment"]).collect()
        warning_message = None
        if segmented_df.height > point_cap:
            segmented_df = segmented_df.sample(n=point_cap, shuffle=True, seed=1)
            warning_message = f"Showing a representative sample of {point_cap:,} rows to keep the chart responsive."
        if return_df:
            return segmented_df

        if segmented_df.select(pl.col("segment").n_unique()).row(0)[0] == 1:
            warning_message = "Comparison group never survives to Arbitration"
            return None, warning_message

        colors = {
            "Comparison Group": "rgba(76, 120, 168, 0.5)",
            "Other Offers": "rgba(165, 170, 175, 0.5)",
        }

        fig = make_subplots(rows=len(prio_factors), cols=1, subplot_titles=prio_factors)

        for i, metric in enumerate(prio_factors, start=1):
            for _, segment in enumerate(["Comparison Group", "Other Offers"]):
                prio_factor_values = segmented_df.filter(segment=segment).get_column(metric).to_list()
                fig.add_trace(
                    go.Box(
                        x=prio_factor_values,
                        y=[segment] * len(prio_factor_values),
                        name=segment,
                        orientation="h",
                        showlegend=i == 1,  # Adjust legend
                        marker_color=colors[segment],
                    ),
                    row=i,
                    col=1,
                )
                fig.update_yaxes(autorange="reversed", row=i, col=1)  # for correct legend ordering
                if metric == "Propensity":
                    fig.update_xaxes(tickformat=",.0%", row=i, col=1)

        fig.update_layout(height=800, width=600, showlegend=False)
        fig.update_yaxes(automargin=True)

        return fig, warning_message

    def rank_boxplot(
        self,
        reference: pl.Expr | list[pl.Expr] | None = None,
        return_df=False,
        additional_filters=None,
    ):
        point_cap = self._boxplot_point_cap()
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

        # TODO mind the size of plotly express boxes, see solution in ADM Datamart Plots
        fig = px.box(ranks, x="Rank", orientation="h", template="pega")
        return fig.update_layout(height=300, xaxis_title="Rank")

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
        df = self._decision_data.getComponentActionImpact(
            top_n=top_n, scope=scope, additional_filters=additional_filters
        )
        if return_df:
            return df

        # Top components by total filtering volume
        component_totals = (
            df.group_by("Component Name")
            .agg(pl.sum("Filtered Decisions").alias("total"))
            .sort("total", descending=True)
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
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
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
        df = self._decision_data.getComponentDrilldown(
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

        # Create figure without subplots - use overlaying x-axes for different scales
        fig = go.Figure()

        # Primary trace: bar chart of filtered decisions
        fig.add_trace(
            go.Bar(
                y=plot_df[scope],
                x=plot_df["Filtered Decisions"],
                orientation="h",
                name="Filtered Decisions",
                marker_color="#cd001f",
                hovertemplate=("<b>%{y}</b><br>Filtered: %{x}<br><extra></extra>"),
                xaxis="x",  # Primary x-axis
            )
        )

        # Determine which metric to show based on sort_by selection
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

        # Add secondary trace with its own x-axis scale if a metric is selected
        if metric_col:
            non_null_values = plot_df.filter(pl.col(metric_col).is_not_null())
            if non_null_values.height > 0:
                # Convert propensity to percentage for display
                x_values = non_null_values[metric_col]
                if is_propensity:
                    x_values = x_values * 100  # Convert to percentage

                # Build hover template
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
                        xaxis="x2",  # Secondary x-axis with independent scale
                        hovertemplate=hover_template,
                    )
                )

        # Configure layout with overlaying x-axes
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

        # Add secondary x-axis configuration if we have a metric
        if metric_label:
            xaxis2_config = {
                "title": f"{metric_label} (%)" if is_propensity else metric_label,
                "overlaying": "x",
                "side": "top",
            }
            # Format propensity axis as percentage
            if is_propensity:
                xaxis2_config["ticksuffix"] = "%"
            layout_config["xaxis2"] = xaxis2_config

        fig.update_layout(**layout_config)

        return fig

    def optionality_per_stage(self, return_df=False):
        df = self._decision_data.get_optionality_data(self._decision_data.sample)
        if return_df:
            return df

        level = self._decision_data.level
        color_discrete_map = self._decision_data.color_mappings.get(level)

        # TODO mind the size of plotly express boxes, see solution in ADM Datamart Plots
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
        # Collect the data to inspect the unique days
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


def offer_quality_piecharts(
    df: pl.LazyFrame,
    propensityTH,
    AvailableNBADStages,
    return_df=False,
    level="Stage Group",
):
    import math

    value_finder_names = [
        "atleast_one_relevant_action",
        "atleast_one_action",
        "only_irrelevant_actions",
        "has_no_offers",
    ]
    all_frames = (
        df.group_by(level)
        .agg(pl.sum(*value_finder_names))
        .collect()  # type: ignore[union-attribute]
        .partition_by(level, as_dict=True)
    )

    # Filter to only include stages that exist in the data
    df_dict = {}  # type: ignore[assignment]
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

    # Calculate grid dimensions
    num_stages = len(stages_to_plot)
    num_cols = min(4, num_stages)
    num_rows = math.ceil(num_stages / num_cols)

    # Dynamic height: base 400px + 300px per additional row
    fig_height = 400 + (num_rows - 1) * 300

    # Build subplot specs grid, filling partial last row with None
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
    propensityTH,
    level="Stage Group",
):
    """Create a single pie chart showing offer quality for a specific stage.

    Parameters
    ----------
    df : pl.LazyFrame
        Offer quality data from get_offer_quality()
    stage : str
        Stage name to display (e.g., "Arbitration", "Output")
    propensityTH : float
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
    trend_df = (
        df.filter(pl.col(level) == stage).group_by("day").agg(pl.sum(*value_finder_names)).collect().sort("day")  # type: ignore[union-attribute]
    )
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
    # Order to match pie charts: green, blue, orange, red
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


# ECDF sends one point per row to the browser; cap to keep it responsive.
_ECDF_MAX_ROWS = 50_000


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

    collected: pl.DataFrame = value_data.select([granularity] + components).collect()  # type: ignore[assignment]
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


def create_win_distribution_plot(
    data: pl.DataFrame,
    win_count_col: str,
    scope_config: dict[str, str | list[str]],
    title_suffix: str,
    y_axis_title: str,
) -> tuple[go.Figure, pl.DataFrame]:
    """
    Create a win distribution bar chart with highlighted selected items.

    This function creates a bar chart showing win counts across actions, groups, or issues
    based on the scope configuration. It automatically aggregates data appropriately and
    highlights the selected item in red while showing others in grey.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing win distribution data with action identifiers and win counts
    win_count_col : str
        Column name containing win counts to plot (e.g., "original_win_count", "new_win_count")
    scope_config : dict[str, str | list[str]]
        Configuration dictionary from get_scope_config() containing:
        - level: "Action", "Group", or "Issue"
        - group_cols: List of columns for grouping
        - x_col: Column name for x-axis
        - selected_value: Value to highlight in red
        - plot_title_prefix: Prefix for plot title
    title_suffix : str
        Suffix to add to plot title (e.g., "Current Performance", "After Lever Adjustment")
    y_axis_title : str
        Title for y-axis (e.g., "Current Win Count", "New Win Count")

    Returns
    -------
    tuple[go.Figure, pl.DataFrame]
        - Plotly figure with bar chart
        - Processed plot data (aggregated if needed)

    Notes
    -----
    - For Action level: Shows individual actions
    - For Group/Issue level: Automatically aggregates data by summing win counts
    - Selected item is highlighted in red (#FF0000), others in grey
    - "No Winner" bar (if present in data) is shown in orange (#FFA500) to highlight interactions without winners
    - If selected item not found, uses light blue as fallback color
    - X-axis labels are hidden to avoid clutter, scope level shown as x-axis title
    - "No Winner" data is calculated and added by get_win_distribution_data() when all_interactions parameter is provided

    Examples
    --------
    >>> scope_config = get_scope_config("Service", "Cards", "MyAction")
    >>> fig, plot_data = create_win_distribution_plot(
    ...     distribution_data,
    ...     "new_win_count",
    ...     scope_config,
    ...     "After Lever Adjustment",
    ...     "New Win Count"
    ... )
    """
    if scope_config["level"] == "Action":
        plot_data = data
    else:
        # Aggregate data based on scope level, but handle "No Winner" separately
        no_winner_data = data.filter(pl.col("Action") == "No Winner")
        regular_data = data.filter(pl.col("Action") != "No Winner")

        if regular_data.height > 0:
            aggregated_regular = (
                regular_data.group_by(scope_config["group_cols"])
                .agg(pl.sum(win_count_col))
                .sort(win_count_col, descending=True)
            )

            # If we have "No Winner" data, we need to select only the columns that match aggregated_regular
            if no_winner_data.height > 0:
                # Select only the columns that exist in aggregated_regular
                group_cols = list(scope_config["group_cols"])  # type: ignore[arg-type]
                columns_to_keep = group_cols + [win_count_col]
                no_winner_data_selected = no_winner_data.select(columns_to_keep)
                plot_data = pl.concat([aggregated_regular, no_winner_data_selected])
            else:
                plot_data = aggregated_regular
        else:
            # If no regular data, just use no_winner_data (select appropriate columns)
            if no_winner_data.height > 0:
                group_cols = list(scope_config["group_cols"])  # type: ignore[arg-type]
                columns_to_keep = group_cols + [win_count_col]
                plot_data = no_winner_data.select(columns_to_keep)
            else:
                plot_data = pl.DataFrame()

    # Create the plot
    fig = go.Figure()

    # Create hover template based on the level in hierarchy
    if scope_config["x_col"] == "Group" and "Issue" in plot_data.columns:
        # Show pyIssue in hover when level is pyGroup
        hover_template = "<b>%{text}</b><br>Issue: %{customdata}<br>Win Count: %{y}<extra></extra>"
        customdata = plot_data["Issue"]
    elif scope_config["x_col"] == "Action" and "Group" in plot_data.columns and "Issue" in plot_data.columns:
        # Show both pyGroup and pyIssue in hover when level is pyName (Action)
        hover_template = (
            "<b>%{text}</b><br>Group: %{customdata[0]}<br>Issue: %{customdata[1]}<br>Win Count: %{y}<extra></extra>"
        )
        customdata = list(zip(plot_data["Group"], plot_data["Issue"]))
    else:
        # Default hover template
        hover_template = "<b>%{text}</b><br>Win Count: %{y}<extra></extra>"
        customdata = None

    fig.add_trace(
        go.Bar(
            x=plot_data[scope_config["x_col"]],
            y=plot_data[win_count_col],
            text=plot_data[scope_config["x_col"]],
            textposition="auto",
            hovertemplate=hover_template,
            customdata=customdata,
        )
    )

    # Create color scheme with special handling for "No Winner"
    colors = ["grey"] * plot_data.shape[0]
    x_values = list(plot_data[scope_config["x_col"]])

    # Highlight the selected item in red
    try:
        selected_index = x_values.index(scope_config["selected_value"])
        colors[selected_index] = "#FF0000"
    except ValueError:
        # Selected value not found in the data
        pass

    # Highlight "No Winner" in orange if present
    try:
        no_winner_index = x_values.index("No Winner")
        colors[no_winner_index] = "#FFA500"  # Orange color for "No Winner"
    except ValueError:
        # "No Winner" not found in the data
        pass

    # Apply colors, use lightblue as fallback if no special highlighting
    if all(color == "grey" for color in colors):
        fig.data[0]["marker_color"] = "lightblue"
    else:
        fig.data[0]["marker_color"] = colors

    fig.update_yaxes(title=y_axis_title)
    fig.update_xaxes(showticklabels=False, title=scope_config["level"])
    fig.update_layout(
        title=f"{scope_config['plot_title_prefix']} - {title_suffix} (Selected: {scope_config['selected_value']})",
        showlegend=False,
    )

    return fig, plot_data


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
        "#1f77b4",  # Blue for Selected Actions
        "#ff7f0e",  # Orange for Others
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
                        showlegend=i == 1,  # Show legend only for the first plot
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
