from typing import List, Optional, Union, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .utils import NBADScope_Mapping

import polars as pl

from .utils import apply_filter


class Plot:
    def __init__(self, decision_data):
        self._decision_data = decision_data

    def threshold_deciles(self, thresholding_on, thresholding_name, return_df=False):
        df = self._decision_data.getThresholdingData(thresholding_on)
        if return_df:
            return df

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=df["Decile"], y=df["Count"], name="Impressions"))
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
            yaxis_title="Volume",
        )
        fig.update_yaxes(title_text=thresholding_name, secondary_y=True)
        fig.update_yaxes(rangemode="tozero")
        fig.layout.yaxis2.tickformat = ",.2%"
        fig.layout.yaxis2.showgrid = False
        return fig

    # @st.cache_data(hash_funcs=polars_lazyframe_hashing)
    def distribution_as_treemap(
        self, df: pl.LazyFrame, stage: str, scope_options: List[str]
    ):
        fig = px.treemap(
            df.collect(),
            path=[px.Constant(f"All Actions {stage}")] + scope_options,
            values="Decisions",
            template="pega",
        ).update_traces(
            root_color="lightgrey"
        )  # TODO some day we may have colors associated with stages
        return fig

    # @st.cache_data(hash_funcs=polars_lazyframe_hashing)
    def sensitivity(
        self,
        win_rank: int = 1,
        hide_priority=True,
        limit_xaxis_range=True,
        return_df=False,
        reference_group=None,
    ):
        df = self._decision_data.get_sensitivity(win_rank, reference_group)
        if return_df:
            return df
        n = (
            df.filter(pl.col("Factor") == "Priority")
            .select("Influence")
            .collect()
            .item()
        )
        plotData = df.with_columns(
            pl.format("{}%", (100.0 * pl.col("Influence") / n).round(2)).alias(
                "Relative"
            )
        )

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
        if limit_xaxis_range:
            layout_args["xaxis_range"] = [0, n]

        fig.update_yaxes(
            autorange="reversed",
            title="Prioritization Factor",
        ).update_xaxes(
            title="Decisions",
            # tickformat="",
        ).update(layout_coloraxis_showscale=False).update_layout(**layout_args)

        return fig

    # @st.cache_data(hash_funcs=polars_lazyframe_hashing)
    def global_winloss_distribution(self, level, win_rank, return_df=False):
        # level, cat = getScope(level)
        df = self._decision_data.get_win_loss_distribution_data(level, win_rank)
        if return_df:
            return df
        fig = px.bar(
            df.collect(),
            x="Percentage",
            y="Status",
            orientation="h",
            color=level,
            category_orders={"Status": ["Wins", "Losses"]},
        )

        fig.update_layout(
            title=f"Wins and Losses of {NBADScope_Mapping[level]}s in Arbitration",
            font_size=12,
            polar_angularaxis_rotation=90,
            xaxis_title="",
            yaxis_title="",
        )
        fig.update_xaxes(tickformat=".2%").update_layout(
            legend_title_text=f"{NBADScope_Mapping[level]}"
        )

        return fig

    def propensity_vs_optionality(self, stage="Arbitration", df=None, return_df=False):
        if df is None:
            df = self._decision_data.sample
        plotData = self._decision_data.get_optionality_data(df).filter(
            pl.col(self._decision_data.level) == stage
        )
        if return_df:
            return plotData
        plotData = plotData.collect()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=plotData["nOffers"], y=plotData["Interactions"], name="Optionality"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plotData["nOffers"],
                y=plotData["AverageBestPropensity"],
                yaxis="y2",
                name="Propensity",
                mode="markers+lines",
            ),
            secondary_y=True,
        )
        fig.update_layout(
            template="pega",
            xaxis_title="Number of Actions per Customer",
            yaxis_title="Decisions",
        )
        fig.update_yaxes(title_text="Propensity", secondary_y=True)
        fig.layout.yaxis2.tickformat = ",.2%"
        fig.layout.yaxis2.showgrid = False
        return fig

    def optionality_funnel(self, df):
        plot_data = self._decision_data.get_optionality_funnel(df=df).collect()
        total_interactions = (
            plot_data.filter(pl.col("StageGroup") == plot_data.row(0)[0])
            .select(pl.sum("Interactions"))
            .row(0)[0]
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
                ((pl.col("Interactions") / total_interactions) * 100).alias(
                    "percentage"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df_with_percent["StageGroup"],
                    y=df_with_percent["percentage"],
                    mode="lines",
                    stackgroup="one",
                    name=f"{action_count} {'Action' if action_count == '1' else 'Actions'}",
                    line=dict(width=0.5, color=colors[i]),
                    hovertemplate="%{customdata} interactions (%{y:.1f}%)<br>with %{meta}<extra></extra>",
                    customdata=df_with_percent["Interactions"],
                    meta=[
                        f"{action_count} {'action' if action_count == '1' else 'actions'}"
                        for _ in range(len(df_with_percent))
                    ],
                )
            )

        fig.update_layout(
            xaxis_title="Funnel Stage",
            yaxis_title="Percentage of Interactions",
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

    def action_variation(self, stage="Final", return_df=False):
        df = self._decision_data.getActionVariationData(stage)
        if return_df:
            return df
        return (
            px.line(
                df.collect(),
                y="DecisionsFraction",
                x="ActionsFraction",
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
        self, stage: str, scope: str, return_df=False
    ) -> Tuple[go.Figure, Optional[str]]:
        df = self._decision_data.getDistributionData(
            stage,
            scope,
            trend=True,
        ).collect()

        if return_df:
            return df.lazy()

        if df.select(pl.col("day").n_unique()).get_column("day")[0] > 1:
            fig = px.area(
                data_frame=df,
                x="day",
                y="Decisions",
                color=scope,
                template="pega",
            )
            warning_message = None
        else:
            warning_message = (
                "Insufficient data: Trend analysis requires data from multiple days. "
                "Currently, the dataset contains information for only one day. Hence, a trend can't be detected. "
                "A scatter plot will be displayed instead for the available data."
            )
            fig = px.scatter(
                data_frame=df,
                x="day",
                y="Decisions",
                color=scope,
                template="pega",
            )

        fig.update_layout(
            xaxis_title="", legend_title_text=f"{NBADScope_Mapping[scope]}"
        )

        return fig, warning_message

    def decision_funnel(
        self,
        scope: str,
        additional_filters: Optional[Union[pl.Expr, List[pl.Expr]]] = None,
        return_df=False,
    ):
        remaining_df, filter_df = self._decision_data.getFunnelData(
            scope, additional_filters
        )
        if return_df:
            return remaining_df, filter_df

        unique_scope_values = filter_df.select("pyIssue").unique().to_series().to_list()
        colors = px.colors.qualitative.Light24
        color_map = {
            val: colors[i % len(colors)] for i, val in enumerate(unique_scope_values)
        }
        remaining_fig = (
            px.funnel(
                remaining_df.sort(
                    [self._decision_data.level, "count", scope]
                ).collect(),
                y="average_actions",
                x=self._decision_data.level,
                color=scope,
                # title=f"Distribution of {scope}s over the stages",
                hover_data=["count", "average_actions"],
                labels={self._decision_data.level: "Stage"},
                template="pega",
                color_discrete_map=color_map,
            )
            .update_xaxes(
                categoryorder="array",
            )
            .update_layout(
                showlegend=True,
                xaxis_title="",
                legend_title_text=f"{NBADScope_Mapping[scope]}",
                legend=dict(traceorder="reversed"),
            )
        )
        filter_fig = px.bar(
            filter_df,
            x="average_actions",
            y=self._decision_data.level,
            color=scope,
            hover_data=["count", "average_actions"],
            color_discrete_map=color_map,
            category_orders={"StageGroup": self._decision_data.AvailableNBADStages},
        ).update_layout(
            template="plotly_white",
            xaxis_title="Filtered Actions per Decision",
        )

        return remaining_fig, filter_fig

    def filtering_components(
        self,
        stages: List[str],
        top_n,
        AvailableNBADStages,
        additional_filters: Optional[Union[pl.Expr, List[pl.Expr]]] = None,
        return_df=False,
    ):
        df = self._decision_data.getFilterComponentData(top_n, additional_filters)
        if return_df:
            return df
        top_n_actions_dict = {}
        for stage in [x for x in stages if x != "Final"]:
            top_n_actions_dict[stage] = (
                df.filter(pl.col(self._decision_data.level) == stage)
                .get_column("pxComponentName")
                .to_list()
            )

        fig = px.bar(
            df.with_columns(
                pl.col("Filtered Decisions").cast(pl.Float32)
            ),  # TODO expect the data to be float...
            x="Filtered Decisions",
            y="pxComponentName",
            color="Filtered Decisions",
            color_continuous_scale="reds",
            orientation="h",
            facet_col=self._decision_data.level,
            facet_col_wrap=2,
            template="pega",
            category_orders={self._decision_data.level: AvailableNBADStages},
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
            title=f"Top {top_n} filter components",
            font_size=12,
            polar_angularaxis_rotation=90,
            showlegend=False,  # TODO still showing...
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
        # TODO have a nice hover showing both the individual colored totals as the total bar
        fig = px.histogram(
            df.collect(),
            x=metric if horizontal else scope,
            y=scope if horizontal else metric,
            color=breakdown,
            orientation="h" if horizontal else "v",
            template="pega",
        ).update_layout(legend_title_text=f"{NBADScope_Mapping[breakdown]}")

        if horizontal:
            fig = (
                fig.update_xaxes(automargin=True, title=metric)
                .update_yaxes(title="")
                .update_layout(
                    yaxis={"categoryorder": "total ascending"}, xaxis_title_text="Count"
                )
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
        reference: Optional[Union[pl.Expr, List[pl.Expr]]] = None,
        sample_size=10000,
        return_df=False,
    ) -> Tuple[go.Figure, Optional[str]]:
        df = self._decision_data.arbitration_stage
        if return_df:
            return df
        prio_factors = [
            "Propensity",
            "Value",
            "Context Weight",
            "Levers",
        ]  # TODO lets not repeat all over the place, also allow for alias (w/o py etc)
        row_count = df.select("Propensity").collect().height
        sample_size = sample_size if row_count > sample_size else row_count
        segmented_df = (
            (
                df.with_columns(
                    segment=pl.when(reference)  # pl.col("pyName").is_in(models)
                    .then(pl.lit("Selected Actions"))
                    .otherwise(pl.lit("Others"))
                ).select(prio_factors + ["segment"])
            )
            .collect()
            .sample(n=sample_size)
        )

        if segmented_df.select(pl.col("segment").n_unique()).row(0)[0] == 1:
            warning_message = "Action in selected group never survives to Arbitration"
            return None, warning_message

        colors = {
            "Selected Actions": "rgba(76, 120, 168, 0.5)",
            "Others": "rgba(165, 170, 175, 0.5)",
        }

        fig = make_subplots(rows=len(prio_factors), cols=1, subplot_titles=prio_factors)

        for i, metric in enumerate(prio_factors, start=1):
            for _, segment in enumerate(["Selected Actions", "Others"]):
                fig.add_trace(
                    go.Box(
                        x=segmented_df.filter(segment=segment)
                        .get_column(metric)
                        .to_list(),
                        y=[segment] * sample_size,
                        name=segment,
                        orientation="h",
                        showlegend=i == 1,  # Adjust legend
                        marker_color=colors[segment],
                    ),
                    row=i,
                    col=1,
                )
                fig.update_yaxes(
                    autorange="reversed", row=i, col=1
                )  # for correct legend ordering

        fig.update_layout(height=800, width=600, showlegend=False)
        fig.update_yaxes(automargin=True)

        return fig, None

    def rank_boxplot(
        self,
        reference: Optional[Union[pl.Expr, List[pl.Expr]]] = None,
        return_df=False,
    ):
        df = self._decision_data.sample
        if return_df:
            return df
        ranks = (
            apply_filter(df, reference)
            .filter(
                pl.col(self._decision_data.level).is_in(
                    self._decision_data.stages_from_arbitration_down
                )
            )
            .select("pxRank")
            .collect()
        )
        fig = px.box(ranks, x="pxRank", orientation="h", template="pega")
        return fig.update_layout(height=300, xaxis_title="Rank")

    def optionality_per_stage(self, return_df=False):
        df = self._decision_data.get_optionality_data(self.sample)
        if return_df:
            return df
        fig = px.box(
            df.collect(),
            x=self._decision_data.level,
            y="nOffers",
            color=self._decision_data.level,
            template="pega",
        )
        fig.update_layout(
            template="pega",
            title="Number of Actions per Customer",
            xaxis_title="Stage",
            yaxis_title="Number of Actions",
            legend_title_text="Stage",
        )
        fig.update_xaxes(
            categoryorder="array",
            categoryarray=list(self._decision_data.self.AvailableNBADStages),
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
        if unique_days == 1:
            warning = "Insufficient data: Trend analysis requires data from multiple days. Currently, the dataset contains information for only one day. Hence, a trend can't be detected. "

            # Create a scatter plot instead of a line plot
            fig = px.scatter(
                collected_df,
                x="day",
                y="nOffers",
                color=self._decision_data.level,
                template="pega",
            )
        else:
            # Create the line plot as usual
            fig = px.line(
                collected_df,
                x="day",
                y="nOffers",
                color=self._decision_data.level,
                template="pega",
            )

        fig.update_layout(legend_title_text="Stage")
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Number of Unique Offers")

        return fig, warning


def offer_quality_piecharts(
    df: pl.LazyFrame,
    propensityTH,
    AvailableNBADStages,
    return_df=False,
    level="StageGroup",
):
    value_finder_names = [
        "atleast_one_relevant_action",
        "only_irrelevant_actions",
        "has_no_offers",
    ]
    all_frames = (
        df.group_by(level)
        .agg(pl.sum(value_finder_names))
        .collect()
        .partition_by(level, as_dict=True)
    )
    # TODO Temporary solution to fit the pie charts into the screen, pick only first 5 stages
    df = {}
    AvailableNBADStages = AvailableNBADStages[:5]
    for stage in AvailableNBADStages[:5]:
        df[(stage,)] = all_frames[(stage,)]
    if return_df:
        return df

    fig = make_subplots(
        rows=1,
        cols=len(AvailableNBADStages),
        specs=[[{"type": "domain"}] * len(AvailableNBADStages)],
        subplot_titles=AvailableNBADStages,
        horizontal_spacing=0.1,
    )

    for i, stage in enumerate(AvailableNBADStages):
        plotdf = df[(stage,)].drop(level)
        fig.add_trace(
            go.Pie(
                values=list(plotdf.to_numpy())[0],
                labels=list(
                    plotdf.rename(
                        {
                            "atleast_one_relevant_action": "At least one relevant action",
                            "only_irrelevant_actions": "Only irrelevant actions",
                            "has_no_offers": "Without actions",
                        }
                    ).columns
                ),
                name=stage,
                # visible=False,
                sort=False,
            ),
            1,
            i + 1,
        )

    rounding = 3
    fig.update_layout(
        title_text=f"Distribution of customers per stage at propensity threshold {round(float(propensityTH), rounding):.1%}",
    )
    fig.update_traces(marker=dict(colors=["#219e3f", "#fca52e", "#cd001f"]))
    return fig


def getTrendChart(
    df: pl.LazyFrame, stage: str = "Output", return_df=False, level="StageGroup"
):
    value_finder_names = [
        "atleast_one_relevant_action",
        "only_irrelevant_actions",
        "has_no_offers",
    ]
    df = (
        df.filter(pl.col(level) == stage)
        .group_by("day")
        .agg(pl.sum(value_finder_names))
        .collect()
    ).sort("day")
    if return_df:
        return df.lazy()
    trend_melted = (
        df.melt(
            id_vars=["day"],
            value_vars=[
                "has_no_offers",
                "atleast_one_relevant_action",
                "only_irrelevant_actions",
            ],
            variable_name="status",
        )
        .sort("day")
        .rename({"value": "interactions"})
    )
    fig = px.line(
        trend_melted,
        x="day",
        y="interactions",
        color="status",
        title=f"Interactions in Trouble at {stage} stage",
    )

    return fig


def plot_priority_component_distribution(
    value_data: pl.LazyFrame, component: str, granularity: str
):
    histogram = px.histogram(
        value_data.collect(),
        x=component,
        nbins=20,
        title=f"{component} Distribution",
        color=granularity,
        template="pega",
    ).update_layout(
        legend_title_text=NBADScope_Mapping[granularity],
        xaxis_title=component,
        yaxis_title="Number of Actions",
    )

    box_plot = px.box(
        value_data.collect(),
        x=granularity,
        y=component,
        title=f"{component} Distribution by Issue",
        template="pega",
    ).update_layout(
        xaxis_title=NBADScope_Mapping[granularity],
        yaxis_title=component,
        showlegend=False,
    )

    return histogram, box_plot
