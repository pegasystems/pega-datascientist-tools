from typing import List, Optional, Union, Tuple, TYPE_CHECKING, cast
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .utils import NBADScope_Mapping

import polars as pl

from .utils import apply_filter


class Plot:
    def __init__(self, decision_data):
        self._decision_data = decision_data

    def plot_threshold_deciles(self, thresholding_name, return_df=False):
        df = self._decision_data.whatever_preprocessing
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
    def plot_distribution_as_treemap(
        self, df: pl.LazyFrame, stage: str, scope_options: List[str]
    ):
        NBADStages_Mapping = self._decision_data.NBADStages_Mapping
        fig = px.treemap(
            df.collect(),
            path=[px.Constant(f"All Actions {NBADStages_Mapping[stage]}")]
            + scope_options,
            values="Decisions",
            template="pega",
        ).update_traces(
            root_color="lightgrey"
        )  # TODO some day we may have colors associated with stages
        return fig

    # @st.cache_data(hash_funcs=polars_lazyframe_hashing)
    def plot_sensitivity(
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
        fig = px.bar(
            data_frame=plotData.collect(),
            y="Factor",
            x="Influence",
            text="Relative",
            color="Influence",
            color_continuous_scale="Reds",
            range_color=[0, n],
            orientation="h",
            template="pega",
        )

        layout_args = {"showlegend": False}
        if limit_xaxis_range:
            layout_args["xaxis_range"] = [0, n]

        fig.update_yaxes(
            autorange="reversed",
            title="Prioritization Factor",
        ).update_xaxes(
            title="Decisions",
            # tickformat="",
        ).update(
            layout_coloraxis_showscale=False
        ).update_layout(
            **layout_args
        )

        return fig

    # @st.cache_data(hash_funcs=polars_lazyframe_hashing)
    def plot_global_winloss_distribution(self, level, win_rank, return_df=False):
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

    def plot_propensity_vs_optionality(self, stage="Arbitration", return_df=False):
        plotData = (
            self._decision_data.get_optionality_data.filter(
                pl.col("pxEngagementStage") == stage
            )
            .collect()
            .to_pandas(use_pyarrow_extension_array=True)
        )
        if return_df:
            return pl.from_pandas(plotData)

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

    def plot_action_variation(self, stage="Final", return_df=False):
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

    def plot_trend_chart(
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

    # @st.cache_data(hash_funcs=polars_lazyframe_hashing)
    def plot_decision_funnel(
        self,
        scope: str,
        NBADStages_Mapping: dict,
        additional_filters: Optional[Union[pl.Expr, List[pl.Expr]]] = None,
        return_df=False,
        # models=[],  # trick to make streamlit caching work even if dataframe has filters applied
    ):
        df = self._decision_data.getFunnelData(scope, additional_filters)
        if return_df:
            return df
        fig = (
            px.funnel(
                df.with_columns(
                    # TODO perhaps the re-mapping of stage names can be done in plotly as well
                    # instead of changing the data like we do here
                    pl.col("pxEngagementStage")
                    .replace(
                        NBADStages_Mapping
                    )  # Replacing with "remaining" view labels
                    .cast(pl.Enum(list(NBADStages_Mapping.values())))
                )
                .sort(["pxEngagementStage", "count", scope])
                .collect()
                .to_pandas(use_pyarrow_extension_array=True),
                y="count",
                x="pxEngagementStage",
                color=scope,
                # title=f"Distribution of {scope}s over the stages",
                hover_data=["count"],
                labels={"pxEngagementStage": "Stage"},
                template="pega",
            )
            .update_xaxes(
                categoryorder="array",
                categoryarray=list(NBADStages_Mapping.values()),
            )
            .update_layout(
                showlegend=True,
                xaxis_title="",
                legend_title_text=f"{NBADScope_Mapping[scope]}",
            )
        )
        return fig

    def plot_filtering_components(
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
                df.filter(pl.col("pxEngagementStage") == stage)
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
            facet_col="pxEngagementStage",
            facet_col_wrap=2,
            template="pega",
            category_orders={"pxEngagementStage": AvailableNBADStages},
        )

        # TODO generalize this
        # Ouch! TODO use the generic stuff from utils
        order = ["Suitability", "Arbitration", "Eligibility", "Applicability"]
        index = 0
        for row in range(1, 3):
            for col in range(1, 3):
                fig.update_traces(
                    textposition="auto",
                    text=top_n_actions_dict[order[index]],
                    row=row,
                    col=col,
                    showlegend=False,  # TODO: still showing...
                )
                index += 1

        fig.update_yaxes(showticklabels=False, matches=None, title="").update_xaxes(
            title=""
        )

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
    def plot_distribution(
        self,
        df: pl.LazyFrame,
        scope: str,
        breakdown: str,
        metric: str = "Decisions",
        horizontal=False,
    ):
        # TODO have a nice hover showing both the individual colored totals as the total bar
        fig = px.histogram(
            df.collect().to_pandas(),
            x=metric if horizontal else scope,
            y=scope if horizontal else metric,
            color=breakdown,
            orientation="h" if horizontal else "v",
            template="pega",
        ).update_layout(legend_title_text=f"{NBADScope_Mapping[scope]}")

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
    def plot_prio_factor_boxplots(
        self,
        reference: Optional[Union[pl.Expr, List[pl.Expr]]] = None,
        sample_size=10000,
        return_df=False,
    ) -> Tuple[go.Figure, Optional[str]]:
        df = self._decision_data.arbitration_stage
        if return_df:
            return df
        prio_factors = [
            "FinalPropensity",
            "Value",
            "ContextWeight",
            "Weight",
        ]  # TODO lets not repeat all over the place, also allow for alias (w/o py etc)
        row_count = df.select("FinalPropensity").collect().height
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

    def plot_rank_boxplot(
        self,
        reference: Optional[Union[pl.Expr, List[pl.Expr]]] = None,
        return_df=False,
    ):
        df = self._decision_data.sample
        if return_df:
            return df
        ranks = (
            # TODO: generalize ["Arbitration", "Final"], consider using generic aggregation func
            apply_filter(df, reference)
            .filter(pl.col("pxEngagementStage").is_in(["Arbitration", "Final"]))
            .select("pxRank")
            .collect()
        )
        fig = px.box(ranks, x="pxRank", orientation="h", template="pega")
        return fig.update_layout(height=300, xaxis_title="Rank")

    def plot_optionality_per_stage(self, return_df=False):
        df = self._decision_data.get_optionality_data
        if return_df:
            return df
        fig = px.box(
            df.with_columns(
                # TODO perhaps the re-mapping of stage names can be done in plotly as well
                # instead of changing the data like we do here
                pl.col("pxEngagementStage")
                .replace(
                    self._decision_data.NBADStages_Mapping
                )  # Replacing with "remaining" view labels
                .cast(pl.Enum(list(self._decision_data.NBADStages_Mapping.values())))
            )
            .collect()
            .to_pandas(use_pyarrow_extension_array=True),
            x="pxEngagementStage",
            y="nOffers",
            color="pxEngagementStage",
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
            categoryarray=list(self._decision_data.NBADStages_Mapping.values()),
            title="",
        )

        return fig

    def plot_optionality_trend(
        self, df: pl.LazyFrame, NBADStages_Mapping, return_df=False
    ):
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
                collected_df.with_columns(
                    pl.col("pxEngagementStage")
                    .replace(
                        NBADStages_Mapping
                    )  # Replacing with "remaining" view labels
                    .cast(pl.Enum(list(NBADStages_Mapping.values())))
                ).to_pandas(),
                x="day",
                y="nOffers",
                color="pxEngagementStage",
                template="pega",
            )
        else:
            # Create the line plot as usual
            fig = px.line(
                collected_df.with_columns(
                    pl.col("pxEngagementStage")
                    .replace(
                        NBADStages_Mapping
                    )  # Replacing with "remaining" view labels
                    .cast(pl.Enum(list(NBADStages_Mapping.values())))
                ).to_pandas(),
                x="day",
                y="nOffers",
                color="pxEngagementStage",
                template="pega",
            )

        fig.update_layout(legend_title_text="Stage")
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Number of Unique Offers")

        return fig, warning


def plot_offer_quality_piecharts(
    df: pl.LazyFrame,
    propensityTH,
    NBADStages_FilterView,
    NBADStages_Mapping,
    return_df=False,
):
    value_finder_names = [
        "atleast_one_relevant_action",
        "only_irrelevant_actions",
        "has_no_offers",
    ]
    df = (
        df.group_by("pxEngagementStage")
        .agg(pl.sum(value_finder_names))
        .collect()
        .partition_by("pxEngagementStage", as_dict=True)
    )
    if return_df:
        return df

    fig = make_subplots(
        rows=1,
        cols=len(NBADStages_FilterView),
        specs=[[{"type": "domain"}] * len(NBADStages_FilterView)],
        subplot_titles=[NBADStages_Mapping[v] for v in NBADStages_FilterView],
        horizontal_spacing=0.1,
    )

    for i, stage in enumerate(NBADStages_FilterView):
        plotdf = df[stage].drop("pxEngagementStage")
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


def getTrendChart(df: pl.LazyFrame, stage: str = "Final", return_df=False):
    value_finder_names = [
        "atleast_one_relevant_action",
        "only_irrelevant_actions",
        "has_no_offers",
    ]
    df = (
        df.filter(pl.col("pxEngagementStage") == stage)
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


def plot_value_distribution(value_data: pl.LazyFrame, scope: str):
    fig = px.histogram(
        value_data.collect(),
        x="Value_max",
        nbins=20,
        title="Value Distribution",
        color=scope,
        template="pega",
    ).update_layout(
        legend_title_text=f"{NBADScope_Mapping[scope]}",
        xaxis_title="Value",
        yaxis_title="Number of Actions",
    )
    return fig
