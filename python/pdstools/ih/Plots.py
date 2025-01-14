import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import polars as pl

from ..utils import cdh_utils
from ..utils.namespaces import LazyNamespace
from ..utils.types import QUERY

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from .IH import IH as IH_Class
try:
    import plotly as plotly
    import plotly.express as px


except ImportError as e:  # pragma: no cover
    logger.debug(f"Failed to import optional dependencies: {e}")


class Plots(LazyNamespace):
    def __init__(self, ih: "IH_Class"):
        super().__init__()
        self.ih = ih

    def overall_gauges(
        self,
        condition: Union[str, pl.Expr],
        *,
        metric: Optional[str] = "Engagement",
        by: Optional[str] = "Channel",
        reference_values: Optional[Dict[str, float]] = None,
        title: Optional[str] = None,
        query: Optional[QUERY] = None,
        # facet: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        import plotly as plotly
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        plot_data = self.ih.aggregates.summary_success_rates(
            by=[condition, by], query=query
        )

        if return_df:
            return plot_data

        if title is None:
            title = f"{metric} Overall Rates"

        plot_data = plot_data.collect()

        cols = plot_data[by].unique().shape[0]  # TODO can be None
        rows = (
            plot_data[condition].unique().shape[0]
        )  # TODO generalize to support pl expression, see ADM plots, eg facet in bubble chart

        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{"type": "indicator"} for c in range(cols)] for t in range(rows)],
        )
        fig.update_layout(
            height=270 * rows,
            autosize=True,
            title=title,
            margin=dict(b=10, t=120, l=10, r=10),
        )
        index = 0
        for row in plot_data.iter_rows(named=True):
            ref_value = (
                reference_values.get(row[by], None) if reference_values else None
            )
            gauge = {
                "axis": {"tickformat": ",.2%"},
                "threshold": {
                    "line": {"color": "red", "width": 2},
                    "thickness": 0.75,
                    "value": ref_value,
                },
            }
            if ref_value:
                if row[f"SuccessRate_{metric}"] < ref_value:
                    gauge = {
                        "axis": {"tickformat": ",.2%"},
                        "bar": {
                            "color": (
                                "#EC5300"
                                if row[f"SuccessRate_{metric}"] < (0.75 * ref_value)
                                else "#EC9B00"
                            )
                        },
                        "threshold": {
                            "line": {"color": "red", "width": 2},
                            "thickness": 0.75,
                            "value": ref_value,
                        },
                    }

            trace1 = go.Indicator(
                mode="gauge+number+delta",
                number={"valueformat": ",.2%"},
                value=row[f"SuccessRate_{metric}"],
                delta={"reference": ref_value, "valueformat": ",.2%"},
                title={"text": f"{row[by]}: {row[condition]}"},
                gauge=gauge,
            )
            r, c = divmod(index, cols)
            fig.add_trace(trace1, row=(r + 1), col=(c + 1))
            index = index + 1

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        return fig

    def response_count_tree_map(
        self,
        *,
        by: Optional[List[str]] = None,
        title: Optional[str] = None,
        query: Optional[QUERY] = None,
        # facet: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        import plotly as plotly
        import plotly.express as px

        if by is None:
            by = [
                f
                for f in ["Direction", "Channel", "Issue", "Group", "Name"]
                if f in self.ih.data.collect_schema().names()
            ]
        elif isinstance(by, str):
            by = [by]

        plot_data = self.ih.aggregates.summary_outcomes(
            by=by,
            query=query,
        )
        if return_df:
            return plot_data

        fig = px.treemap(
            plot_data.collect(),
            path=[px.Constant("ALL")] + ["Outcome"] + by,
            values="Count",
            color="Count",
            branchvalues="total",
            # color_continuous_scale=px.colors.sequential.RdBu_r,
            title=title,
            height=640,
            template="pega",
        )
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textinfo="label+value+percent parent")
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        return fig

    def success_rate_tree_map(
        self,
        *,
        metric: Optional[str] = "Engagement",
        by: Optional[List[str]] = None,
        title: Optional[str] = None,
        query: Optional[QUERY] = None,
        # facet: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        import plotly as plotly
        import plotly.express as px

        if by is None:
            by = [
                f
                for f in ["Direction", "Channel", "Issue", "Group", "Name"]
                if f in self.ih.data.collect_schema().names()
            ]

        plot_data = self.ih.aggregates.summary_success_rates(by=by, query=query)

        if return_df:
            return plot_data

        if title is None:
            title = f"{metric} Rates for All Actions"

        plot_data = (
            plot_data.collect()
            .with_columns(
                CTR_DisplayValue=pl.col(f"SuccessRate_{metric}").round(3),
            )
            .filter(pl.col(f"SuccessRate_{metric}") > 0)
        )

        fig = px.treemap(
            plot_data,
            path=[px.Constant("ALL")] + by,
            values="CTR_DisplayValue",
            color="CTR_DisplayValue",
            color_continuous_scale=px.colors.sequential.RdBu,
            title=title,
            hover_data=[
                f"StdErr_{metric}",
                f"Positives_{metric}",
                f"Negatives_{metric}",
            ],
            height=640,
            template="pega",
        )
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textinfo="label+value")
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        return fig

    def action_distribution(
        self,
        *,
        by: Optional[str] = "Name",
        title: Optional[str] = "Action Distribution",
        query: Optional[QUERY] = None,
        color: Optional[str] = None,
        facet: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        plot_data = self.ih.aggregates.summary_outcomes(
            by=[by, color, facet], query=query
        )

        if return_df:
            return plot_data

        fig = px.bar(
            plot_data.collect(),
            x="Count",
            y=by,
            color=color,
            facet_col=facet,
            template="pega",
            title=title,
        )

        fig.update_layout(barmode="stack")
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_layout(yaxis=dict(title=""))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        return fig

    # def success_rates_trend_bar(
    #     self,
    #     condition: Union[str, pl.Expr],
    #     *,
    #     metric: Optional[str] = "Engagement",
    #     every: Union[str, timedelta] = "1d",
    #     by: Optional[str] = None,
    #     title: Optional[str] = None,
    #     query: Optional[QUERY] = None,
    #     facet: Optional[str] = None,
    #     return_df: Optional[bool] = False,
    # ):

    #     plot_data = self.ih.aggregates.summary_success_rates(
    #         every=every,
    #         by=[condition] + [by],  # TODO generalize to support pl expression
    #         query=query,
    #     )

    #     if return_df:
    #         return plot_data

    #     if title is None:
    #         title = f"{metric} Rates over Time"

    #     fig = px.bar(
    #         plot_data.collect(),
    #         x="OutcomeTime",
    #         y=f"SuccessRate_{metric}",
    #         color=condition,
    #         error_y=f"StdErr_{metric}",
    #         facet_row=by,
    #         barmode="group",
    #         custom_data=[condition],
    #         template="pega",
    #         title=title,
    #     )
    #     fig.update_yaxes(tickformat=",.3%").update_layout(xaxis_title=None)
    #     return fig

    def success_rate(
        self,
        *,
        metric: Optional[str] = "Engagement",
        every: Union[str, timedelta] = "1d",
        title: Optional[str] = None,
        query: Optional[QUERY] = None,
        facet: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        import plotly as plotly
        import plotly.express as px

        plot_data = self.ih.aggregates.summary_success_rates(
            every=every, by=facet, query=query
        )

        if return_df:
            return plot_data

        if title is None:
            title = f"Success Rates Trend of {metric}"

        fig = px.line(
            plot_data.collect(),
            x="OutcomeTime",
            y=f"SuccessRate_{metric}",
            color=facet,
            facet_row=facet,
            # custom_data=[experiment_field] if experiment_field is not None else None,
            template="pega",
            title=title,
        )

        fig.update_yaxes(tickformat=",.3%", title=None).update_layout(xaxis_title=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        return fig

    def response_count(
        self,
        *,
        every: Union[str, timedelta] = "1d",
        title: Optional[str] = "Responses",
        query: Optional[QUERY] = None,
        facet: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        import plotly as plotly
        import plotly.express as px

        plot_data = self.ih.aggregates.ih.aggregates.summary_outcomes(
            every=every, by=facet, query=query
        ).collect()

        if return_df:
            return plot_data.lazy()

        fig = px.bar(
            plot_data,
            x="OutcomeTime",
            y="Count",
            color="Outcome",
            template="pega",
            title=title,
            facet_row=facet,
        )
        fig.update_layout(xaxis_title=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        return fig

    def model_performance_trend(
        self,
        *,
        metric: Optional[str] = "Engagement",
        every: Union[str, timedelta] = "1d",
        by: Optional[str] = None,
        title: Optional[str] = "Model Performance over Time",
        query: Optional[QUERY] = None,
        facet: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        import plotly as plotly
        import plotly.express as px

        plot_data = (
            self.ih.aggregates._summary_interactions(
                every=every, by=cdh_utils.safe_flatten_list([by, facet]), query=query
            )
            .filter(
                pl.col.Propensity.is_not_null()
                & pl.col(f"Interaction_Outcome_{metric}").is_not_null()
            )
            .group_by(cdh_utils.safe_flatten_list([by, facet, "OutcomeTime"]))
            .agg(
                pl.map_groups(
                    exprs=[f"Interaction_Outcome_{metric}", "Propensity"],
                    function=lambda data: cdh_utils.auc_from_probs(data[0], data[1]),
                    return_dtype=pl.Float64,
                ).alias("Performance")
            )
            .sort(["OutcomeTime"])
        ).with_columns(pl.col("Performance") * 100)

        if return_df:
            return plot_data

        fig = px.line(
            plot_data.collect(),
            y="Performance",
            x="OutcomeTime",
            color=by,
            facet_row=facet,
            template="pega",
            title=title,
        )

        fig.update_layout(yaxis=dict(range=[50, None]), xaxis_title=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        return fig
