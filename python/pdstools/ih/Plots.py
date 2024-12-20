from typing import TYPE_CHECKING, Dict, List, Optional
import polars as pl
import plotly as plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from ..utils.namespaces import LazyNamespace

if TYPE_CHECKING:
    from .IH import IH as IH_Class


class Plots(LazyNamespace):
    def __init__(self, ih: "IH_Class"):
        super().__init__()
        self.ih = ih

    def overall_gauges(
        self,
        metric: str,
        experiment_field: str,
        by: Optional[str] = "Channel",
        reference_values: Optional[Dict[str, float]] = None,
        title: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        plot_data = self.ih.aggregates.summary_success_rates(
            by=[experiment_field, by],
        )

        if return_df:
            return plot_data

        if title is None:
            title = f"{metric} Overall Rates"

        plot_data = plot_data.collect()

        cols = plot_data[by].unique().shape[0]  # TODO can be None
        rows = plot_data[experiment_field].unique().shape[0]

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
                title={"text": f"{row[by]}: {row[experiment_field]}"},
                gauge=gauge,
            )
            r, c = divmod(index, cols)
            fig.add_trace(trace1, row=(r + 1), col=(c + 1))
            index = index + 1

        return fig

    def conversion_overall_gauges(
        self,
        experiment_field: str,
        by: Optional[str] = "Channel",
        reference_values: Optional[Dict[str, float]] = None,
        title: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        return self.overall_gauges(
            metric="Conversion",
            experiment_field=experiment_field,
            by=by,
            reference_values=reference_values,
            title=title,
            return_df=return_df,
        )

    def egagement_overall_gauges(
        self,
        experiment_field: str,
        by: Optional[str] = "Channel",
        reference_values: Optional[Dict[str, float]] = None,
        title: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        return self.overall_gauges(
            metric="Engagement",
            experiment_field=experiment_field,
            by=by,
            reference_values=reference_values,
            title=title,
            return_df=return_df,
        )

    def success_rates_tree_map(
        self,
        metric: str,
        by: Optional[List[str]] = None,
        title: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        if by is None:
            by = [
                f
                for f in ["Direction", "Channel", "Issue", "Group", "Name"]
                if f in self.ih.data.collect_schema().names()
            ]

        plot_data = self.ih.aggregates.summary_success_rates(
            by=by,
        )

        if return_df:
            return plot_data

        if title is None:
            title = f"{metric} Rates for All Actions"

        plot_data = plot_data.collect().with_columns(
            CTR_DisplayValue=pl.col(f"SuccessRate_{metric}").round(3),
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

        return fig

    def conversion_success_rates_tree_map(
        self,
        by: Optional[List[str]] = None,
        title: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        return self.success_rates_tree_map(
            metric="Conversion",
            by=by,
            title=title,
            return_df=return_df,
        )

    def engagement_success_rates_tree_map(
        self,
        by: Optional[List[str]] = None,
        title: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        return self.success_rates_tree_map(
            metric="Engagement",
            by=by,
            title=title,
            return_df=return_df,
        )

    def success_rates_trend_bar(
        self,
        metric: str,
        experiment_field: str,
        every: str = "1d",
        by: Optional[str] = None,
        title: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):

        plot_data = self.ih.aggregates.summary_success_rates(
            every=every,
            by=[experiment_field] + [by],
        )
        if return_df:
            return plot_data

        if title is None:
            title = f"{metric} Rates over Time"

        fig = px.bar(
            plot_data.collect(),
            x="OutcomeTime",
            y=f"SuccessRate_{metric}",
            color=experiment_field,
            error_y=f"StdErr_{metric}",
            facet_row=by,
            barmode="group",
            custom_data=[experiment_field],
            template="pega",
            title=title,
        )
        fig.update_yaxes(tickformat=",.3%").update_layout(xaxis_title=None)
        return fig

    def conversion_success_rates_trend_bar(
        self,
        experiment_field: str,
        every: str = "1d",
        by: Optional[str] = None,
        title: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        return self.success_rates_trend_bar(
            metric="Conversion",
            experiment_field=experiment_field,
            every=every,
            by=by,
            title=title,
            return_df=return_df,
        )

    def engagement_success_rates_trend_bar(
        self,
        experiment_field: str,
        every: str = "1d",
        by: Optional[str] = None,
        title: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        return self.success_rates_trend_bar(
            metric="Engagement",
            experiment_field=experiment_field,
            every=every,
            by=by,
            title=title,
            return_df=return_df,
        )

    def success_rates_trend_line(
        self,
        metric: str,
        every: Optional[str] = "1d",
        by: Optional[str] = None,
        title: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        plot_data = self.ih.aggregates.summary_success_rates(
            every=every,
            by=by,
        )
        if return_df:
            return plot_data

        fig = px.line(
            plot_data.collect(),
            x="OutcomeTime",
            y=f"SuccessRate_{metric}",
            color=by,
            facet_row=by,
            # custom_data=[experiment_field] if experiment_field is not None else None,
            template="pega",
            title=title,
        )

        fig.update_yaxes(tickformat=",.3%").update_layout(xaxis_title=None)
        return fig

    def conversion_success_rates_trend_line(
        self,
        every: Optional[str] = "1d",
        by: Optional[str] = None,
        title: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        return self.success_rates_trend_line(
            metric="Conversion",
            every=every,
            by=by,
            title=title,
            return_df=return_df,
        )

    def engagement_success_rates_trend_line(
        self,
        every: Optional[str] = "1d",
        by: Optional[str] = None,
        title: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        return self.success_rates_trend_line(
            metric="Engagement",
            every=every,
            by=by,
            title=title,
            return_df=return_df,
        )
