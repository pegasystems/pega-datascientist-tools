from typing import TYPE_CHECKING, Dict, List, Optional
import polars as pl
import plotly.io as pio
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

    def experiment_gauges(
        self,
        experiment_field: str,
        by: Optional[str] = "Channel",
        positive_labels: Optional[List[str]] = None,
        negative_labels: Optional[List[str]] = None,
        reference_values: Optional[Dict] = None,
        title: Optional[str] = "Experiment Overview",
        return_df:Optional[bool] = False,
    ):
        # TODO currently only supporting a single by

        plot_data = self.ih.aggregates.summary_by_experiment(
            experiment_field=experiment_field,
            by=by,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
        )

        if return_df:
            return plot_data
        
        plot_data = plot_data.collect()

        cols = plot_data[by].unique().shape[0]
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
                if row["CTR"] < ref_value:
                    gauge = {
                        "axis": {"tickformat": ",.2%"},
                        "bar": {
                            "color": (
                                "#EC5300"
                                if row["CTR"] < (0.75 * ref_value)
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
                value=row["CTR"],
                delta={"reference": ref_value, "valueformat": ",.2%"},
                title={"text": f"{row[by]}: {row[experiment_field]}"},
                gauge=gauge,
            )
            r, c = divmod(index, cols)
            fig.add_trace(trace1, row=(r + 1), col=(c + 1))
            index = index + 1

        return fig

    def tree_map(
        self,
        experiment_field: str,
        by: Optional[List[str]] = None,
        positive_labels: List[str] = None,
        negative_labels: List[str] = None,
        title: Optional[str] = "Detailed Click Through Rates",
        return_df:Optional[bool] = False,
    ):
        if by is None:
            by = [f for f in ["Channel", "Issue", "Group", "Name"] if f in self.ih.data.collect_schema().names()]

        plot_data = self.ih.aggregates.summary_by_experiment(
            experiment_field=experiment_field,
            by=by,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
        )

        if return_df:
            return plot_data
        
        plot_data = plot_data.collect()

        fig = px.treemap(
            plot_data.with_columns(
                CTR_DisplayValue=pl.col("CTR").round(3),
            ),
            path=[px.Constant("ALL")] + [experiment_field] + by,
            values="CTR_DisplayValue",
            color="CTR_DisplayValue",
            color_continuous_scale=px.colors.sequential.RdBu,
            title=title,
            hover_data=["StdErr", "Positives", "Negatives"],
            height=640,
        )
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textinfo="label+value")
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

        return fig
