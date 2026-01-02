"""Plotting utilities for Interaction History visualization."""

import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import polars as pl

from ..utils import cdh_utils
from ..utils.namespaces import LazyNamespace
from ..utils.types import QUERY

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from plotly.graph_objs import Figure

    from .IH import IH as IH_Class

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    logger.debug(f"Failed to import optional dependencies: {e}")


class Plots(LazyNamespace):
    """Visualization methods for Interaction History data.

    This class provides plotting capabilities for analyzing customer
    interaction data. It is accessed through the `plot` attribute of an
    :class:`~pdstools.ih.IH.IH` instance.

    All plot methods support:
    - Custom titles via `title` parameter
    - Data filtering via `query` parameter
    - Returning underlying data via `return_df=True`

    Attributes
    ----------
    ih : IH
        Reference to the parent IH instance.

    See Also
    --------
    pdstools.ih.IH : Main analysis class.
    pdstools.ih.Aggregates : Aggregation methods.

    Examples
    --------
    >>> ih = IH.from_ds_export("interaction_history.zip")
    >>> ih.plot.success_rate(metric="Engagement")
    >>> ih.plot.response_count_tree_map()
    """

    def __init__(self, ih: "IH_Class"):
        """Initialize a Plots instance.

        Parameters
        ----------
        ih : IH
            The parent IH instance providing the data.
        """
        super().__init__()
        self.ih = ih

    def overall_gauges(
        self,
        condition: Union[str, pl.Expr],
        *,
        metric: str = "Engagement",
        by: str = "Channel",
        reference_values: Optional[Dict[str, float]] = None,
        title: Optional[str] = None,
        query: Optional[QUERY] = None,
        return_df: bool = False,
    ) -> Union["Figure", pl.LazyFrame]:
        """Create gauge charts showing success rates by condition and dimension.

        Generates a grid of gauge charts displaying success rates for combinations
        of the specified condition and grouping dimension, with optional reference
        values for comparison.

        Parameters
        ----------
        condition : Union[str, pl.Expr]
            Column to condition on (e.g., "Experiment", "Issue").
        metric : str, default "Engagement"
            Metric to display: "Engagement", "Conversion", or "OpenRate".
        by : str, default "Channel"
            Dimension for grouping (column name).
        reference_values : Dict[str, float], optional
            Reference values per dimension for comparison thresholds.
        title : str, optional
            Custom title. If None, auto-generated from metric.
        query : QUERY, optional
            Polars expression to filter the data.
        return_df : bool, default False
            If True, return data as LazyFrame instead of figure.

        Returns
        -------
        Figure or pl.LazyFrame
            Plotly figure, or LazyFrame if `return_df=True`.

        Examples
        --------
        >>> ih.plot.overall_gauges("Issue", metric="Engagement", by="Channel")
        """
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
        return_df: bool = False,
    ) -> Union["Figure", pl.LazyFrame]:
        """Create a treemap of response count distribution.

        Displays hierarchical response counts across dimensions, allowing
        exploration of how responses are distributed across categories.

        Parameters
        ----------
        by : List[str], optional
            Hierarchy dimensions. Defaults to Direction, Channel, Issue, Group, Name.
        title : str, optional
            Custom title.
        query : QUERY, optional
            Polars expression to filter the data.
        return_df : bool, default False
            If True, return data as LazyFrame instead of figure.

        Returns
        -------
        Figure or pl.LazyFrame
            Plotly treemap, or LazyFrame if `return_df=True`.

        See Also
        --------
        success_rate_tree_map : Treemap colored by success rates.

        Examples
        --------
        >>> ih.plot.response_count_tree_map()
        >>> ih.plot.response_count_tree_map(by=["Channel", "Issue", "Name"])
        """

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
        metric: str = "Engagement",
        by: Optional[List[str]] = None,
        title: Optional[str] = None,
        query: Optional[QUERY] = None,
        return_df: bool = False,
    ) -> Union["Figure", pl.LazyFrame]:
        """Create a treemap colored by success rates.

        Displays hierarchical success rates across dimensions, with color
        indicating performance levels for easy identification of high and
        low performing areas.

        Parameters
        ----------
        metric : str, default "Engagement"
            Metric to display: "Engagement", "Conversion", or "OpenRate".
        by : List[str], optional
            Hierarchy dimensions. Defaults to Direction, Channel, Issue, Group, Name.
        title : str, optional
            Custom title. If None, auto-generated from metric.
        query : QUERY, optional
            Polars expression to filter the data.
        return_df : bool, default False
            If True, return data as LazyFrame instead of figure.

        Returns
        -------
        Figure or pl.LazyFrame
            Plotly treemap, or LazyFrame if `return_df=True`.

        See Also
        --------
        response_count_tree_map : Treemap by response counts.

        Examples
        --------
        >>> ih.plot.success_rate_tree_map(metric="Conversion")
        """

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
        by: str = "Name",
        title: str = "Action Distribution",
        query: Optional[QUERY] = None,
        color: Optional[str] = None,
        facet: Optional[str] = None,
        return_df: bool = False,
    ) -> Union["Figure", pl.LazyFrame]:
        """Create a bar chart of action distribution.

        Displays action counts across categories, optionally colored and
        faceted by additional dimensions.

        Parameters
        ----------
        by : str, default "Name"
            Dimension for y-axis categories.
        title : str, default "Action Distribution"
            Chart title.
        query : QUERY, optional
            Polars expression to filter the data.
        color : str, optional
            Dimension for bar coloring.
        facet : str, optional
            Dimension for faceting.
        return_df : bool, default False
            If True, return data as LazyFrame instead of figure.

        Returns
        -------
        Figure or pl.LazyFrame
            Plotly bar chart, or LazyFrame if `return_df=True`.

        Examples
        --------
        >>> ih.plot.action_distribution(by="Name", color="Channel")
        """
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
        metric: str = "Engagement",
        every: Union[str, timedelta] = "1d",
        title: Optional[str] = None,
        query: Optional[QUERY] = None,
        facet: Optional[str] = None,
        return_df: bool = False,
    ) -> Union["Figure", pl.LazyFrame]:
        """Create a line chart of success rates over time.

        Displays success rate trends for the specified metric, optionally
        faceted by dimension for comparative analysis.

        Parameters
        ----------
        metric : str, default "Engagement"
            Metric to display: "Engagement", "Conversion", or "OpenRate".
        every : str or timedelta, default "1d"
            Time aggregation period (e.g., "1d", "1w", "1mo").
        title : str, optional
            Custom title. If None, auto-generated from metric.
        query : QUERY, optional
            Polars expression to filter the data.
        facet : str, optional
            Dimension for faceting and coloring.
        return_df : bool, default False
            If True, return data as LazyFrame instead of figure.

        Returns
        -------
        Figure or pl.LazyFrame
            Plotly line chart, or LazyFrame if `return_df=True`.

        See Also
        --------
        response_count : Response counts over time.
        model_performance_trend : Model AUC over time.

        Examples
        --------
        >>> ih.plot.success_rate(metric="Conversion", every="1w")
        """

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
        title: str = "Responses",
        query: Optional[QUERY] = None,
        facet: Optional[str] = None,
        return_df: bool = False,
    ) -> Union["Figure", pl.LazyFrame]:
        """Create a bar chart of response counts over time.

        Displays response counts colored by outcome type, optionally
        faceted by dimension.

        Parameters
        ----------
        every : str or timedelta, default "1d"
            Time aggregation period (e.g., "1d", "1w", "1mo").
        title : str, default "Responses"
            Chart title.
        query : QUERY, optional
            Polars expression to filter the data.
        facet : str, optional
            Dimension for faceting.
        return_df : bool, default False
            If True, return data as LazyFrame instead of figure.

        Returns
        -------
        Figure or pl.LazyFrame
            Plotly bar chart, or LazyFrame if `return_df=True`.

        See Also
        --------
        success_rate : Success rates over time.

        Examples
        --------
        >>> ih.plot.response_count(every="1w", facet="Channel")
        """

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
        metric: str = "Engagement",
        every: Union[str, timedelta] = "1d",
        by: Optional[str] = None,
        title: str = "Model Performance over Time",
        query: Optional[QUERY] = None,
        facet: Optional[str] = None,
        return_df: bool = False,
    ) -> Union["Figure", pl.LazyFrame]:
        """Create a line chart of model AUC over time.

        Displays model performance (Area Under the ROC Curve) calculated from
        propensity scores and actual outcomes. Higher AUC indicates better
        predictive accuracy.

        Parameters
        ----------
        metric : str, default "Engagement"
            Metric for AUC calculation: "Engagement", "Conversion", or "OpenRate".
        every : str or timedelta, default "1d"
            Time aggregation period (e.g., "1d", "1w", "1mo").
        by : str, optional
            Dimension for line coloring.
        title : str, default "Model Performance over Time"
            Chart title.
        query : QUERY, optional
            Polars expression to filter the data.
        facet : str, optional
            Dimension for faceting.
        return_df : bool, default False
            If True, return data as LazyFrame instead of figure.

        Returns
        -------
        Figure or pl.LazyFrame
            Plotly line chart (y-axis in AUC percentage), or LazyFrame if `return_df=True`.

        See Also
        --------
        success_rate : Success rate trends.

        Examples
        --------
        >>> ih.plot.model_performance_trend(by="Channel", every="1w")
        """

        plot_data = (
            self.ih.aggregates.summarize_by_interaction(
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
                    returns_scalar=True,
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
