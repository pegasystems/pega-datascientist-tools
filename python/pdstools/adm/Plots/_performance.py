"""Performance / volume time-series plots and proposition success rates."""

from __future__ import annotations

import logging
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import polars as pl

from ...utils import cdh_utils
from ..Aggregates import TREND_WEIGHTED_METRICS, trend_by_expression
from ._base import _PlotsBase
from ._helpers import add_full_success_rate_markers, add_metric_limit_lines, requires

if TYPE_CHECKING:
    from datetime import timedelta

    from ...utils.plot_utils import Figure
    from ...utils.types import QUERY

logger = logging.getLogger(__name__)


class _PerformancePlotsMixin(_PlotsBase):
    @requires({"SnapshotTime"})
    def over_time(
        self,
        metric: str = "Performance",
        by: pl.Expr | str | list[str] = "ModelID",
        *,
        every: str | timedelta = "1d",
        cumulative: bool = True,
        query: QUERY | None = None,
        facet: str | None = None,
        facet_col_spacing: float | None = None,
        show_metric_limits: bool = False,
        return_df: bool = False,
    ):
        """Statistics over time

        The data comes from :meth:`~pdstools.adm.Aggregates.Aggregates.metric_over_time`,
        which describes how the intervals and changes are calculated.

        Parameters
        ----------
        metric : str, optional
            The metric to plot, by default "Performance"
        by : Union[pl.Expr, str, list[str]], optional
            The column(s) to group by, by default "ModelID". When a list of
            column names is passed, the values are concatenated with " / " into
            a single combined dimension that is encoded as colour. To keep the
            chart readable, the top 10 combinations by total ``metric`` are
            kept and the rest are dropped with a warning.
        every : Union[str, timedelta], optional
            Interval as in Polars ``group_by_dynamic``, by default "1d". Set to
            "auto" to choose daily, weekly or monthly intervals from the date
            range.
        cumulative : bool, optional
            Whether to show the cumulative metric value per interval or the
            change per interval, by default True. For "SuccessRate" the change
            is the success rate within the interval. Intervals with a 100%
            success rate are marked with a red X; the y-axis is scaled to the
            other values.
        query : Optional[QUERY], optional
            The query to apply to the data, by default None
        facet : Optional[str], optional
            Whether to facet the plot into subplots, by default None
        facet_col_spacing : float, optional
            Horizontal spacing between facet columns, using Plotly Express'
            ``facet_col_spacing`` scale from 0 to 1. By default, Plotly's
            standard spacing is used.
        show_metric_limits : bool, optional
            Whether to show dashed horizontal lines and subtle background
            bands at the metric limit thresholds (from MetricLimits.csv), by
            default False. Only applies when metric is "Performance".
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Figure | pl.LazyFrame
            Plotly line chart or LazyFrame if return_df=True

        Raises
        ------
        InsufficientTrendHistoryError
            If the success rate per interval is requested but no interval has
            a valid change in responses.

        Examples
        --------
        >>> # Default: performance over time, one line per model
        >>> fig = dm.plot.over_time()

        >>> # SuccessRate over time, grouped by Channel
        >>> fig = dm.plot.over_time(metric="SuccessRate", by="Channel")

        >>> # Group by multiple dimensions at once (combined into a single
        >>> # colour-encoded series)
        >>> fig = dm.plot.over_time(by=["Channel", "Direction"])

        >>> # Period-over-period response-count changes, faceted by Direction
        >>> fig = dm.plot.over_time(
        ...     metric="ResponseCount",
        ...     cumulative=False,
        ...     facet="Direction",
        ... )

        >>> # Success rate per interval, with the interval chosen from the date range
        >>> fig = dm.plot.over_time(
        ...     metric="SuccessRate",
        ...     cumulative=False,
        ...     every="auto",
        ... )

        >>> # Return the data instead of the figure
        >>> df = dm.plot.over_time(return_df=True)

        """
        metric_formatting = {
            "SuccessRate": ":.4%",
            "Performance": ":.2",
            "Positives": ":.d",
            "ResponseCount": ":.d",
        }
        by_col = trend_by_expression(by).meta.output_name()

        if self.datamart.model_data is None:
            raise ValueError("Visualisation requires model_data")

        every = cdh_utils.resolve_trend_interval(cdh_utils._apply_query(self.datamart.model_data, query), every)
        df = self.datamart.aggregates.metric_over_time(
            metric,
            by,
            every=every,
            cumulative=cumulative,
            query=query,
            facet=facet,
        )
        if return_df:
            return df
        final_df = df.collect()

        is_success_rate_change = metric == "SuccessRate" and not cumulative
        if is_success_rate_change and final_df.height == 0:
            raise cdh_utils.InsufficientTrendHistoryError(
                "The success rate per interval requires observations in at least two distinct intervals per model",
            )
        plot_metric = f"{metric}_change" if f"{metric}_change" in final_df.columns else metric

        max_series = 10
        unique_series = final_df.get_column(by_col).unique().to_list()
        if len(unique_series) > max_series:
            ranking_metric = "ResponseCount" if metric in TREND_WEIGHTED_METRICS else metric
            top_series = (
                final_df.group_by(by_col)
                .agg(pl.sum(ranking_metric).alias("__total"))
                .sort("__total", descending=True, nulls_last=True)
                .head(max_series)
                .get_column(by_col)
                .to_list()
            )
            dropped = len(unique_series) - len(top_series)
            logger.warning(
                "over_time: %d unique values for `%s` exceeds the cap of %d; "
                "keeping the top %d by total %s and dropping %d.",
                len(unique_series),
                by_col,
                max_series,
                max_series,
                ranking_metric,
                dropped,
            )
            final_df = final_df.filter(pl.col(by_col).is_in(top_series))

        import plotly.express as px

        facet_values = final_df.get_column(facet).unique(maintain_order=True).to_list() if facet else []
        facet_col_wrap = max(2, int(len(facet_values) ** 0.5)) if facet else None

        try:
            interval_label = cdh_utils.trend_interval_label(every) if isinstance(every, str) else None
        except ValueError:
            interval_label = None
        success_rate_labels = {
            "SuccessRate": f"{interval_label} Success Rate" if interval_label else f"Success Rate per {every}",
            "TotalResponses": "Responses (cumulative)",
            "PositivesInInterval": "Positives in interval",
            "ResponsesInInterval": "Responses in interval",
            "PeriodsCovered": "Snapshot gap (intervals)",
        }
        success_rate_hover = {
            "TotalResponses": ":,.0f",
            "PositivesInInterval": ":,.0f",
            "ResponsesInInterval": ":,.0f",
            "PeriodsCovered": ":.0f",
        }
        title = "over all models" if facet is None else f"per {facet}"
        line_kwargs: dict[str, Any] = {
            "x": "SnapshotTime",
            "y": plot_metric,
            "color": by_col,
            "hover_data": {
                by_col: ":.d",
                plot_metric: metric_formatting[metric.split("_")[0]],
            },
            "markers": True,
            "title": f"{metric} over time, per {by_col} {title}",
            "facet_col": facet,
            "template": "pega",
        }
        if is_success_rate_change:
            line_kwargs["hover_data"].update({"ResponseCount": False, "Positives": False, **success_rate_hover})
            line_kwargs["labels"] = success_rate_labels
        elif plot_metric == "Performance":
            line_kwargs["labels"] = {"Performance": "Performance (AUC)"}
        if facet_col_wrap is not None:
            line_kwargs["facet_col_wrap"] = facet_col_wrap
        if facet_col_spacing is not None:
            line_kwargs["facet_col_spacing"] = facet_col_spacing
        if facet_values:
            line_kwargs["category_orders"] = {facet: facet_values}

        fig = px.line(final_df, **line_kwargs)
        fig.update_xaxes(hoverformat="%b %d, %Y")

        if plot_metric == "Performance":
            fig.update_yaxes(range=[50, 100])
        elif is_success_rate_change:
            # Scale to the success rates below 100%; mark the 100% intervals instead.
            full_success_rate = final_df.filter(pl.col(metric) >= 1.0)
            partial_max = final_df.filter(pl.col(metric) < 1.0).select(pl.col(metric).max()).item()
            y_max = 1.0 if partial_max is None else (min(1.0, float(partial_max) * 1.05) if partial_max > 0 else 0.01)
            if full_success_rate.height:
                fig = add_full_success_rate_markers(
                    fig,
                    full_success_rate,
                    y=y_max * 0.96,
                    hover_data={by_col: True, "SuccessRate": ":.2%", **success_rate_hover},
                    labels={**success_rate_labels, "SnapshotTime": "Interval", "SuccessRate": "Success Rate"},
                    facet=facet,
                    facet_values=facet_values,
                    facet_col_wrap=facet_col_wrap,
                    facet_col_spacing=facet_col_spacing,
                )
            fig.update_yaxes(range=[0, y_max], tickformat=".2%")
        elif metric == "SuccessRate":
            fig.update_yaxes(tickformat=".2%", rangemode="tozero")

        if show_metric_limits and metric == "Performance":
            fig = add_metric_limit_lines(fig, orientation="horizontal")

        return fig

    @requires({"ModelID", "Name"})
    def proposition_success_rates(
        self,
        metric: str = "SuccessRate",
        by: str = "Name",
        *,
        top_n: int = 0,
        query: QUERY | None = None,
        facet: str | None = None,
        return_df: bool = False,
    ):
        """Proposition Success Rates

        Parameters
        ----------
        metric : str, optional
            The metric to plot, by default "SuccessRate"
        by : str, optional
            By which column to group the, by default "Name"
        top_n : int, optional
            Whether to take a top_n on the `by` column, by default 0
        query : Optional[QUERY], optional
            A query to apply to the data, by default None
        facet : Optional[str], optional
            What facetting column to apply to the graph, by default None
        return_df : bool, optional
            Whether to return a DataFrame instead of the graph, by default False

        Returns
        -------
        Figure | pl.LazyFrame
            Plotly histogram figure or LazyFrame if return_df=True

        Examples
        --------
        >>> # Default: average success rate per proposition
        >>> fig = dm.plot.proposition_success_rates()

        >>> # Top-10 propositions by success rate, faceted by Channel
        >>> fig = dm.plot.proposition_success_rates(top_n=10, facet="Channel")

        >>> # Use ResponseCount as the metric grouped by Configuration
        >>> fig = dm.plot.proposition_success_rates(
        ...     metric="ResponseCount",
        ...     by="Configuration",
        ... )

        """
        if self.datamart.model_data is None:
            raise ValueError("Visualisation requires model_data")

        df = cdh_utils._apply_query(self.datamart.model_data, query).select(
            {"ModelID", "Name", metric, by, facet},
        )

        if return_df:
            return df

        import plotly.express as px

        title = "over all models" if facet is None else f"per {facet}"
        fig = px.histogram(
            df.collect(),
            x=metric,
            y=by,
            color=by,
            facet_col=facet,
            facet_col_wrap=5,
            histfunc="avg",
            title=f"{metric} of each proposition {title}",
            template="pega",
        )
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_layout(showlegend=False)
        fig.update_yaxes(dtick=1, automargin=True)
        errors = df.group_by(by).agg(pl.std("SuccessRate").fill_null(0)).collect()
        for i, bar in enumerate(fig.data):
            fig.data[i]["error_x"] = {
                "array": [errors.filter(Name=bar["name"])["SuccessRate"].item()],
                "valueminus": 0,
            }
        return fig

    def gains_chart(
        self,
        value: str,
        *,
        index: str | None = None,
        by: str | list[str] | None = None,
        query: QUERY | None = None,
        title: str | None = None,
        return_df: bool = False,
    ) -> Figure | pl.LazyFrame:
        """Generate a gains chart showing cumulative distribution of a metric.

        Creates a gains/lift chart to visualize model response skewness. Shows what
        percentage of the total value (e.g., responses, positives) is driven by what
        percentage of models. Useful for identifying if a small number of models
        drive most of the volume.

        Parameters
        ----------
        value : str
            Column name containing the metric to compute gains for (e.g., "ResponseCount", "Positives")
        index : str, optional
            Column name to normalize by (e.g., population size). If None, uses model count.
        by : str | list[str], optional
            Column(s) to group by for separate gain curves (e.g., "Channel" or ["Channel", "Direction"])
        query : QUERY, optional
            Optional query to filter the data before computing gains
        title : str, optional
            Chart title. If None, uses "Gains Chart"
        return_df : bool, default False
            If True, return the gains data instead of the figure

        Returns
        -------
        Figure | pl.LazyFrame
            Plotly figure showing the gains chart, or LazyFrame if return_df=True

        Examples
        --------
        >>> # Single gains curve for response count
        >>> fig = datamart.plot.gains_chart(value="ResponseCount")

        >>> # Gains curves by channel for positives
        >>> fig = datamart.plot.gains_chart(
        ...     value="Positives",
        ...     by=["Channel", "Direction"],
        ...     title="Cumulative Positives by Channel"
        ... )
        """
        from ...utils import cdh_utils, report_utils

        # Get the last snapshot of data
        df = self.datamart.aggregates.last()

        # Apply query if provided
        if query is not None:
            df = cdh_utils._apply_query(df, query)

        # Calculate gains
        gains_data = report_utils.gains_table(df, value=value, index=index, by=by)

        if return_df:
            return gains_data.lazy()

        import plotly.express as px

        # Create the plot
        if by is None:
            fig = px.area(
                gains_data,
                x="cum_x",
                y="cum_y",
                title=title or "Gains Chart",
                template="pega",
            )
        else:
            by_as_list = by if isinstance(by, list) else [by]
            # Create a combined label for the legend
            legend_col = gains_data.select(pl.concat_str(by_as_list, separator="/").alias("By"))["By"]

            fig = px.line(
                gains_data,
                x="cum_x",
                y="cum_y",
                color=legend_col,
                title=title or "Gains Chart",
                template="pega",
            )
            fig = fig.update_layout(legend_title="/".join(by_as_list))

        # Add diagonal reference line (represents perfect equality)
        fig.add_shape(
            type="line",
            line=dict(color="grey", dash="dash"),
            x0=0,
            x1=1,
            y0=0,
            y1=1,
        )

        # Configure axes and layout
        return (
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            .update_layout(
                autosize=False,
                width=400,
                height=400,
            )
            .update_yaxes(constrain="domain", title="% of Responders")
            .update_xaxes(tickformat=",.0%", constrain="domain", title="% of Population")
            .update_yaxes(tickformat=",.0%")
        )

    def performance_volume_distribution(
        self,
        *,
        by: str | list[str] | None = None,
        query: QUERY | None = None,
        bin_width: int = 3,
        title: str | None = None,
        return_df: bool = False,
    ) -> Figure | pl.LazyFrame:
        """Generate a performance vs volume distribution chart.

        Shows how response volume is distributed across different model performance
        ranges. Helps identify if volume is driven by high-performing or low-performing
        models. Ideally, most volume should be in the 60-80 AUC range.

        Parameters
        ----------
        by : str | list[str], optional
            Column(s) to group by for separate curves (e.g., "Channel" or ["Channel", "Direction"])
            If None, creates a single curve for all data
        query : QUERY, optional
            Optional query to filter the data before analysis
        bin_width : int, default 3
            Width of performance bins in AUC points (default creates bins of 3: 50-53, 53-56, etc.)
        title : str, optional
            Chart title. If None, uses "Performance vs Volume"
        return_df : bool, default False
            If True, return the binned data instead of the figure

        Returns
        -------
        Figure | pl.LazyFrame
            Plotly figure showing performance distribution, or LazyFrame if return_df=True

        Notes
        -----
        Performance is binned from 50-100 using the specified bin_width. The chart shows
        what percentage of responses fall into each performance bin, grouped by the `by`
        parameter if provided.

        Examples
        --------
        >>> # Single curve for all channels
        >>> fig = datamart.plot.performance_volume_distribution()

        >>> # Separate curves per channel
        >>> fig = datamart.plot.performance_volume_distribution(
        ...     by=["Channel", "Direction"],
        ...     title="Performance Distribution by Channel"
        ... )
        """
        from ...utils import cdh_utils

        # Get model data (raises ValueError if not loaded)
        df = self.datamart._require_model_data()

        # Apply query if provided
        if query is not None:
            df = cdh_utils._apply_query(df, query)

        # Determine grouping columns
        group_cols: list[str] = []
        if by is not None:
            by_list = by if isinstance(by, list) else [by]
            group_cols = by_list.copy()

        # Create combined grouping column if needed
        if by is not None:
            by_list = by if isinstance(by, list) else [by]
            df = df.with_columns(pl.concat_str(by_list, separator="/").alias("_GroupBy"))
            group_cols = ["_GroupBy"]

        # Polars 2 replaces cut() with bin_intervals().
        performance = pl.col("Performance") * 100
        breaks = list(range(50, 100, bin_width))
        bin_intervals = getattr(performance, "bin_intervals", None)
        if bin_intervals is not None:  # pragma: no cover - Polars 2 only; covered by the Polars 2 CI job
            labels = [
                f"(-inf, {breaks[0]})",
                *(f"[{left}, {right})" for left, right in pairwise(breaks)),
                f"[{breaks[-1]}, inf)",
            ]
            performance_binned = bin_intervals(breaks, labels=labels)
        else:
            performance_binned = performance.cut(breaks=breaks, left_closed=True)

        # Bin performance and aggregate
        df = (
            df.with_columns(performance_binned.alias("PerformanceBinned"))
            .group_by([*group_cols, "PerformanceBinned"])
            .agg(
                pl.sum("ResponseCount"),
                (pl.min("Performance") * 100).round(2).alias("break_label"),
            )
        )

        # Calculate proportions within each group
        if group_cols:
            df = df.with_columns(
                (pl.col("ResponseCount") / pl.col("ResponseCount").sum().over(group_cols[0])).alias("Proportion")
            )
        else:
            df = df.with_columns((pl.col("ResponseCount") / pl.col("ResponseCount").sum()).alias("Proportion"))

        collected_df: pl.DataFrame = df.sort([*group_cols, "PerformanceBinned", "Proportion"]).collect()

        if return_df:
            return collected_df.lazy()

        import plotly.graph_objects as go

        # Create the plot
        fig = go.Figure()

        if group_cols:
            # Multiple curves
            groups = collected_df[group_cols[0]].unique().sort()
            for group in groups:
                group_df = collected_df.filter(pl.col(group_cols[0]) == group)
                fig.add_trace(
                    go.Scatter(
                        x=group_df["PerformanceBinned"],
                        y=group_df["Proportion"],
                        line_shape="spline",
                        name=group,
                    )
                )
        else:
            # Single curve
            fig.add_trace(
                go.Scatter(
                    x=collected_df["PerformanceBinned"],
                    y=collected_df["Proportion"],
                    line_shape="spline",
                    name="All Channels",
                )
            )

        # Configure layout
        fig = (
            fig.update_yaxes(tickformat=",.0%")
            .update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=collected_df["PerformanceBinned"].unique().sort().to_list(),
            )
            .update_layout(
                template="pega",
                title=title or "Performance vs Volume",
                xaxis_title="Model Performance",
                yaxis_title="Percentage of Responses",
            )
        )

        # Apply legend color ordering
        return cdh_utils.legend_color_order(fig)

    @requires({"ModelID", "Positives", "ResponseCount"})
    def success_rate_intervals(
        self,
        *,
        confidence_level: float = 0.95,
        query: QUERY | None = None,
        return_df: bool = False,
    ):
        """Positives against negatives per model, with success rate confidence intervals.

        Each model is a dot at (1 + negatives, 1 + positives) on log axes,
        so lines through the origin are constant success rates. The dashed
        line is the average success rate of the models in the panel. The error
        bars are the Wilson confidence interval of the success rate. There is
        one panel per channel and direction, each with its own average, as
        success rates differ widely between channels. Models are coloured by whether that interval is entirely below
        (``Very low`` / ``Below average``), entirely above, or consistent
        with the average; see
        :meth:`pdstools.adm.Aggregates.Aggregates.success_rate_intervals`.

        Parameters
        ----------
        confidence_level : float, default 0.95
            Confidence level of the interval.
        query : QUERY, optional
            Restricts the models in scope.
        return_df : bool, default False
            Return the plotted data instead of the figure.

        Returns
        -------
        Figure | pl.DataFrame | None
            Plotly figure, the data if ``return_df``, or None if no model
            has responses.

        Examples
        --------
        >>> fig = dm.plot.success_rate_intervals()
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        data = self.datamart.aggregates.success_rate_intervals(confidence_level=confidence_level, query=query).collect()
        if data.height == 0:
            return None
        if return_df:
            return data

        colors = {
            "Very low": "#d62728",
            "Below average": "#ff9f43",
            "Consistent with average": "#7f8c8d",
            "Above average": "#2ca02c",
        }
        label_columns = [
            c
            for c in ("Configuration", "Channel", "Direction", "Issue", "Group", "Name", "Treatment")
            if c in data.columns
        ]
        group_cols = [c for c in ("Channel", "Direction") if c in data.columns]
        panels = data.select(group_cols).unique(maintain_order=True).sort(group_cols).rows() if group_cols else [()]
        n_cols = min(3, len(panels))
        n_rows = -(-len(panels) // n_cols)
        titles = ["/".join(str(v) for v in key) for key in panels]
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=titles if group_cols else None,
            horizontal_spacing=0.08,
            vertical_spacing=0.12 if n_rows > 1 else 0.0,
        )
        legend_shown: set[str] = set()
        for i, key in enumerate(panels):
            row, col = divmod(i, n_cols)
            panel = data.filter(*[pl.col(c) == v for c, v in zip(group_cols, key, strict=True)])
            for category, color in colors.items():
                part = panel.filter(pl.col("SuccessRateCategory") == category)
                if part.height == 0:
                    continue
                responses = part["ResponseCount"]
                hover = [
                    "<br>".join(f"{c}: {r[c]}" for c in label_columns)
                    + f"<br>Positives: {r['Positives']:,.0f}<br>Negatives: {r['Negatives']:,.0f}"
                    + f"<br>Success rate: {r['SuccessRate']:.2%} "
                    + f"({r['SuccessRate_CI_Lower']:.2%} - {r['SuccessRate_CI_Upper']:.2%})"
                    for r in part.iter_rows(named=True)
                ]
                fig.add_trace(
                    go.Scatter(
                        x=(part["Negatives"] + 1).to_list(),
                        y=(part["Positives"] + 1).to_list(),
                        mode="markers",
                        name=category,
                        legendgroup=category,
                        showlegend=category not in legend_shown,
                        marker=dict(color=color, size=6, opacity=0.8),
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=((part["SuccessRate_CI_Upper"] - part["SuccessRate"]) * responses).to_list(),
                            arrayminus=((part["SuccessRate"] - part["SuccessRate_CI_Lower"]) * responses).to_list(),
                            color=color,
                            thickness=1,
                        ),
                        hovertext=hover,
                        hoverinfo="text",
                    ),
                    row=row + 1,
                    col=col + 1,
                )
                legend_shown.add(category)

            average = panel["AverageSuccessRate"][0]
            x_max = max(float(panel.select(pl.col("Negatives").max().cast(pl.Float64)).item()) + 1, 10.0)
            fig.add_trace(
                go.Scatter(
                    x=[1, x_max],
                    y=[1, 1 + (x_max - 1) * average / (1 - average) if average < 1 else x_max],
                    mode="lines",
                    name="Average success rate",
                    legendgroup="average",
                    showlegend=i == 0,
                    line=dict(color="black", dash="dash", width=1),
                    hovertext=f"Average success rate: {average:.2%}",
                    hoverinfo="text",
                ),
                row=row + 1,
                col=col + 1,
            )
        fig.update_xaxes(type="log", title_text="Negatives + 1")
        fig.update_yaxes(type="log", title_text="Positives + 1")
        fig.update_layout(
            template="pega",
            title=f"Success rate per model ({confidence_level:.0%} confidence interval)",
            height=max(450, 380 * n_rows),
        )
        return fig
