"""Performance / volume time-series plots and proposition success rates."""

from __future__ import annotations

import logging
from datetime import timedelta

import polars as pl

from ...utils import cdh_utils
from ...utils.plot_utils import Figure
from ...utils.types import QUERY
from ._base import _PlotsBase
from ._helpers import add_metric_limit_lines, requires

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
        show_metric_limits: bool = False,
        return_df: bool = False,
    ):
        """Statistics over time

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
            By what time period to group, by default "1d", see https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.truncate.html
            for periods.
        cumulative : bool, optional
            Whether to show cumulative values or period-over-period changes, by default True
        query : Optional[QUERY], optional
            The query to apply to the data, by default None
        facet : Optional[str], optional
            Whether to facet the plot into subplots, by default None
        show_metric_limits : bool, optional
            Whether to show dashed horizontal lines at the metric limit
            thresholds (from MetricLimits.csv), by default False.
            Only applies when metric is "Performance".
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Figure | pl.LazyFrame
            Plotly line chart or LazyFrame if return_df=True

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

        >>> # Return the data instead of the figure
        >>> df = dm.plot.over_time(return_df=True)

        """
        percentage_metrics = ["Performance", "SuccessRate"]
        metric_formatting = {
            "SuccessRate": ":.4%",
            "Performance": ":.2",
            "Positives": ":.d",
            "ResponseCount": ":.d",
        }
        if isinstance(by, list):
            if len(by) == 0:
                raise ValueError("`by` must contain at least one column name")
            if len(by) == 1:
                by = pl.col(by[0])
            else:
                by_name = " / ".join(by)
                by = pl.concat_str(
                    [pl.col(c).cast(pl.Utf8).fill_null("<null>") for c in by],
                    separator=" / ",
                ).alias(by_name)
        elif not isinstance(by, pl.Expr):
            by = pl.col(by)
        by_col = by.meta.output_name()

        if self.datamart.model_data is None:
            raise ValueError("Visualisation requires model_data")

        columns_to_select: set[str] = {"SnapshotTime", metric, "ResponseCount"}
        columns_to_select.update(by.meta.root_names())
        if facet:
            columns_to_select.add(facet)

        is_percentage = metric in percentage_metrics
        metric_scaling = pl.lit(100.0 if metric == "Performance" else 1.0)

        df = (
            cdh_utils._apply_query(self.datamart.model_data, query).sort("SnapshotTime").select(list(columns_to_select))
        )

        grouping_columns = [by_col]
        if facet:
            grouping_columns.append(facet)
        df = df.with_columns(by).set_sorted("SnapshotTime")

        # Filter out null values in SnapshotTime and grouping columns to avoid issues with group_by_dynamic
        null_filters = [pl.col("SnapshotTime").is_not_null()] + [pl.col(col).is_not_null() for col in grouping_columns]
        df = df.filter(pl.all_horizontal(null_filters))

        agg_expr = [
            (
                metric_scaling * cdh_utils.weighted_average_polars(metric, "ResponseCount")
                if is_percentage
                else pl.sum(metric)
            ).alias(metric),
        ]
        if is_percentage:
            agg_expr.append(pl.sum("ResponseCount"))

        df = (
            df.with_columns(pl.col("SnapshotTime"))
            .group_by(grouping_columns + ["SnapshotTime"])
            .agg(agg_expr)
            .sort("SnapshotTime", by_col)
        )

        unique_intervals = df.select(pl.col("SnapshotTime").unique()).collect().height

        if not cumulative and unique_intervals <= 1:
            logger.warning(
                f"Only one {every} interval of data found. Cannot calculate interval-to-interval differences. "
                "Automatically switching to cumulative mode to show values instead.",
            )
            cumulative = True

        plot_metric = metric
        if not cumulative:
            plot_metric = f"{metric}_change"
            df = (
                df.group_by_dynamic(
                    "SnapshotTime",
                    every=every,
                    group_by=grouping_columns,
                )
                .agg(pl.last(metric))
                .with_columns(
                    pl.col(metric).diff().over(grouping_columns).alias(plot_metric),
                )
            )

        if return_df:
            return df
        final_df = df.collect()

        max_series = 10
        unique_series = final_df.get_column(by_col).unique().to_list()
        if len(unique_series) > max_series:
            ranking_metric = "ResponseCount" if is_percentage else metric
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

        facet_col_wrap = None
        if facet:
            unique_facet_values = final_df.select(facet).unique().height
            facet_col_wrap = max(2, int(unique_facet_values**0.5))

        title = "over all models" if facet is None else f"per {facet}"
        fig = px.line(
            final_df,
            x="SnapshotTime",
            y=plot_metric,
            color=by_col,
            hover_data={
                by_col: ":.d",
                plot_metric: metric_formatting[metric.split("_")[0]],
            },
            markers=True,
            title=f"{metric} over time, per {by_col} {title}",
            facet_col=facet,
            facet_col_wrap=facet_col_wrap,
            template="pega",
        )

        if metric == "SuccessRate":
            fig.update_yaxes(tickformat=".2%")
            fig.update_layout(yaxis={"rangemode": "tozero"})

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
        fig = (
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

        return fig

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

        # Bin performance and aggregate
        df = (
            df.with_columns(
                (pl.col("Performance") * 100)
                .cut(breaks=[p for p in range(50, 100, bin_width)], left_closed=True)
                .alias("PerformanceBinned"),
            )
            .group_by(group_cols + ["PerformanceBinned"])
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

        collected_df: pl.DataFrame = df.sort(group_cols + ["PerformanceBinned", "Proportion"]).collect()

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
        fig = cdh_utils.legend_color_order(fig)

        return fig
