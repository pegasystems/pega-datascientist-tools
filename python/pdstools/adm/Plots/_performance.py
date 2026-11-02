"""Performance / volume time-series plots and proposition success rates."""

from __future__ import annotations

import logging
from datetime import timedelta
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import polars as pl

from ...utils import cdh_utils
from ._base import _PlotsBase
from ._helpers import add_metric_limit_lines, requires

if TYPE_CHECKING:
    from ...utils.plot_utils import Figure
    from ...utils.types import QUERY

logger = logging.getLogger(__name__)

_AUTO_DAILY_MAX_SPAN = timedelta(days=28)
_AUTO_MONTHLY_MIN_SPAN = timedelta(days=365 * 5)


def _resolve_trend_interval(
    df: pl.LazyFrame,
    every: str | timedelta,
) -> str | timedelta:
    if every != "auto":
        return every

    start, end = (
        df.select(
            pl.col("SnapshotTime").min().alias("__start"),
            pl.col("SnapshotTime").max().alias("__end"),
        )
        .collect()
        .row(0)
    )
    if start is None or end is None:
        raise ValueError("Cannot infer a trend interval without valid SnapshotTime values")

    span = end - start
    if span <= _AUTO_DAILY_MAX_SPAN:
        return "1d"
    if span > _AUTO_MONTHLY_MIN_SPAN:
        return "1mo"
    return "1w"


def _period_success_rate(
    df: pl.LazyFrame,
    *,
    every: str | timedelta,
    grouping_columns: list[str],
) -> pl.LazyFrame:
    model_grouping_columns = list(dict.fromkeys(["ModelID", *grouping_columns]))
    period_data = (
        df.group_by_dynamic(
            "SnapshotTime",
            every=every,
            group_by=model_grouping_columns,
        )
        .agg(
            pl.col("Positives").last(),
            pl.col("ResponseCount").last(),
        )
        .sort(*model_grouping_columns, "SnapshotTime")
    )
    period_index = (
        period_data.select(["SnapshotTime", *model_grouping_columns])
        .collect()
        .upsample("SnapshotTime", every=every, group_by=model_grouping_columns)
        .sort(*model_grouping_columns, "SnapshotTime")
        .with_columns(
            pl.int_range(pl.len()).over(model_grouping_columns).alias("__period_index"),
        )
        .select(["SnapshotTime", *model_grouping_columns, "__period_index"])
    )
    periods_elapsed = pl.col("__period_index").diff().over(model_grouping_columns).clip(lower_bound=1)
    total_responses = period_data.group_by(["SnapshotTime", *grouping_columns]).agg(
        pl.sum("ResponseCount").alias("TotalResponses"),
    )

    return (
        period_data.join(
            period_index.lazy(),
            on=["SnapshotTime", *model_grouping_columns],
        )
        .sort(*model_grouping_columns, "SnapshotTime")
        .with_columns(
            pl.col("Positives").diff().over(model_grouping_columns).alias("__positive_change"),
            pl.col("ResponseCount").diff().over(model_grouping_columns).alias("__response_change"),
            periods_elapsed.alias("__periods_elapsed"),
        )
        .with_columns(
            (pl.col("__positive_change") / pl.col("__periods_elapsed")).alias("__positives_per_period"),
            (pl.col("__response_change") / pl.col("__periods_elapsed")).alias("__responses_per_period"),
        )
        .filter(
            (pl.col("__positives_per_period") >= 0) & (pl.col("__responses_per_period") > 0),
        )
        .group_by(["SnapshotTime", *grouping_columns])
        .agg(
            pl.sum("__positives_per_period").alias("Positives"),
            pl.sum("__responses_per_period").alias("ResponseCount"),
            pl.sum("__positive_change").alias("PositivesInInterval"),
            pl.sum("__response_change").alias("ResponsesInInterval"),
            pl.max("__periods_elapsed").alias("PeriodsCovered"),
        )
        .join(total_responses, on=["SnapshotTime", *grouping_columns])
        .with_columns((pl.col("Positives") / pl.col("ResponseCount")).alias("SuccessRate"))
        .sort("SnapshotTime", *grouping_columns)
    )


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
            By what time period to group, by default "1d". Set to "auto" to use
            daily buckets for ranges up to 28 days, weekly buckets up to five
            years, and monthly buckets for longer ranges.
        cumulative : bool, optional
            Whether to show the cumulative metric value at each observed time
            bucket or the period-over-period change, by default True. When
            False for "SuccessRate", the method calculates period CTR from the
            changes in cumulative Positives and ResponseCount per model. For
            other metrics, it differences consecutive bucket values. Changes
            are divided by the number of elapsed buckets when observations are
            separated by a gap; missing buckets are not plotted.
            Exact 100% period SuccessRate values remain in the line data, but
            the y-axis range is based on non-100% values. Red X markers identify
            the clipped spikes.
            Cumulative "Performance" (AUC) uses a fixed 50-100 y-axis; other
            trends include zero in the y-axis range.
        query : Optional[QUERY], optional
            The query to apply to the data, by default None
        facet : Optional[str], optional
            Whether to facet the plot into subplots, by default None
        facet_col_spacing : float, optional
            Horizontal spacing between facet columns, using Plotly Express'
            ``facet_col_spacing`` scale from 0 to 1. By default, Plotly's
            standard spacing is used.
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

        >>> # Period CTR, with interval chosen from the available date range
        >>> fig = dm.plot.over_time(
        ...     metric="SuccessRate",
        ...     cumulative=False,
        ...     every="auto",
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
        is_period_success_rate = metric == "SuccessRate" and not cumulative
        if is_period_success_rate:
            missing_columns = {"ModelID", "Positives", "ResponseCount"} - set(
                self.datamart.model_data.collect_schema().names(),
            )
            if missing_columns:
                raise ValueError(
                    f"Period SuccessRate requires model-level cumulative columns: {', '.join(sorted(missing_columns))}",
                )
            columns_to_select.update({"ModelID", "Positives"})

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

        every = _resolve_trend_interval(df, every)
        if is_period_success_rate:
            df = _period_success_rate(
                df,
                every=every,
                grouping_columns=grouping_columns,
            )
            if df.select(pl.col("SnapshotTime").n_unique()).collect().item() == 0:
                raise ValueError("Period SuccessRate requires observations in at least two distinct periods per model")
            plot_metric = metric
        else:
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
                .group_by([*grouping_columns, "SnapshotTime"])
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
                    .sort(*grouping_columns, "SnapshotTime")
                )
                period_index = (
                    df.collect()
                    .upsample("SnapshotTime", every=every, group_by=grouping_columns)
                    .sort(*grouping_columns, "SnapshotTime")
                    .with_columns(
                        pl.int_range(pl.len()).over(grouping_columns).alias("__period_index"),
                    )
                    .select([*grouping_columns, "SnapshotTime", "__period_index"])
                )
                periods_elapsed = pl.col("__period_index").diff().over(grouping_columns).clip(lower_bound=1)
                df = (
                    df.join(period_index.lazy(), on=[*grouping_columns, "SnapshotTime"])
                    .sort(*grouping_columns, "SnapshotTime")
                    .with_columns(
                        (pl.col(metric).diff().over(grouping_columns) / periods_elapsed).alias(plot_metric),
                    )
                    .drop("__period_index")
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

        facet_values = final_df.get_column(facet).unique(maintain_order=True).to_list() if facet else []
        outlier_df: pl.DataFrame | None = None
        valid_line_df = final_df
        period_y_max = 1.0
        spike_marker_y = 0.96
        if is_period_success_rate:
            outlier_df = final_df.filter(pl.col(plot_metric) == 1.0)
            valid_line_df = final_df.filter(pl.col(plot_metric) != 1.0)
            valid_max = valid_line_df.get_column(plot_metric).max() if valid_line_df.height else None
            if valid_max is not None:
                period_y_max = min(1.0, valid_max * 1.05) if valid_max > 0 else 0.01
            spike_marker_y = period_y_max * 0.96

        import plotly.express as px

        facet_col_wrap = None
        if facet:
            facet_col_wrap = max(2, int(len(facet_values) ** 0.5))

        period_name = {
            "1d": "Daily",
            "1w": "Weekly",
            "1mo": "Monthly",
            timedelta(days=1): "Daily",
            timedelta(days=7): "Weekly",
        }.get(every)
        period_ctr_label = f"{period_name} CTR" if period_name else f"CTR per {every}"
        period_hover_labels = {
            "SuccessRate": period_ctr_label,
            "TotalResponses": "Total Responses (cumulative)",
            "PositivesInInterval": "Positives in interval",
            "ResponsesInInterval": "Responses in interval",
            "PeriodsCovered": "Max periods between snapshots",
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
        if is_period_success_rate:
            line_kwargs["hover_data"].update(
                {
                    "ResponseCount": False,
                    "Positives": False,
                    "TotalResponses": ":,.0f",
                    "PositivesInInterval": ":,.0f",
                    "ResponsesInInterval": ":,.0f",
                    "PeriodsCovered": ":.0f",
                },
            )
            line_kwargs["labels"] = period_hover_labels
        if facet_col_wrap is not None:
            line_kwargs["facet_col_wrap"] = facet_col_wrap
        if facet_col_spacing is not None:
            line_kwargs["facet_col_spacing"] = facet_col_spacing
        if facet_values:
            line_kwargs["category_orders"] = {facet: facet_values}

        fig = px.line(final_df, **line_kwargs)
        fig.update_xaxes(hoverformat="%b %d, %Y")

        if outlier_df is not None and outlier_df.height:
            spike_marker_data = outlier_df.with_columns(
                pl.lit(spike_marker_y).alias("__spike_marker_y"),
            )
            spike_marker_kwargs: dict[str, Any] = {
                "x": "SnapshotTime",
                "y": "__spike_marker_y",
                "hover_data": {
                    by_col: True,
                    "SuccessRate": ":.2%",
                    "TotalResponses": ":,.0f",
                    "PositivesInInterval": ":,.0f",
                    "ResponsesInInterval": ":,.0f",
                    "PeriodsCovered": ":.0f",
                    "__spike_marker_y": False,
                },
                "labels": {
                    **period_hover_labels,
                    "SnapshotTime": "100% period",
                    "SuccessRate": "CTR in flagged period",
                },
                "template": "pega",
                "color_discrete_sequence": ["#d62728"],
            }
            if facet:
                spike_marker_kwargs["facet_col"] = facet
                spike_marker_kwargs["facet_col_wrap"] = facet_col_wrap
                spike_marker_kwargs["category_orders"] = {facet: facet_values}
            if facet_col_spacing is not None:
                spike_marker_kwargs["facet_col_spacing"] = facet_col_spacing
            spike_marker_fig = px.scatter(spike_marker_data, **spike_marker_kwargs)
            for index, trace in enumerate(spike_marker_fig.data):
                trace.name = "CTR 100%"
                trace.showlegend = index == 0
                trace.marker = {"color": "#d62728", "size": 9, "symbol": "x"}
                trace.mode = "markers"
            fig.add_traces(spike_marker_fig.data)

        if metric == "Performance" and cumulative:
            fig.update_yaxes(range=[50, 100])
        elif metric == "SuccessRate" and is_period_success_rate:
            fig.update_yaxes(range=[0, period_y_max], tickformat=".2%")
        elif metric == "SuccessRate":
            fig.update_yaxes(tickformat=".2%", rangemode="tozero")
        else:
            fig.update_yaxes(rangemode="tozero")

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
