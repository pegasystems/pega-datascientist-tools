"""Predictor-level performance, contribution, heatmap, and count plots."""

from __future__ import annotations

import polars as pl
import polars.selectors as cs

from ...utils import cdh_utils
from ...utils.plot_utils import get_colorscale
from ...utils.types import QUERY
from ._base import _PlotsBase
from ._helpers import requires


class _PredictorPlotsMixin(_PlotsBase):
    def _boxplot_pre_aggregated(
        self,
        df: pl.LazyFrame,
        *,
        y_col: str,
        metric_col: str,
        metric_weight_col: str | None = None,
        legend_col: str | None = None,
        color_discrete_map: dict[str, str] | None = None,
        return_df: bool = False,
    ):
        if legend_col is None:
            legend_col = y_col

        if metric_weight_col is None:
            mean_expr = pl.mean(metric_col)
        else:
            mean_expr = cdh_utils.weighted_average_polars(metric_col, metric_weight_col)
        pre_aggs = (
            df.group_by(list({y_col, legend_col}))
            .agg(
                median=pl.median(metric_col),
                mean=mean_expr,
                q1=pl.quantile(metric_col, 0.25),
                q3=pl.quantile(metric_col, 0.75),
                min=pl.min(metric_col),
                max=pl.max(metric_col),
                whisker_low=pl.quantile(metric_col, 0.1),
                whisker_high=pl.quantile(metric_col, 0.9),
            )
            .with_columns(iqr=pl.col("q3") - pl.col("q1"))
            # limiting the decimals should reduce the size of
            # the plotly data in HTML significantly. TODO not sure thats true!!
            .with_columns(cs.numeric().round(3))
            .sort(["median", y_col], descending=True)
            .collect()
        )

        if return_df:
            return pre_aggs

        if pre_aggs.select(pl.first().len()).item() == 0:
            return None

        import plotly.graph_objects as go

        from ...utils.pega_template import colorway

        fig = go.Figure()

        # Build a color map from the provided global map, falling back to the
        # pega colorway for any category not covered.
        present_categories = sorted(pre_aggs.select(pl.col(legend_col).unique())[legend_col].to_list())
        base_map: dict[str, str] = color_discrete_map or {}
        fallback_index = 0
        color_map: dict[str, str] = {}
        for cat in present_categories:
            if cat in base_map:
                color_map[cat] = base_map[cat]
            else:
                color_map[cat] = colorway[fallback_index % len(colorway)]
                fallback_index += 1

        # Track which categories have been added to legend
        legend_added = set()

        # Create a box plot for each predictor
        for i, row in enumerate(pre_aggs.iter_rows(named=True)):
            show_in_legend = row[legend_col] not in legend_added
            if show_in_legend:
                legend_added.add(row[legend_col])

            fig.add_trace(
                go.Box(
                    q1=[row["q1"]],
                    median=[row["median"]],
                    q3=[row["q3"]],
                    lowerfence=[row["whisker_low"]],
                    upperfence=[row["whisker_high"]],
                    mean=[row["mean"]],
                    y=[row[y_col]],
                    boxpoints=False,
                    marker=dict(color=color_map[row[legend_col]]),
                    name=row[legend_col],
                    legendgroup=row[legend_col],
                    orientation="h",
                    showlegend=show_in_legend,
                ),
            )

        fig.update_layout(
            title=f"{metric_col} by {legend_col}",
            xaxis_title=metric_col,
            yaxis_title="",
            template="pega",
        )

        # set y-axis category order to show highest median values at the top
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=pre_aggs[y_col].to_list(),
            automargin=True,
            autorange="reversed",
        )
        return fig

    @requires(
        combined_columns={
            "ModelID",
            "PredictorName",
            "PredictorCategory",
            "ResponseCountBin",
            "Type",
            "EntryType",
        },
    )
    def predictor_performance(
        self,
        *,
        metric: str = "Performance",
        top_n: int | None = None,
        active_only: bool = False,
        query: QUERY | None = None,
        return_df: bool = False,
    ):
        """Plots a box plot of the performance of the predictors

        Use the query argument to drill down to a more specific subset
        If top n is given, chooses the top predictors based on the
        weighted average performance across models, ordered by their median performance.

        Parameters
        ----------
        metric : str, optional
            The metric to plot, by default "Performance"
            This is more for future-proofing, once FeatureImportance gets more used.
        top_n : Optional[int], optional
            The top n predictors to plot, by default None
        active_only : bool, optional
            Whether to only consider active predictor performance, by default False
        query : Optional[QUERY], optional
            The query to apply to the data, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Figure | pl.LazyFrame | None
            Plotly box plot figure, LazyFrame if return_df=True, or None if no data

        See Also
        --------
        pdstools.adm.ADMDatamart.apply_predictor_categorization : how to override the out of the box predictor categorization

        Examples
        --------
        >>> # Default: all predictors ranked by performance
        >>> fig = dm.plot.predictor_performance()

        >>> # Top-15 active predictors only
        >>> fig = dm.plot.predictor_performance(top_n=15, active_only=True)

        >>> # Filter to a specific channel and return the raw data
        >>> df = dm.plot.predictor_performance(
        ...     query={"Channel": "Web"},
        ...     return_df=True,
        ... )

        """
        # in combined_data, the performance metric is renamed to PredictorPerformance
        metric = "PredictorPerformance" if metric == "Performance" else metric
        df = cdh_utils._apply_query(
            self.datamart.aggregates.last(table="combined_data")
            .with_columns(
                pl.col("PredictorPerformance") * 100.0,
            )
            .filter(pl.col("EntryType") != "Classifier")
            .filter(pl.col("ResponseCountBin") > 0)
            .unique(subset=["ModelID", "PredictorName"], keep="first"),
            query=query,
            allow_empty=True,
        )
        if active_only:
            df = df.filter(pl.col("EntryType") == "Active")
        if top_n:
            df = self.datamart.aggregates._top_n(
                df,
                top_n,
                metric,
            )

        fig = self._boxplot_pre_aggregated(
            df,
            y_col="PredictorName",
            metric_col=metric,
            # It is confusing, but ResponseCountBin is the correct total response
            # count for a predictor. It is the sum of BinResponseCount. Thanks SK :)
            metric_weight_col="ResponseCountBin",
            legend_col="PredictorCategory",
            color_discrete_map=self.datamart.predictor_category_color_map,
            return_df=return_df,
        )

        if fig is not None and not return_df:
            fig.update_xaxes(title="Performance")
            fig.update_layout(title=f"{metric} by Predictor Category")

        return fig

    @requires(
        combined_columns={
            "ModelID",
            "PredictorName",
            "PredictorCategory",
            "ResponseCountBin",
            "Type",
            "EntryType",
        },
    )
    def predictor_category_performance(
        self,
        *,
        metric: str = "Performance",
        active_only: bool = False,
        query: QUERY | None = None,
        return_df: bool = False,
    ):
        """Plot the predictor category performance

        Parameters
        ----------
        metric : str, optional
            The metric to plot, by default "Performance"
        active_only : bool, optional
            Whether to only analyze active predictors, by default False
        query : Optional[QUERY], optional
            An optional query to apply, by default None
        return_df : bool, optional
            An optional flag to get the dataframe instead, by default False

        Returns
        -------
        px.Figure
            A Plotly figure

        See Also
        --------
        pdstools.adm.ADMDatamart.apply_predictor_categorization : how to override the out of the box predictor categorization

        Examples
        --------
        >>> # Default: performance box plot per predictor category
        >>> fig = dm.plot.predictor_category_performance()

        >>> # Active predictors only, filtered to a specific channel
        >>> fig = dm.plot.predictor_category_performance(
        ...     active_only=True,
        ...     query={"Channel": "Web"},
        ... )

        >>> # Return underlying data for further analysis
        >>> df = dm.plot.predictor_category_performance(return_df=True)

        """
        metric = "PredictorPerformance" if metric == "Performance" else metric

        df = cdh_utils._apply_query(
            self.datamart.aggregates.last(table="combined_data")
            .with_columns(
                pl.col("PredictorPerformance") * 100.0,
            )
            .filter(pl.col("EntryType") != "Classifier")
            .filter(pl.col("ResponseCountBin") > 0),
            query=query,
        )
        if active_only:
            df = df.filter(pl.col("EntryType") == "Active")

        fig = self._boxplot_pre_aggregated(
            df,
            y_col="PredictorCategory",
            metric_col="PredictorPerformance",
            legend_col="PredictorCategory",
            color_discrete_map=self.datamart.predictor_category_color_map,
            return_df=return_df,
        )
        if fig is not None and not return_df:
            fig.update_xaxes(title="Performance")

        return fig

    @requires(
        combined_columns={
            "PredictorName",
            "PredictorPerformance",
            "BinResponseCount",
            "PredictorCategory",
        },
    )
    def predictor_contribution(
        self,
        *,
        by: str = "Configuration",
        query: QUERY | None = None,
        return_df: bool = False,
    ):
        """Plots the predictor contribution for each configuration

        Parameters
        ----------
        by : str, optional
            By which column to plot the contribution, by default "Configuration"
        query : Optional[QUERY], optional
            An optional query to apply to the data, by default None
        return_df : bool, optional
            An optional flag to get a Dataframe instead, by default False

        Returns
        -------
        px.Figure
            A plotly figure

        See Also
        --------
        pdstools.adm.ADMDatamart.apply_predictor_categorization : how to override the out of the box predictor categorization

        Examples
        --------
        >>> # Default: contribution per Configuration
        >>> fig = dm.plot.predictor_contribution()

        >>> # Contribution grouped by Channel
        >>> fig = dm.plot.predictor_contribution(by="Channel")

        >>> # Return the contribution data for further processing
        >>> df = dm.plot.predictor_contribution(return_df=True)

        """
        df = (
            cdh_utils._apply_query(
                self.datamart.aggregates.last(table="combined_data"),
                query,
            )
            .filter(pl.col("PredictorName") != "Classifier")
            # Huh?
            .with_columns((pl.col("PredictorPerformance") - 0.5) * 2)
            .group_by(by, "PredictorCategory")
            .agg(
                Performance=cdh_utils.weighted_average_polars(
                    "PredictorPerformance",
                    "BinResponseCount",
                ),
            )
            .with_columns(
                Contribution=(pl.col("Performance") / pl.sum("Performance").over(by)) * 100,
            )
            .sort("PredictorCategory")
        )
        if return_df:
            return df

        import plotly.express as px

        fig = px.bar(
            df.collect(),
            x="Contribution",
            y=by,
            color="PredictorCategory",
            orientation="h",
            template="pega",
            title="Contribution of different sources",
        )
        return fig

    @requires(
        combined_columns={
            "PredictorName",
            "Name",
            "Performance",
            "PredictorPerformance",
            "ResponseCountBin",
        },
    )
    def predictor_performance_heatmap(
        self,
        *,
        top_predictors: int = 20,
        top_groups: int | None = None,
        by: str = "Name",
        active_only: bool = False,
        query: QUERY | None = None,
        return_df: bool = False,
    ):
        """Generate a heatmap showing predictor performance across different groups.

        Parameters
        ----------
        top_predictors : int, optional
            Number of top-performing predictors to include, by default 20
        top_groups : int, optional
            Number of top groups to include, by default None (all groups)
        by : str, optional
            Column to group by for the heatmap, by default "Name"
        active_only : bool, optional
            Whether to only include active predictors, by default False
        query : Optional[QUERY], optional
            Optional query to filter the data, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Union[Figure, pl.LazyFrame]
            Plotly heatmap figure or DataFrame if return_df=True

        Examples
        --------
        >>> # Default: top-20 predictors vs proposition (Name)
        >>> fig = dm.plot.predictor_performance_heatmap()

        >>> # Top-10 predictors across top-5 configurations, active only
        >>> fig = dm.plot.predictor_performance_heatmap(
        ...     top_predictors=10,
        ...     top_groups=5,
        ...     by="Configuration",
        ...     active_only=True,
        ... )

        >>> # Return the pivot data for further processing
        >>> df = dm.plot.predictor_performance_heatmap(return_df=True)

        """
        if isinstance(by, str):
            by_name = by
        else:
            by = by.alias("Predictor")
            by_name = "Predictor"

        df = self.datamart.aggregates.predictor_performance_pivot(
            query=query,
            by=by,
            top_predictors=top_predictors,
            top_groups=top_groups,
            active_only=active_only,
        )

        # Filter out rows with null values in the grouping column before transpose
        # to avoid "Column with new names can't have null values" error
        filtered_df = df.filter(pl.col(by_name).is_not_null()).collect()

        # Check if dataframe is empty after filtering
        if filtered_df.height == 0:
            if return_df:
                return pl.LazyFrame({by_name: []})
            return None

        collected = filtered_df.transpose(
            include_header=True,
            header_name=by_name,
            column_names=by_name,
        )

        if return_df:
            return collected.lazy()

        import plotly.express as px

        title = "over all models"
        fig = px.imshow(
            collected.select(pl.all().exclude(by_name)),
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale=get_colorscale("Performance"),
            title=f"Top predictors {title}",
            range_color=[0.5, 1],
            y=collected[by_name],
        )

        fig.update_yaxes(dtick=1, automargin=True)
        fig.update_xaxes(
            dtick=1,
        )
        return fig

    def predictor_count(
        self,
        *,
        by: str | list[str] = ["EntryType", "Type"],
        query: QUERY | None = None,
        return_df: bool = False,
    ):
        """Generate a box plot showing the distribution of predictor counts by type.

        Parameters
        ----------
        by : Union[str, list[str]], optional
            Column(s) to group predictors by, by default ["EntryType", "Type"]
        query : Optional[QUERY], optional
            Optional query to filter the data, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Union[Figure, pl.LazyFrame]
            Plotly box plot figure or DataFrame if return_df=True

        Examples
        --------
        >>> # Default: predictor count distribution by EntryType and Type
        >>> fig = dm.plot.predictor_count()

        >>> # Distribution by EntryType only
        >>> fig = dm.plot.predictor_count(by="EntryType")

        >>> # Return the raw counts
        >>> df = dm.plot.predictor_count(return_df=True)

        """
        # Normalize to list[str] — if `by` is a bare string, treating it as
        # Iterable[str] would iterate over characters, giving wrong group_by keys.
        by_list: list[str] = [by] if isinstance(by, str) else by

        df = cdh_utils._apply_query(
            self.datamart.aggregates.last(table="combined_data"),
            query,
        ).filter(pl.col("EntryType") != "Classifier")

        collected = df.group_by(["ModelID"] + by_list).agg(Count=pl.n_unique("PredictorName")).collect()

        if len(by_list) > 1:
            collected = pl.concat(
                [
                    collected,
                    collected.group_by(["ModelID"] + by_list[1:])
                    .agg(pl.col("Count").sum())
                    .with_columns(pl.lit("Overall").alias(by_list[0])),
                ],
                how="diagonal_relaxed",
            )
        collected = collected.sort(by_list)

        if return_df:
            return collected

        import plotly.express as px

        fig = px.box(
            collected.with_columns(_Type=pl.concat_str(reversed(by_list), separator=" / ")),
            x="Count",
            y="_Type",
            color=by_list[0],
            template="pega",
        )

        # Update title and x-axis label if we have a figure (not None and not a DataFrame)
        if fig is not None and not return_df:
            fig.update_layout(title="Predictors by Type")
            fig.update_xaxes(title="Number of Predictors")
            fig.update_layout(margin=dict(l=150))
            fig.update_yaxes(title="")
            # Reverse the legend order
            fig.update_layout(legend=dict(traceorder="reversed"))

        return fig
