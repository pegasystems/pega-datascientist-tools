"""Model-overview plots: bubble_chart, tree_map, action_overlap, partitioned_plot."""

from __future__ import annotations

from typing import Literal
from collections.abc import Callable

import polars as pl

from ...utils import cdh_utils
from ...utils.plot_utils import get_colorscale
from ...utils.types import QUERY
from ._base import _PlotsBase
from ._helpers import (
    add_bottom_left_text_to_bubble_plot,
    add_metric_limit_lines,
    requires,
)


class _OverviewPlotsMixin(_PlotsBase):
    @requires({"ModelID", "Performance", "SuccessRate", "ResponseCount", "Name"})
    def bubble_chart(
        self,
        *,
        last: bool = True,
        rounding: int = 5,
        query: QUERY | None = None,
        facet: str | pl.Expr | None = None,
        color: str | None = "Performance",
        show_metric_limits: bool = False,
        return_df: bool = False,
    ):
        """The Bubble Chart, as seen in Prediction Studio

        Parameters
        ----------
        last : bool, optional
            Whether to only include the latest snapshot, by default True
        rounding: int, optional
            To how many digits to round the performance number
        query : Optional[QUERY], optional
            The query to apply to the data, by default None
        facet : Optional[Union[str, pl.Expr]], optional
            Column name or Polars expression to facet the plot into subplots, by default None
        show_metric_limits : bool, optional
            Whether to show dashed vertical lines at the ModelPerformance
            metric limit thresholds (from MetricLimits.csv), by default False
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Figure | pl.LazyFrame
            Plotly scatter figure or LazyFrame if return_df=True

        Examples
        --------
        >>> from pdstools import ADMDatamart
        >>> dm = ADMDatamart.from_ds_export(base_path="/my_export_folder")

        >>> # Basic bubble chart using the latest snapshot
        >>> fig = dm.plot.bubble_chart()

        >>> # Facet by Channel to see performance per channel
        >>> fig = dm.plot.bubble_chart(facet="Channel")

        >>> # Include all snapshots and show metric-limit guidelines
        >>> fig = dm.plot.bubble_chart(last=False, show_metric_limits=True)

        >>> # Return the underlying data instead of the figure
        >>> df = dm.plot.bubble_chart(return_df=True)

        """
        # why do we need this select? it's not ideal because columns used in the query are not always selected
        columns_to_select = list(
            set(
                [
                    "ModelID",
                    color,
                    "SuccessRate",
                    "ResponseCount",
                    "ModelTechnique",
                    "SnapshotTime",
                    "LastUpdate",
                    "Configuration",
                    *self.datamart.context_keys,
                ],
            ),
        ) + (["Performance"] if color != "Performance" else [])
        if facet is not None:
            if isinstance(facet, pl.Expr):
                facet_columns = facet.meta.root_names()
                columns_to_select.extend(col for col in facet_columns if col not in columns_to_select)
                facet_name = facet.meta.output_name()
            else:
                if facet not in columns_to_select:
                    columns_to_select.append(facet)
                facet_name = facet
        else:
            facet_name = None
        df = (
            (self.datamart.aggregates.last() if last else self.datamart._require_model_data())
            .select(*columns_to_select)
            .with_columns((pl.col("Performance") * pl.lit(100)).round(rounding))
        )

        if facet_name is not None and isinstance(facet, pl.Expr):
            df = df.with_columns(facet.alias(facet_name))
        df = cdh_utils._apply_query(df, query)

        if return_df:
            return df

        import plotly.express as px

        title = "over all models"
        fig = px.scatter(
            df.collect().with_columns(
                pl.col("LastUpdate").dt.strftime("%v"),
                pl.col(self.datamart.context_keys).fill_null(""),
            ),
            x="Performance",
            y="SuccessRate",
            color=color,
            size="ResponseCount",
            facet_col=facet_name,
            facet_col_wrap=2,
            hover_name="Name",
            hover_data=["ModelID"] + self.datamart.context_keys + ["LastUpdate"],
            title=f"Bubble Chart {title}",
            template="pega",
            facet_row_spacing=0.01,
            labels={"LastUpdate": "Last Updated"},
        )
        fig = add_bottom_left_text_to_bubble_plot(fig, df, 1)
        if show_metric_limits:
            fig = add_metric_limit_lines(fig)
        fig.update_traces(marker=dict(line=dict(color="black")))
        fig.update_yaxes(tickformat=".3%")
        return fig

    def tree_map(
        self,
        metric: Literal[
            "ResponseCount",
            "Positives",
            "Performance",
            "SuccessRate",
            "percentage_without_responses",
        ] = "Performance",
        *,
        by: str = "Name",
        query: QUERY | None = None,
        return_df: bool = False,
    ):
        """Generate a tree map visualization showing hierarchical model metrics.

        Parameters
        ----------
        metric : Literal["ResponseCount", "Positives", "Performance", "SuccessRate", "percentage_without_responses"], optional
            The metric to visualize in the tree map, by default "Performance"
        by : str, optional
            Column to group by for the tree map hierarchy, by default "Name"
        query : Optional[QUERY], optional
            Optional query to filter the data, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Union[Figure, pl.LazyFrame]
            Plotly treemap figure or DataFrame if return_df=True

        Examples
        --------
        >>> # Default: Performance tree map grouped by proposition name
        >>> fig = dm.plot.tree_map()

        >>> # ResponseCount tree map grouped by Channel
        >>> fig = dm.plot.tree_map(metric="ResponseCount", by="Channel")

        >>> # Return underlying summary data
        >>> df = dm.plot.tree_map(return_df=True)

        """
        # TODO: clean up implementation a bit

        group_by = self.datamart.context_keys[: self.datamart.context_keys.index(by)] if by != "ModelID" else [by]
        df = self.datamart.aggregates.model_summary(by=by, query=query).select(
            pl.col(group_by).cast(pl.Utf8).fill_null("Missing"),
            pl.col("count").alias("Model Count"),
            pl.col("Percentage_without_responses").alias(
                "Percentage without responses",
            ),
            pl.col("ResponseCount_sum").alias("Total number of responses"),
            pl.col("Weighted_success_rate").alias("Weighted average Success Rate"),
            pl.col("Weighted_performance").alias("Weighted average Performance"),
            pl.col("Positives_sum").alias("Total number of positives"),
        )
        if return_df:
            return df

        import plotly.express as px

        context_keys = [px.Constant("All contexts")] + group_by

        colorscale = get_colorscale(metric)

        label_map = {
            "ResponseCount": "Total number of responses",
            "Positives": "Total number of positives",
            "Performance": "Weighted average Performance",
            "SuccessRate": "Weighted average Success Rate",
            "percentage_without_responses": "Percentage without responses",
        }

        hover_data = {
            "Model Count": ":.d",
            "Percentage without responses": ":.0%",
            "Total number of positives": ":.d",
            "Total number of responses": ":.d",
            "Weighted average Success Rate": ":.3%",
            "Weighted average Performance": ":.2%",
        }

        fig = px.treemap(
            df.collect(),
            path=context_keys,
            color=label_map.get(metric),
            values="Model Count",
            title=f"{label_map.get(metric)} by {by}",
            hover_data=hover_data,
            color_continuous_scale=colorscale,
            range_color=[0.5, 1] if metric == "Performance" else None,
        )

        return fig

    def action_overlap(
        self,
        group_col: str | list[str] | pl.Expr = "Channel",
        overlap_col="Name",
        *,
        show_fraction=True,
        query: QUERY | None = None,
        return_df: bool = False,
    ):
        """Generate an overlap matrix heatmap showing shared actions across different groups.

        Parameters
        ----------
        group_col : Union[str, list[str], pl.Expr], optional
            Column(s) to group by for overlap analysis, by default "Channel"
        overlap_col : str, optional
            Column containing values to analyze for overlap, by default "Name"
        show_fraction : bool, optional
            Whether to show overlap as fraction or absolute count, by default True
        query : Optional[QUERY], optional
            Optional query to filter the data, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Union[Figure, pl.LazyFrame]
            Plotly heatmap showing action overlap or DataFrame if return_df=True

        Examples
        --------
        >>> # Default: action overlap across Channels
        >>> fig = dm.plot.action_overlap()

        >>> # Overlap shown as absolute counts rather than fractions
        >>> fig = dm.plot.action_overlap(show_fraction=False)

        >>> # Overlap of configurations across Channel/Direction combinations
        >>> fig = dm.plot.action_overlap(
        ...     group_col=["Channel", "Direction"],
        ...     overlap_col="Configuration",
        ... )

        >>> # Return the overlap matrix as a DataFrame
        >>> df = dm.plot.action_overlap(return_df=True)

        """
        df = cdh_utils._apply_query(
            self.datamart._require_model_data(),
            query,
        )

        if isinstance(group_col, list):
            group_col_name = "/".join(group_col)
            df = df.with_columns(
                pl.concat_str(*group_col, separator="/").alias(group_col_name),
            )
        elif isinstance(group_col, pl.Expr):
            group_col_name = group_col.meta.output_name()
            df = df.with_columns(group_col.alias(group_col_name))
        else:
            group_col_name = group_col

        overlap_data = cdh_utils.overlap_matrix(
            df.group_by(group_col_name).agg(pl.col(overlap_col).unique()).sort(group_col_name).collect(),
            overlap_col,
            by=group_col_name,
            show_fraction=show_fraction,
        )

        if return_df:
            return overlap_data

        import plotly.express as px

        plt = px.imshow(
            overlap_data.drop(group_col_name),
            text_auto=".1%" if show_fraction else ".d",
            aspect="equal",
            title=f"Overlap of {overlap_col}s",
            x=overlap_data[group_col_name],
            y=overlap_data[group_col_name],
            template="pega",
            labels=dict(
                x=f"{group_col_name} on x",
                y=f"{group_col_name} on y",
                color="Overlap",
            ),
        )
        plt.update_coloraxes(showscale=False)
        return plt

    def partitioned_plot(
        self,
        func: Callable,
        facets: list[dict[str, str | None]],
        show_plots: bool = True,
        *args,
        **kwargs,
    ):
        """Execute a plotting function across multiple faceted subsets of data.

        This method applies a given plotting function to multiple filtered subsets of data,
        where each subset is defined by the facet conditions. It's useful for generating
        multiple plots with different filter conditions applied.

        Parameters
        ----------
        func : Callable
            The plotting function to execute for each facet
        facets : list[dict[str, Optional[str]]]
            list of dictionaries defining filter conditions for each facet
        show_plots : bool, optional
            Whether to display the plots as they are generated, by default True
        *args : tuple
            Additional positional arguments to pass to the plotting function
        **kwargs : dict
            Additional keyword arguments to pass to the plotting function

        Returns
        -------
        list[Figure]
            list of Plotly figures, one for each facet condition

        Examples
        --------
        >>> # Bubble chart separately for Web and Email channels
        >>> figs = dm.plot.partitioned_plot(
        ...     dm.plot.bubble_chart,
        ...     facets=[{"Channel": "Web"}, {"Channel": "Email"}],
        ... )

        >>> # Same but suppress auto-display
        >>> figs = dm.plot.partitioned_plot(
        ...     dm.plot.bubble_chart,
        ...     facets=[{"Channel": "Web"}, {"Channel": "Email"}],
        ...     show_plots=False,
        ... )

        """
        figs = []
        existing_query = kwargs.get("query")
        for facet in facets:
            combined_query = existing_query
            for k, v in facet.items():
                if v is None:
                    new_query = pl.col(k).is_null()
                else:
                    new_query = pl.col(k).eq(v)
                if combined_query is not None:
                    combined_query = cdh_utils._combine_queries(
                        combined_query,
                        new_query,
                    )
                else:
                    combined_query = new_query

            kwargs["query"] = combined_query
            fig = func(*args, **kwargs)
            figs.append(fig)
        if show_plots and fig is not None:
            # fig.update_layout(title=title)
            fig.show()

        return figs
