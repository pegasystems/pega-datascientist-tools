import logging
from typing import TYPE_CHECKING, Optional

import polars as pl
from ..utils.cdh_utils import _apply_query
from ..utils.namespaces import LazyNamespace
from ..utils.types import QUERY

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from .ImpactAnalyzer import ImpactAnalyzer as ImpactAnalyzer_Class
try:
    import plotly as plotly
    import plotly.express as px


except ImportError as e:  # pragma: no cover
    logger.debug(f"Failed to import optional dependencies: {e}")


class Plots(LazyNamespace):
    def __init__(self, ia: "ImpactAnalyzer_Class"):
        super().__init__()
        self.ia = ia

    @staticmethod
    def _get_experiment_color_map():
        """Get consistent color mapping for default experiments."""
        default_experiments = [
            "Adaptive Models vs Random Propensity",
            "NBA vs No Levers",
            "NBA vs Only Eligibility Rules",
            "NBA vs Propensity Only",
            "NBA vs Random",
        ]
        return default_experiments

    @staticmethod
    def _get_facet_config(data, facet):
        """
        Determine optimal faceting configuration based on number of distinct facet values.

        For <= 3 distinct values: use column faceting only
        For > 3 distinct values: use column wrapping for better layout

        Parameters
        ----------
        data : polars.DataFrame
            The plot data (already collected)
        facet : str or None
            The facet column name

        Returns
        -------
        dict
            Dictionary with 'facet_col' and 'facet_col_wrap' keys
        """
        if facet is None:
            return {"facet_col": None, "facet_col_wrap": None}

        if facet not in data.columns:
            return {"facet_col": facet, "facet_col_wrap": None}

        distinct_values = data[facet].n_unique()

        if distinct_values <= 3:
            return {"facet_col": facet, "facet_col_wrap": None}
        elif distinct_values == 4:
            return {"facet_col": facet, "facet_col_wrap": 2}
        else:
            return {"facet_col": facet, "facet_col_wrap": 3}

    def overview(
        self,
        *,
        title: Optional[str] = None,
        query: Optional[QUERY] = None,
        metric: Optional[str] = "CTR_Lift",
        facet: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        """Create an overview bar chart showing experiment performance metrics.

        Displays a horizontal bar chart comparing different Impact Analyzer experiments
        across the specified metric (CTR_Lift or Value_Lift).

        Parameters
        ----------
        title : Optional[str], optional
            Custom title for the plot. If None, generates a default title based on the metric.
        query : Optional[QUERY], optional
            Query to filter the experiment data. See pdstools.utils.cdh_utils._apply_query for details.
        metric : Optional[str], optional
            The metric to display on the x-axis, by default "CTR_Lift".
            Options: "CTR_Lift", "Value_Lift".
        facet : Optional[str], optional
            Column name to create separate subplots for each unique value, by default None.
            Common options: "Channel".
        return_df : Optional[bool], optional
            If True, returns the underlying data instead of the plot, by default False.

        Returns
        -------
        Union[plotly.graph_objects.Figure, polars.LazyFrame]
            A Plotly figure object or the underlying data if return_df is True.

        Examples
        --------
        >>> # Basic overview plot
        >>> ia.plot.overview()

        >>> # Overview with Value_Lift metric
        >>> ia.plot.overview(metric="Value_Lift")

        >>> # Faceted by Channel
        >>> ia.plot.overview(facet="Channel")

        >>> # Get underlying data
        >>> data = ia.plot.overview(return_df=True)
        """
        grouping_columns = [facet] if facet is not None else []
        plot_data = self.ia.summarize_experiments(by=grouping_columns)

        # Filter out rows with invalid metric values (NaN, Infinity)
        plot_data = plot_data.filter(
            pl.col(metric).is_not_null() & pl.col(metric).is_finite()
        )

        if query is not None:
            plot_data = _apply_query(plot_data, query=query)

        if return_df:
            return plot_data

        collected_data = plot_data.collect()
        facet_config = self._get_facet_config(collected_data, facet)

        fig = px.bar(
            collected_data,
            y="Experiment",
            x=metric,
            color="Experiment",
            facet_col=facet_config["facet_col"],
            facet_col_wrap=facet_config["facet_col_wrap"],
            facet_col_spacing=0.1,
            facet_row_spacing=0.25,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            category_orders={"Experiment": self._get_experiment_color_map()},
            template="pega",
        )

        fig.update_layout(
            showlegend=False,
            title=title or f"Overview Impact Analyzer Experiments - {metric}",
            margin=dict(l=60, r=60, t=80, b=80),
        )

        fig.update_yaxes(automargin=True, title="")
        fig.update_xaxes(
            title="",
            tickformat=".1%",
            matches=None if facet is not None else "x",
            showticklabels=True,
        )

        return fig

    def control_groups_trend(
        self,
        *,
        title: Optional[str] = None,
        query: Optional[QUERY] = None,
        metric: Optional[str] = "CTR",
        facet: Optional[str] = None,
        every: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        """Create a trend line chart showing control group performance over time.

        Displays a line chart showing how different Impact Analyzer control groups
        perform over time for the specified metric (CTR or ValuePerImpression).

        Parameters
        ----------
        title : Optional[str], optional
            Custom title for the plot. If None, generates a default title based on the metric.
        query : Optional[QUERY], optional
            Query to filter the control group data. See pdstools.utils.cdh_utils._apply_query for details.
        metric : Optional[str], optional
            The metric to display on the y-axis, by default "CTR".
            Options: "CTR", "ValuePerImpression".
        facet : Optional[str], optional
            Column name to create separate subplots for each unique value, by default None.
            Common options: "Channel".
        every : Optional[str], optional
            Time period for aggregating timestamps, by default None.
            Uses Polars truncate syntax: "1d", "1w", "1mo", "1q", "1y".
            When specified, timestamps are truncated to the given period before summarization.
        return_df : Optional[bool], optional
            If True, returns the underlying data instead of the plot, by default False.

        Returns
        -------
        Union[plotly.graph_objects.Figure, polars.LazyFrame]
            A Plotly figure object or the underlying data if return_df is True.

        Examples
        --------
        >>> # Basic control groups trend plot
        >>> ia.plot.control_groups_trend()

        >>> # Control groups trend with ValuePerImpression metric
        >>> ia.plot.control_groups_trend(metric="ValuePerImpression")

        >>> # Weekly aggregated trend
        >>> ia.plot.control_groups_trend(every="1w")

        >>> # Faceted by Channel with monthly aggregation
        >>> ia.plot.control_groups_trend(facet="Channel", every="1mo")

        >>> # Get underlying data
        >>> data = ia.plot.control_groups_trend(return_df=True)
        """
        if every is not None:
            grouping_columns = [pl.col("SnapshotTime").dt.truncate(every)] + (
                [facet] if facet is not None else []
            )
        else:
            grouping_columns = ["SnapshotTime"] + ([facet] if facet is not None else [])

        plot_data = self.ia.summarize_control_groups(by=grouping_columns).filter(
            pl.col(metric).is_not_null() & pl.col(metric).is_finite()
        )

        if query is not None:
            plot_data = _apply_query(plot_data, query=query)

        if return_df:
            return plot_data

        collected_data = plot_data.collect()
        facet_config = self._get_facet_config(collected_data, facet)

        fig = px.line(
            collected_data,
            y=metric,
            x="SnapshotTime",
            color="ControlGroup",
            facet_col=facet_config["facet_col"],
            facet_col_wrap=facet_config["facet_col_wrap"],
            facet_col_spacing=0.1,
            facet_row_spacing=0.15,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            template="pega",
        )

        fig.update_layout(
            hovermode="x unified",
            title=title or f"Trend of {metric} for Impact Analyzer Control Groups",
            margin=dict(l=60, r=60, t=80, b=80),
        )

        fig.update_yaxes(
            title="",
            tickformat=".1%",
            matches=None if facet is not None else "y",
            showticklabels=True,
        )
        fig.update_xaxes(title="")

        return fig

    def trend(
        self,
        *,
        title: Optional[str] = None,
        query: Optional[QUERY] = None,
        metric: Optional[str] = "CTR_Lift",
        facet: Optional[str] = None,
        every: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        """Create a trend line chart showing experiment performance over time.

        Displays a line chart showing how different Impact Analyzer experiments
        perform over time for the specified metric (CTR_Lift or Value_Lift).

        Parameters
        ----------
        title : Optional[str], optional
            Custom title for the plot. If None, generates a default title based on the metric.
        query : Optional[QUERY], optional
            Query to filter the experiment data. See pdstools.utils.cdh_utils._apply_query for details.
        metric : Optional[str], optional
            The metric to display on the y-axis, by default "CTR_Lift".
            Options: "CTR_Lift", "Value_Lift".
        facet : Optional[str], optional
            Column name to create separate subplots for each unique value, by default None.
            Common options: "Channel".
        every : Optional[str], optional
            Time period for aggregating timestamps, by default None.
            Uses Polars truncate syntax: "1d", "1w", "1mo", "1q", "1y".
            When specified, timestamps are truncated to the given period before summarization.
        return_df : Optional[bool], optional
            If True, returns the underlying data instead of the plot, by default False.

        Returns
        -------
        Union[plotly.graph_objects.Figure, polars.LazyFrame]
            A Plotly figure object or the underlying data if return_df is True.

        Examples
        --------
        >>> # Basic experiment trend plot
        >>> ia.plot.trend()

        >>> # Experiment trend with Value_Lift metric
        >>> ia.plot.trend(metric="Value_Lift")

        >>> # Weekly aggregated trend
        >>> ia.plot.trend(every="1w")

        >>> # Faceted by Channel with monthly aggregation
        >>> ia.plot.trend(facet="Channel", every="1mo")

        >>> # Get underlying data
        >>> data = ia.plot.trend(return_df=True)
        """
        if every is not None:
            grouping_columns = [pl.col("SnapshotTime").dt.truncate(every)] + (
                [facet] if facet is not None else []
            )
        else:
            grouping_columns = ["SnapshotTime"] + ([facet] if facet is not None else [])

        plot_data = self.ia.summarize_experiments(by=grouping_columns).filter(
            pl.col(metric).is_not_null() & pl.col(metric).is_finite()
        )

        if query is not None:
            plot_data = _apply_query(plot_data, query=query)

        if return_df:
            return plot_data

        collected_data = plot_data.collect()
        facet_config = self._get_facet_config(collected_data, facet)

        fig = px.line(
            collected_data,
            y=metric,
            x="SnapshotTime",
            color="Experiment",
            facet_col=facet_config["facet_col"],
            facet_col_wrap=facet_config["facet_col_wrap"],
            facet_col_spacing=0.1,
            facet_row_spacing=0.15,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            category_orders={"Experiment": self._get_experiment_color_map()},
            template="pega",
        )

        fig.update_layout(
            hovermode="x unified",
            title=title or f"Trend of {metric} for Impact Analyzer Experiments",
            margin=dict(l=60, r=60, t=80, b=80),
        )

        fig.update_yaxes(
            title="",
            tickformat=".1%",
            matches=None if facet is not None else "y",
            showticklabels=True,
        )
        fig.update_xaxes(title="")

        return fig
