"""Plotting utilities for Impact Analyzer visualization."""

import logging
from typing import TYPE_CHECKING, List, Optional, Union

import polars as pl

from ..utils.cdh_utils import _apply_query
from ..utils.namespaces import LazyNamespace
from ..utils.types import QUERY

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from plotly.graph_objs import Figure

    from .ImpactAnalyzer import ImpactAnalyzer as ImpactAnalyzer_Class

try:
    import plotly.express as px
except ImportError as e:  # pragma: no cover
    logger.debug(f"Failed to import optional dependencies: {e}")


class Plots(LazyNamespace):
    """Visualization methods for Impact Analyzer experiment data.

    This class provides plotting capabilities for analyzing Impact Analyzer
    experiment results. It is accessed through the `plot` attribute of an
    :class:`~pdstools.impactanalyzer.ImpactAnalyzer.ImpactAnalyzer` instance.

    All plot methods support:
    - Custom titles via `title` parameter
    - Data filtering via `query` parameter
    - Faceting by dimension via `facet` parameter
    - Returning underlying data via `return_df=True`

    Attributes
    ----------
    ia : ImpactAnalyzer
        Reference to the parent ImpactAnalyzer instance.

    See Also
    --------
    pdstools.impactanalyzer.ImpactAnalyzer : Main analysis class.

    Examples
    --------
    >>> ia = ImpactAnalyzer.from_pdc("export.json")
    >>> ia.plot.overview()
    >>> ia.plot.trend(metric="Value_Lift", facet="Channel")
    """

    def __init__(self, ia: "ImpactAnalyzer_Class"):
        """Initialize a Plots instance.

        Parameters
        ----------
        ia : ImpactAnalyzer
            The parent ImpactAnalyzer instance providing the data.
        """
        super().__init__()
        self.ia = ia

    @staticmethod
    def _get_experiment_color_map() -> List[str]:
        """Get ordered list of experiment names for consistent coloring.

        Returns
        -------
        List[str]
            Alphabetically ordered list of default experiment names.
        """
        default_experiments = [
            "Adaptive Models vs Random Propensity",
            "NBA vs No Levers",
            "NBA vs Only Eligibility Rules",
            "NBA vs Propensity Only",
            "NBA vs Random",
        ]
        return default_experiments

    @staticmethod
    def _get_facet_config(data: pl.DataFrame, facet: Optional[str]) -> dict:
        """Determine optimal faceting configuration.

        Automatically selects column wrapping based on the number of
        distinct facet values for better layout.

        Parameters
        ----------
        data : pl.DataFrame
            The collected plot data.
        facet : str or None
            Column name for faceting, or None for no faceting.

        Returns
        -------
        dict
            Configuration with keys 'facet_col' and 'facet_col_wrap'.
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
        metric: str = "CTR_Lift",
        facet: Optional[str] = None,
        return_df: bool = False,
    ) -> Union["Figure", pl.LazyFrame]:
        """Create a bar chart comparing experiment performance.

        Displays a horizontal bar chart comparing Impact Analyzer experiments
        for the specified lift metric.

        Parameters
        ----------
        title : str, optional
            Custom title. If None, auto-generated from metric.
        query : QUERY, optional
            Polars expression to filter the data.
        metric : str, default "CTR_Lift"
            Metric to display: "CTR_Lift" or "Value_Lift".
        facet : str, optional
            Column name for faceting (e.g., "Channel").
        return_df : bool, default False
            If True, return data as LazyFrame instead of figure.

        Returns
        -------
        Figure or pl.LazyFrame
            Plotly figure, or LazyFrame if `return_df=True`.

        See Also
        --------
        trend : Time series of experiment metrics.
        control_groups_trend : Time series of control group metrics.

        Examples
        --------
        >>> ia.plot.overview()
        >>> ia.plot.overview(metric="Value_Lift", facet="Channel")
        >>> df = ia.plot.overview(return_df=True)
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
            title=title or f"Overview Impact Analyzer Experiments for {metric}",
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
        metric: str = "CTR",
        facet: Optional[str] = None,
        every: Optional[str] = None,
        return_df: bool = False,
    ) -> Union["Figure", pl.LazyFrame]:
        """Create a line chart of control group metrics over time.

        Displays how different control groups perform over time for
        the specified metric.

        Parameters
        ----------
        title : str, optional
            Custom title. If None, auto-generated from metric.
        query : QUERY, optional
            Polars expression to filter the data.
        metric : str, default "CTR"
            Metric to display: "CTR" or "ValuePerImpression".
        facet : str, optional
            Column name for faceting (e.g., "Channel").
        every : str, optional
            Time aggregation period using Polars syntax:
            "1d" (daily), "1w" (weekly), "1mo" (monthly), etc.
        return_df : bool, default False
            If True, return data as LazyFrame instead of figure.

        Returns
        -------
        Figure or pl.LazyFrame
            Plotly figure, or LazyFrame if `return_df=True`.

        See Also
        --------
        trend : Time series of experiment lift metrics.
        overview : Bar chart of experiment metrics.

        Examples
        --------
        >>> ia.plot.control_groups_trend()
        >>> ia.plot.control_groups_trend(metric="ValuePerImpression", every="1w")
        >>> ia.plot.control_groups_trend(facet="Channel", every="1mo")
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
        metric: str = "CTR_Lift",
        facet: Optional[str] = None,
        every: Optional[str] = None,
        return_df: bool = False,
    ) -> Union["Figure", pl.LazyFrame]:
        """Create a line chart of experiment lift metrics over time.

        Displays how different experiments' lift metrics evolve over time.

        Parameters
        ----------
        title : str, optional
            Custom title. If None, auto-generated from metric.
        query : QUERY, optional
            Polars expression to filter the data.
        metric : str, default "CTR_Lift"
            Metric to display: "CTR_Lift" or "Value_Lift".
        facet : str, optional
            Column name for faceting (e.g., "Channel").
        every : str, optional
            Time aggregation period using Polars syntax:
            "1d" (daily), "1w" (weekly), "1mo" (monthly), etc.
        return_df : bool, default False
            If True, return data as LazyFrame instead of figure.

        Returns
        -------
        Figure or pl.LazyFrame
            Plotly figure, or LazyFrame if `return_df=True`.

        See Also
        --------
        overview : Bar chart of experiment metrics.
        control_groups_trend : Time series of control group metrics.

        Examples
        --------
        >>> ia.plot.trend()
        >>> ia.plot.trend(metric="Value_Lift", every="1w")
        >>> ia.plot.trend(facet="Channel", every="1mo")
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
