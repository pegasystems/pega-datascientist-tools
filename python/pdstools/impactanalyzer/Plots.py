import logging
from typing import TYPE_CHECKING, List, Optional

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
        by: Optional[List[str]] = None,
        title: Optional[str] = None,
        query: Optional[QUERY] = None,
        metric: Optional[str] = "CTR_Lift",
        facet: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        if by is None:
            by = []

        if facet is not None and facet not in by:
            by = by + [facet]

        plot_data = self.ia.summarize_experiments(by=by)

        # Filter out rows with invalid metric values (NaN, Infinity)
        plot_data = plot_data.filter(
            pl.col(metric).is_not_null() & pl.col(metric).is_finite()
        )

        if query is not None:
            plot_data = _apply_query(plot_data, query=query)

        if return_df:
            return plot_data

        if title is None:
            metric_name = "CTR Lift" if metric == "CTR_Lift" else "Value Lift"
            title = f"Overview Impact Analyzer Experiments - {metric_name}"

        tick_format = ".1%" if metric == "CTR_Lift" else ".2f"
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
            title=title,
            margin=dict(l=60, r=60, t=80, b=80),
        )

        fig.update_yaxes(automargin=True, title="")
        fig.update_xaxes(
            title="",
            tickformat=tick_format,
            matches=None if facet is not None else "x",
            showticklabels=True,
        )

        return fig

    def control_groups_trend(
        self,
        *,
        by: Optional[List[str]] = None,
        title: Optional[str] = None,
        query: Optional[QUERY] = None,
        metric: Optional[str] = "CTR",
        facet: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        if by is None:
            by = ["SnapshotTime"]

        if facet is not None and facet not in by:
            by = by + [facet]

        plot_data = self.ia.summarize_control_groups(by=by)

        # Filter out rows with invalid metric values (NaN, Infinity)
        plot_data = plot_data.filter(
            pl.col(metric).is_not_null() & pl.col(metric).is_finite()
        )

        if query is not None:
            plot_data = _apply_query(plot_data, query=query)

        if return_df:
            return plot_data

        if title is None:
            metric_name = "CTR" if metric == "CTR" else "Value Per Impression"
            title = f"Trend of {metric_name} for Impact Analyzer Control Groups"

        tick_format = ".1%" if metric == "CTR" else ".2f"
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
            title=title,
            margin=dict(l=60, r=60, t=80, b=80),
        )

        fig.update_yaxes(
            title="",
            tickformat=tick_format,
            matches=None if facet is not None else "y",
            showticklabels=True,
        )
        fig.update_xaxes(title="")

        return fig

    def trend(
        self,
        *,
        by: Optional[List[str]] = None,
        title: Optional[str] = None,
        query: Optional[QUERY] = None,
        metric: Optional[str] = "CTR_Lift",
        facet: Optional[str] = None,
        return_df: Optional[bool] = False,
    ):
        if by is None:
            by = ["SnapshotTime"]

        if facet is not None and facet not in by:
            by = by + [facet]

        plot_data = self.ia.summarize_experiments(by=by)

        # Filter out rows with invalid metric values (NaN, Infinity)
        plot_data = plot_data.filter(
            pl.col(metric).is_not_null() & pl.col(metric).is_finite()
        )

        if query is not None:
            plot_data = _apply_query(plot_data, query=query)

        if return_df:
            return plot_data

        if title is None:
            metric_name = "CTR Lift" if metric == "CTR_Lift" else "Value Lift"
            title = f"Trend of {metric_name} for Impact Analyzer Experiments"

        tick_format = ".1%" if metric == "CTR_Lift" else ".2f"
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
            title=title,
            margin=dict(l=60, r=60, t=80, b=80),
        )

        fig.update_yaxes(
            title="",
            tickformat=tick_format,
            matches=None if facet is not None else "y",
            showticklabels=True,
        )
        fig.update_xaxes(title="")

        return fig
