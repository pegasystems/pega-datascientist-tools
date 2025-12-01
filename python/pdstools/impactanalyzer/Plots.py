import logging
from typing import TYPE_CHECKING, List, Optional

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

        plot_data = self.ia.summarize_experiments(by=by)

        if query is not None:
            plot_data = _apply_query(plot_data, query=query)

        if return_df:
            return plot_data

        if title is None:
            metric_name = "CTR Lift" if metric == "CTR_Lift" else "Value Lift"
            title = f"Overview Impact Analyzer Experiments - {metric_name}"

        # Determine axis formatting based on metric
        axis_title = "CTR Lift" if metric == "CTR_Lift" else "Value Lift"
        tick_format = ".1%" if metric == "CTR_Lift" else ".2f"

        # todo add some faceting if by != None
        fig = px.bar(
            plot_data.collect(),
            y="Experiment",
            x=metric,
            color="Experiment",
            facet_col=facet,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            category_orders={"Experiment": self._get_experiment_color_map()},
            template="pega",
        )

        fig.update_layout(
            yaxis=dict(
                automargin=True,
                title="",
            ),
            xaxis=dict(
                title=axis_title,
                tickformat=tick_format,
            ),
            showlegend=False,
            title=title,
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

        plot_data = self.ia.summarize_control_groups(by=by)

        if query is not None:
            plot_data = _apply_query(plot_data, query=query)

        if return_df:
            return plot_data

        if title is None:
            metric_name = "CTR" if metric == "CTR" else "Value Per Impression"
            title = f"Trend of {metric_name} for Impact Analyzer Control Groups"

        # Determine axis formatting based on metric
        axis_title = "CTR" if metric == "CTR" else "Value Per Impression"
        tick_format = ".1%" if metric == "CTR" else ".2f"

        fig = px.line(
            plot_data.collect(),
            y=metric,
            x="SnapshotTime",
            color="ControlGroup",
            facet_col=facet,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            template="pega",
        )

        fig.update_layout(
            yaxis=dict(
                title=axis_title,
                tickformat=tick_format,
            ),
            xaxis=dict(
                title="",
            ),
            hovermode="x unified",
            title=title,
        )

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
            by = [
                "SnapshotTime"
            ]  # todo or perhaps + Channel, if so use for faceting maybe

        plot_data = self.ia.summarize_experiments(by=by)

        if query is not None:
            plot_data = _apply_query(plot_data, query=query)

        if return_df:
            return plot_data

        if title is None:
            metric_name = "CTR Lift" if metric == "CTR_Lift" else "Value Lift"
            title = f"Trend of {metric_name} for Impact Analyzer Experiments"

        # Determine axis formatting based on metric
        axis_title = "CTR Lift" if metric == "CTR_Lift" else "Value Lift"
        tick_format = ".1%" if metric == "CTR_Lift" else ".2f"

        fig = px.line(
            plot_data.collect(),
            y=metric,
            x="SnapshotTime",
            color="Experiment",
            facet_col=facet,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            category_orders={"Experiment": self._get_experiment_color_map()},
            template="pega",
        )

        fig.update_layout(
            yaxis=dict(
                title=axis_title,
                tickformat=tick_format,
            ),
            xaxis=dict(
                title="",
            ),
            hovermode="x unified",
            title=title,
        )

        return fig
