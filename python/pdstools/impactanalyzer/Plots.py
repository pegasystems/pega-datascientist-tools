import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import polars as pl

from ..utils import cdh_utils
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

    def overview(
        self,
        *,
        by: List[str] = None,
        title: Optional[str] = None,
        query: Optional[QUERY] = None,
        return_df: Optional[bool] = False,
    ):
        import plotly as plotly
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if by is None:
            by = []

        plot_data = self.ia.summarize_experiments(by=by)

        if return_df:
            return plot_data

        if title is None:
            title = "Overview Impact Analyzer Experiments"

        # todo add some faceting if by != None
        fig = px.bar(
            plot_data.collect(),
            y="Experiment",
            x="CTR_Lift",
            template="pega",
        )

        return fig

    def trend(
        self,
        *,
        by: List[str] = None,
        title: Optional[str] = None,
        query: Optional[QUERY] = None,
        return_df: Optional[bool] = False,
    ):
        import plotly as plotly
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if by is None:
            by = ["SnapshotTime"] # todo or perhaps + Channel, if so use for faceting maybe

        plot_data = self.ia.summarize_experiments(by=by)

        if return_df:
            return plot_data

        if title is None:
            title = "Trend of CTR Lift for Impact Analyzer Experiments"

        fig = px.line(
            plot_data.collect(),
            y="CTR_Lift",
            x="SnapshotTime",
            color="Experiment",
            template="pega",
        )

        return fig
