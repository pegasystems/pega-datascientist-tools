"""Plot methods for :class:`pdstools.adm.ADMDatamart`.

This package decomposes the legacy ``Plots`` god-class into mixins
grouped by concern. The public ``Plots`` class composes those mixins
so that ``dm.plot.<method>`` continues to expose the same surface as
before — no public method names, signatures or return types change.

Layout
------
- ``_helpers``    – ``requires`` decorator, ``MetricLimits`` annotation
  helper, and small plotly building blocks reused by multiple mixins.
- ``_overview``   – :meth:`bubble_chart`, :meth:`tree_map`,
  :meth:`action_overlap`, :meth:`partitioned_plot`.
- ``_performance`` – :meth:`over_time`, :meth:`gains_chart`,
  :meth:`performance_volume_distribution`,
  :meth:`proposition_success_rates`.
- ``_score``      – :meth:`score_distribution`,
  :meth:`multiple_score_distributions`.
- ``_predictors`` – :meth:`predictor_performance`,
  :meth:`predictor_category_performance`, :meth:`predictor_contribution`,
  :meth:`predictor_performance_heatmap`, :meth:`predictor_count`.
- ``_binning``    – :meth:`predictor_binning`,
  :meth:`multiple_predictor_binning`, :meth:`binning_lift`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...utils.namespaces import LazyNamespace
from ...utils.plot_utils import fig_update_facet
from ._base import _PlotsBase
from ._binning import _BinningPlotsMixin
from ._helpers import (
    COLORSCALE_TYPES,
    add_bottom_left_text_to_bubble_plot,
    add_metric_limit_lines,
    distribution_graph,
    requires,
)
from ._overview import _OverviewPlotsMixin
from ._performance import _PerformancePlotsMixin
from ._predictors import _PredictorPlotsMixin
from ._score import _ScorePlotsMixin

if TYPE_CHECKING:  # pragma: no cover
    from ..ADMDatamart import ADMDatamart

__all__ = [
    "COLORSCALE_TYPES",
    "Plots",
    "add_bottom_left_text_to_bubble_plot",
    "add_metric_limit_lines",
    "distribution_graph",
    "fig_update_facet",
    "requires",
]


class Plots(
    _OverviewPlotsMixin,
    _PerformancePlotsMixin,
    _ScorePlotsMixin,
    _PredictorPlotsMixin,
    _BinningPlotsMixin,
    _PlotsBase,
    LazyNamespace,
):
    """Namespace exposing all out-of-the-box ADM datamart plots.

    Methods are defined across the ``_overview``, ``_performance``,
    ``_score``, ``_predictors`` and ``_binning`` submodules and composed
    here. The class is consumed via ``dm.plot.<method>`` on an
    :class:`~pdstools.adm.ADMDatamart` instance.
    """

    dependencies = ["plotly"]
    dependency_group = "adm"

    def __init__(self, datamart: "ADMDatamart"):
        self.datamart = datamart
        super().__init__()
