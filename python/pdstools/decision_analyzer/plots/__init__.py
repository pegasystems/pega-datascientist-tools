"""Decision-analyzer plotting helpers.

This package preserves the public surface of the previous
``decision_analyzer.plots`` module while splitting the implementation
across several focused private submodules:

* ``_sensitivity`` — sensitivity, threshold deciles, prio-factor boxplots.
* ``_winloss`` — global win/loss distribution and win-distribution bar chart.
* ``_optionality`` — propensity-vs-optionality, optionality per stage / trend.
* ``_funnel`` — optionality funnel, decision funnel, decisions-without-actions.
* ``_distribution`` — treemap, action variation, histograms, rank/parameter boxplots.
* ``_components`` — filter-component plots and prioritization-component distributions.
* ``_trend`` — generic trend chart.
* ``_offer_quality`` — offer-quality pie charts and trend.

Submodule names are underscore-prefixed; only this ``__init__`` is the
supported import surface. Imports such as
``from pdstools.decision_analyzer.plots import Plot`` continue to resolve
unchanged.
"""

# ruff: noqa: F401
# Re-export the module-level imports that the legacy single-file module
# exposed via ``dir(plots)``. Some downstream code relied on these
# being addressable as attributes, so we keep them here verbatim.
from typing import cast

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ...utils.pega_template import colorway
from ...utils.plot_utils import simplify_facet_titles
from ..utils import PRIO_FACTORS, apply_filter
from ._common import DEFAULT_BOXPLOT_POINT_CAP, _boxplot_point_cap

# Plot-class method implementations live in private submodules and are
# imported under underscore-prefixed aliases so they don't leak into the
# package's public ``dir()`` surface; they are bound onto :class:`Plot`
# below.
from ._components import component_action_impact as _component_action_impact
from ._components import component_drilldown as _component_drilldown
from ._components import filtering_components as _filtering_components
from ._components import (
    plot_component_overview,
    plot_priority_component_distribution,
)
from ._distribution import action_variation as _action_variation
from ._distribution import (
    create_parameter_distribution_boxplots,
)
from ._distribution import distribution as _distribution
from ._distribution import distribution_as_treemap as _distribution_as_treemap
from ._distribution import rank_boxplot as _rank_boxplot
from ._funnel import decision_funnel as _decision_funnel
from ._funnel import decisions_without_actions_plot as _decisions_without_actions_plot
from ._funnel import optionality_funnel as _optionality_funnel
from ._offer_quality import (
    getTrendChart,
    offer_quality_piecharts,
    offer_quality_single_pie,
)
from ._optionality import optionality_per_stage as _optionality_per_stage
from ._optionality import optionality_trend as _optionality_trend
from ._optionality import propensity_vs_optionality as _propensity_vs_optionality
from ._sensitivity import prio_factor_boxplots as _prio_factor_boxplots
from ._sensitivity import sensitivity as _sensitivity
from ._sensitivity import threshold_deciles as _threshold_deciles
from ._trend import trend_chart as _trend_chart
from ._winloss import (
    create_win_distribution_plot,
)
from ._winloss import global_winloss_distribution as _global_winloss_distribution


class Plot:
    """Plotting facade attached to a :class:`DecisionAnalyzer` instance.

    Method implementations live in the underscore-prefixed submodules
    (``_sensitivity``, ``_funnel``, …) and are bound onto this class
    below so the public surface and call sites remain unchanged.
    """

    def __init__(self, decision_data):
        self._decision_data = decision_data

    _boxplot_point_cap = _boxplot_point_cap

    # _sensitivity
    threshold_deciles = _threshold_deciles
    sensitivity = _sensitivity
    prio_factor_boxplots = _prio_factor_boxplots

    # _winloss
    global_winloss_distribution = _global_winloss_distribution

    # _optionality
    propensity_vs_optionality = _propensity_vs_optionality
    optionality_per_stage = _optionality_per_stage
    optionality_trend = _optionality_trend

    # _funnel
    optionality_funnel = _optionality_funnel
    decision_funnel = _decision_funnel
    decisions_without_actions_plot = _decisions_without_actions_plot

    # _distribution
    distribution_as_treemap = _distribution_as_treemap
    action_variation = _action_variation
    distribution = _distribution
    rank_boxplot = _rank_boxplot

    # _components
    filtering_components = _filtering_components
    component_action_impact = _component_action_impact
    component_drilldown = _component_drilldown

    # _trend
    trend_chart = _trend_chart


__all__ = [
    "DEFAULT_BOXPLOT_POINT_CAP",
    "PRIO_FACTORS",
    "Plot",
    "apply_filter",
    "colorway",
    "create_parameter_distribution_boxplots",
    "create_win_distribution_plot",
    "getTrendChart",
    "offer_quality_piecharts",
    "offer_quality_single_pie",
    "plot_component_overview",
    "plot_priority_component_distribution",
    "simplify_facet_titles",
]
