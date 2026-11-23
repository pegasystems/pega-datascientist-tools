"""ADM Gradient Boosting (AGB) model parsing, scoring, and diagnostics.

This package provides:

- :class:`Split` and :class:`Node` — small dataclasses describing a parsed
  split condition and a tree node.
- :class:`ADMTreesModel` — load and analyse a single AGB model.
- :class:`MultiTrees` — collection of snapshots of the same configuration
  over time.
- :class:`AGB` — Datamart helper for discovering and extracting AGB models.

Construction uses explicit factory classmethods
(``ADMTreesModel.from_file``, ``from_url``, ``from_datamart_blob``,
``from_dict``, and ``MultiTrees.from_datamart``). Every loader accepts
both decoded and encoded AGB model JSON; encoded splits are decoded
automatically using the model's ``inputsEncoder``.  The legacy
``ADMTrees(file, ...)`` polymorphic factory was removed in v5; see
``docs/migration-v4-to-v5.md``.

Where to start for common AGB analyses:

- Tree walking, depth, split counts and gains: ``tree_stats``,
  ``gains_per_split`` and ``grouped_gains_per_split``.
- Model-level summaries: ``metrics`` (see ``metric_descriptions()``) and
  ``predictor_diagnostics()``.
- Split values and context keys: ``all_values_per_split`` and
  ``context_key_counts``.
- Gain per predictor and per predictor category: ``plot.feature_importance_by_gain``
  and ``plot.gain_by_namespace``. To use your own categories, set
  ``plot.predictor_category_expr``, or pass a callable to
  ``compute_categorization_over_time``.
- Predictor roles and gain over the boosting sequence:
  ``plot.feature_role_map``, ``plot.early_vs_late_gain`` and
  ``plot.gain_decay_dual_lens``.
- Snapshots over time: :class:`MultiTrees`.

Every ``plot`` method accepts ``return_df=True`` to return the data behind the chart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._agb import AGB
from ._model import ADMTreesModel
from ._multi import MultiTrees
from ._nodes import Node, Split, SplitOperator, parse_split

if TYPE_CHECKING:
    from ._plots import Plots

__all__ = [
    "AGB",
    "ADMTreesModel",
    "MultiTrees",
    "Node",
    "Plots",
    "Split",
    "SplitOperator",
    "parse_split",
]


def __getattr__(name: str):
    if name == "Plots":
        from ._plots import Plots

        return Plots
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
