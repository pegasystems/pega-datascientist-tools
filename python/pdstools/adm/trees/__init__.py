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
``from_dict``, and ``MultiTrees.from_datamart``).  The legacy
``ADMTrees(file, ...)`` polymorphic factory was removed in v5; see
``docs/migration-v4-to-v5.md``.
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
