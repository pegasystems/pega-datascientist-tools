"""Backward-compatibility shim for ``pdstools.adm.ADMTrees``.

The implementation moved to the :mod:`pdstools.adm.trees` package after a
1864-LOC monolith was split by concern (parsing / nodes / predict /
plots / datamart / public).  This module re-exports every public name so
existing imports — ``from pdstools.adm.ADMTrees import ADMTrees,
ADMTreesModel, MultiTrees, AGB, Split, Node, parse_split`` — keep
working unchanged.
"""

from __future__ import annotations

from .trees import (
    AGB,
    ADMTrees,
    ADMTreesModel,
    MultiTrees,
    Node,
    Split,
    SplitOperator,
    parse_split,
)

# Private helpers re-exported for back-compat with code that imported them
# from this module (e.g. internal callers in datasets.py / tests).
from .trees._nodes import (  # noqa: F401
    _BOOSTER_PATHS,
    _iter_nodes,
    _traverse,
)

__all__ = [
    "AGB",
    "ADMTrees",
    "ADMTreesModel",
    "MultiTrees",
    "Node",
    "Split",
    "SplitOperator",
    "parse_split",
]
