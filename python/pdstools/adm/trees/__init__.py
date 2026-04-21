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
``ADMTrees(file, ...)`` polymorphic factory is still exported for
backward compatibility but is deprecated.
"""

from __future__ import annotations

import warnings

import polars as pl

from ._agb import AGB
from ._model import ADMTreesModel
from ._multi import MultiTrees
from ._nodes import Node, Split, SplitOperator, parse_split

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


class ADMTrees:  # pragma: no cover
    """Deprecated polymorphic factory.

    Retained for backward compatibility with code that calls
    ``ADMTrees(file)`` and expects either an :class:`ADMTreesModel` or a
    ``dict[str, MultiTrees]`` back.  New code should call the explicit
    factory classmethods directly:

    - :meth:`ADMTreesModel.from_file`
    - :meth:`ADMTreesModel.from_url`
    - :meth:`ADMTreesModel.from_datamart_blob`
    - :meth:`ADMTreesModel.from_dict`
    - :meth:`MultiTrees.from_datamart`
    """

    def __new__(cls, file, n_threads: int = 6, **kwargs):
        warnings.warn(
            "ADMTrees(...) is deprecated; use ADMTreesModel.from_file / "
            "from_url / from_datamart_blob / from_dict, or "
            "MultiTrees.from_datamart / from_datamart_grouped for "
            "datamart DataFrames.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(file, pl.DataFrame):
            file = file.filter(pl.col("Modeldata").is_not_null())
            if len(file) > 1:
                return MultiTrees.from_datamart_grouped(file, n_threads=n_threads, **kwargs)
            return ADMTreesModel.from_datamart_blob(
                file.select("Modeldata").item(),
                **kwargs,
            )
        if isinstance(file, pl.Series):
            return ADMTreesModel.from_datamart_blob(file.item(), **kwargs)
        return ADMTreesModel._from_anything(file, **kwargs)
