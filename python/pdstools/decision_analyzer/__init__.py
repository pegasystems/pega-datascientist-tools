"""Decision Analyzer module.

Provides :class:`DecisionAnalyzer` for analysing Pega NBA decision data
(Explainability Extract v1 or Decision Analyzer / EEV2 v2 exports).

Functional API
--------------
For quick scripted/notebook use without instantiating the full class,
the following module-level helpers wrap a :class:`DecisionAnalyzer` and
return plain (Lazy)Frames:

* :func:`compute_funnel` — three-frame action-funnel summary.
* :func:`compute_optionality` — per-stage optionality (avg # remaining offers).
* :func:`compute_overview_stats` — headline KPIs (decisions, customers, …).

Quick start
-----------
>>> import polars as pl
>>> from pdstools.decision_analyzer import compute_funnel
>>> lf = pl.scan_parquet("path/to/extract/**/*.parquet")
>>> available, passing, filtered = compute_funnel(lf, scope="Action")

Optional dependencies
---------------------
The :mod:`pdstools.decision_analyzer.plots` submodule requires ``plotly``
(install with ``pip install 'pdstools[explanations]'``). The aggregation
and scoring APIs work without it; importing the ``plots`` module or
accessing :attr:`DecisionAnalyzer.plot` will raise a friendly
``MissingDependenciesException`` if plotly is not present.
"""

from __future__ import annotations

from . import utils
from .DecisionAnalyzer import DecisionAnalyzer
from ._functional import compute_funnel, compute_optionality, compute_overview_stats

__all__ = [
    "DecisionAnalyzer",
    "compute_funnel",
    "compute_optionality",
    "compute_overview_stats",
    "utils",
]


def __getattr__(name):  # pragma: no cover - thin lazy-import shim
    # Lazily expose the ``plots`` submodule so that simply importing
    # ``pdstools.decision_analyzer`` doesn't pull in plotly.
    if name == "plots":
        from . import plots as _plots

        return _plots
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
