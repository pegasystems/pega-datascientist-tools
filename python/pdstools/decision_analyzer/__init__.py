"""Decision Analyzer module.

Provides :class:`DecisionAnalyzer` for analysing Pega NBA decision data
(Explainability Extract v1 or Decision Analyzer / EEV2 v2 exports).

Optional dependencies
---------------------
The :mod:`pdstools.decision_analyzer.plots` submodule requires ``plotly``
(install with ``pdstools[adm]``). The aggregation and scoring APIs work
without it; accessing :attr:`DecisionAnalyzer.plot` on a system without
plotly raises a friendly
:class:`~pdstools.utils.namespaces.MissingDependenciesException`.
"""

from __future__ import annotations

from . import utils
from .DecisionAnalyzer import DecisionAnalyzer

__all__ = ["DecisionAnalyzer", "plots", "utils"]


def __getattr__(name):  # pragma: no cover - thin lazy-import shim
    # Lazily expose the ``plots`` submodule so that simply importing
    # ``pdstools.decision_analyzer`` doesn't pull in plotly.
    if name == "plots":
        from . import plots as _plots

        return _plots
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
