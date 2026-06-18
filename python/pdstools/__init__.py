"""Pega Data Scientist Tools Python library"""

from __future__ import annotations

__version__ = "5.0.0"

from pathlib import Path
from typing import TYPE_CHECKING

from .adm.ADMDatamart import ADMDatamart
from .ih.IH import IH
from .impactanalyzer.ImpactAnalyzer import ImpactAnalyzer
from .pega_io import Anonymization, read_ds_export
from .prediction.Prediction import Prediction
from .utils import datasets
from .utils.cdh_utils import default_predictor_categorization
from .utils.datasets import cdh_sample, sample_value_finder
from .utils.show_versions import show_versions
from .valuefinder.ValueFinder import ValueFinder

if TYPE_CHECKING:
    from .infinity.client import AsyncInfinity, Infinity

__reports__ = Path(__file__).parents[0] / "reports"

__all__ = [
    "IH",
    "ADMDatamart",
    "Anonymization",
    "AsyncInfinity",
    "ImpactAnalyzer",
    "Infinity",
    "Prediction",
    "ValueFinder",
    "cdh_sample",
    "datasets",
    "default_predictor_categorization",
    "read_ds_export",
    "sample_value_finder",
    "show_versions",
]


def __getattr__(name: str):
    if name in {"Infinity", "AsyncInfinity"}:
        from .infinity import AsyncInfinity, Infinity

        return {"Infinity": Infinity, "AsyncInfinity": AsyncInfinity}[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
