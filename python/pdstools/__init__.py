"""Pega Data Scientist Tools Python library"""

__version__ = "4.0.1"

from pathlib import Path

from polars import enable_string_cache

from .adm.ADMDatamart import ADMDatamart
from .infinity import Infinity
from .pega_io import Anonymization, read_ds_export
from .prediction.Prediction import Prediction
from .utils import datasets
from .utils.cdh_utils import default_predictor_categorization
from .utils.datasets import cdh_sample, sample_value_finder
from .utils.show_versions import show_versions
from .valuefinder.ValueFinder import ValueFinder

enable_string_cache()


__reports__ = Path(__file__).parents[0] / "reports"

__all__ = [
    "ADMDatamart",
    "Anonymization",
    "read_ds_export",
    "Prediction",
    "datasets",
    "default_predictor_categorization",
    "cdh_sample",
    "sample_value_finder",
    "show_versions",
    "ValueFinder",
    "Infinity",
]
