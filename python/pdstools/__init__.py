"""Pega Data Scientist Tools Python library"""

__version__ = "4.0.0-alpha"

from polars import enable_string_cache

enable_string_cache()

import sys
from pathlib import Path

from .adm.ADMDatamart import ADMDatamart

# from .utils.polars_ext import *
# from .utils.show_versions import show_versions
# from .valuefinder.ValueFinder import ValueFinder
from .infinity import Infinity
from .utils import datasets

# from .adm.ADMTrees import ADMTrees, MultiTrees
# from .decision_analyzer import DecisionData
# from .pega_io import API, S3, Anonymization, File, get_token, read_ds_export
# from .prediction import Prediction
# from .utils import cdh_utils, datasets, errors
# from .utils.cdh_utils import default_predictor_categorization
from .utils.datasets import cdh_sample, sample_trees, sample_value_finder

if "streamlit" in sys.modules:
    from .utils import streamlit_utils

__reports__ = Path(__file__).parents[0] / "reports"

__all__ = [
    "API",
    "S3",
    "ADMDatamart",
    "ADMTrees",
    "MultiTrees",
    "BinAggregator",
    "Anonymization",
    "File",
    "get_token",
    "read_ds_export",
    "setupAzureOpenAI",
    "Prediction",
    "NBAD",
    "cdh_utils",
    "datasets",
    "errors",
    "default_predictor_categorization",
    "CDHLimits",
    "cdh_sample",
    "sample_trees",
    "sample_value_finder",
    "show_versions",
    "PegaDefaultTables",
    "ValueFinder",
    "streamlit_utils",
    "pega_template",
    "Infinity",
]
