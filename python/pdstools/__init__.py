"""Pega Data Scientist Tools Python library"""

__version__ = "4.0.0a1"


import sys
from pathlib import Path

from .adm.ADMDatamart import ADMDatamart
from .adm.ADMTrees import ADMTrees, MultiTrees
from .adm.BinAggregator import BinAggregator
from .infinity import Infinity

# from .decision_analyzer import DecisionData
# from .pega_io import API, S3, Anonymization, File, get_token, read_ds_export
# from .prediction import Prediction
# from .utils import cdh_utils, datasets, errors
from .pega_io import Anonymization, read_ds_export
from .prediction import Prediction
from .utils import cdh_utils, datasets
from .utils.cdh_utils import default_predictor_categorization
from .utils.datasets import cdh_sample, sample_trees, sample_value_finder

# from .utils.polars_ext import *
from .utils.show_versions import show_versions
from .valuefinder.ValueFinder import ValueFinder

if "streamlit" in sys.modules:
    from .utils import streamlit_utils

# if "quarto" in sys.modules:
#     from .utils import report_utils

from polars import enable_string_cache

enable_string_cache()

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
    "CDHGuidelines",
    "cdh_utils",
    "datasets",
    "errors",
    "default_predictor_categorization",
    "CDHLimits",
    "cdh_sample",
    "sample_trees",
    "__version__",
    "sample_value_finder",
    "show_versions",
    "PegaDefaultTables",
    "ValueFinder",
    "streamlit_utils",
    "report_utils",
    "pega_template",
    "Infinity",
]
