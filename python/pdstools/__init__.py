"""Pega Data Scientist Tools Python library"""

__version__ = "3.5.2"

from polars import enable_string_cache

enable_string_cache()

import sys
from pathlib import Path

from .adm.ADMDatamart import ADMDatamart
from .adm.ADMTrees import ADMTrees, MultiTrees
from .adm.BinAggregator import BinAggregator
from .decision_analyzer import DecisionData
from .pega_io import API, S3, Anonymization, File, get_token, readDSExport
from .pega_io.API import setupAzureOpenAI
from .prediction import Prediction
from .utils import NBAD, cdh_utils, datasets, errors
from .utils.cdh_utils import defaultPredictorCategorization
from .utils.CDHLimits import CDHLimits
from .utils.datasets import CDHSample, SampleTrees, SampleValueFinder
from .utils.polars_ext import *
from .utils.show_versions import show_versions
from .utils.table_definitions import PegaDefaultTables
from .valuefinder.ValueFinder import ValueFinder

if "streamlit" in sys.modules:
    from .utils import streamlit_utils as streamlit_utils

__reports__ = Path(__file__).parents[0] / "reports"

__all__ = [
    "ADMDatamart",
    "ADMTrees",
    "MultiTrees",
    "BinAggregator",
    "DecisionData",
    "API",
    "S3",
    "Anonymization",
    "File",
    "get_token",
    "readDSExport",
    "setupAzureOpenAI",
    "Prediction",
    "NBAD",
    "cdh_utils",
    "datasets",
    "errors",
    "defaultPredictorCategorization",
    "CDHLimits",
    "CDHSample",
    "SampleTrees",
    "SampleValueFinder",
    "show_versions",
    "PegaDefaultTables",
    "ValueFinder",
]
