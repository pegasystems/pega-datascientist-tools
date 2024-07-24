"""Python pdstools"""

__version__ = "3.4.6"

from polars import enable_string_cache

enable_string_cache()

import sys
from pathlib import Path

from .adm.ADMDatamart import ADMDatamart
from .adm.ADMTrees import ADMTrees, MultiTrees
from .adm.BinAggregator import BinAggregator
from .pega_io import get_token, readDSExport
from .pega_io import File, API, S3
from .pega_io.API import setupAzureOpenAI
from .prediction import Prediction
from .utils import cdh_utils, datasets, errors, hds_utils, NBAD
from .utils.CDHLimits import CDHLimits
from .utils.cdh_utils import defaultPredictorCategorization
from .utils.show_versions import show_versions
from .utils.datasets import CDHSample, SampleTrees, SampleValueFinder
from .utils.hds_utils import Config, DataAnonymization
from .utils.table_definitions import PegaDefaultTables
from .valuefinder.ValueFinder import ValueFinder
from .utils.polars_ext import *
from .decision_analyzer import DecisionData


if "streamlit" in sys.modules:
    from .utils import streamlit_utils

__reports__ = Path(__file__).parents[0] / "reports"
