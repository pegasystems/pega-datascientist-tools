"""Pega Data Scientist Tools Python library"""

__version__ = "3.4.8-beta"

from polars import enable_string_cache

enable_string_cache()

import sys
from pathlib import Path

from .adm.ADMDatamart import ADMDatamart as ADMDatamart
from .adm.ADMTrees import ADMTrees as ADMTrees
from .adm.ADMTrees import MultiTrees as MultiTrees
from .adm.BinAggregator import BinAggregator as BinAggregator
from .decision_analyzer import DecisionData as DecisionData
from .pega_io import API as API
from .pega_io import S3 as S3
from .pega_io import Anonymization as Anonymization
from .pega_io import File as File
from .pega_io import get_token as get_token
from .pega_io import readDSExport as readDSExport
from .pega_io.API import setupAzureOpenAI as setupAzureOpenAI
from .prediction import Prediction as Prediction
from .utils import NBAD as NBAD
from .utils import cdh_utils as cdh_utils
from .utils import datasets as datasets
from .utils import errors as errors
from .utils.cdh_utils import (
    default_predictor_categorization as default_predictor_categorization,
)
from .utils.CDHLimits import CDHLimits as CDHLimits
from .utils.datasets import CDHSample as CDHSample
from .utils.datasets import SampleTrees as SampleTrees
from .utils.datasets import SampleValueFinder as SampleValueFinder
from .utils.polars_ext import *
from .utils.show_versions import show_versions as show_versions
from .utils.table_definitions import PegaDefaultTables as PegaDefaultTables
from .valuefinder.ValueFinder import ValueFinder as ValueFinder

if "streamlit" in sys.modules:
    from .utils import streamlit_utils as streamlit_utils

__reports__ = Path(__file__).parents[0] / "reports"
