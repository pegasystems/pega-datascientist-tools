"""Python pdstools"""

__version__ = "3.0.1"

from polars.polars import toggle_string_cache

toggle_string_cache(True)

import sys
from pathlib import Path

from .adm.ADMDatamart import ADMDatamart
from .adm.ADMTrees import ADMTrees, MultiTrees
from .pega_io import getToken, readClientCredentialFile, readDSExport
from .pega_io import File, API, S3
from .utils import cdh_utils, datasets, errors, hds_utils
from .utils.cdh_utils import defaultPredictorCategorization
from .utils.datasets import CDHSample, SampleTrees, SampleValueFinder
from .utils.hds_utils import Config, DataAnonymization
from .valuefinder.ValueFinder import ValueFinder

if "streamlit" in sys.modules:
    from .utils import streamlit_utils

__reports__ = Path(__file__).parents[0] / "reports"
