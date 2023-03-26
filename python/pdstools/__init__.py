"""Python pdstools"""

__version__ = "3.0.1"

from polars.polars import toggle_string_cache

toggle_string_cache(True)

from pathlib import Path

from .adm.ADMDatamart import ADMDatamart
from .adm.ADMTrees import ADMTrees, MultiTrees
from .utils import cdh_utils, datasets, errors, hds_utils, streamlit_utils
from .utils.datasets import CDHSample, SampleTrees, SampleValueFinder
from .utils.hds_utils import Config, DataAnonymization
from .valuefinder.ValueFinder import ValueFinder

__reports__ = Path(__file__).parents[0] / "reports"
