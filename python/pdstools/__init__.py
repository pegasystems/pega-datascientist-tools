"""Python pdstools."""

__version__ = "3.0.0"
from polars.polars import toggle_string_cache

toggle_string_cache(True)

from .adm.ADMDatamart import ADMDatamart
from .adm.ADMTrees import ADMTrees, MultiTrees
from .valuefinder.ValueFinder import ValueFinder
from .utils import cdh_utils
from .utils import datasets
from .utils import errors
