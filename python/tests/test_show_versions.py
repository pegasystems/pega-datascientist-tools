"""
Testing the functionality of utils/show_versions functions
"""

import datetime
import pathlib
import sys

import numpy as np
import polars as pl
import pytest
from pytz import timezone

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools import cdh_utils, datasets


