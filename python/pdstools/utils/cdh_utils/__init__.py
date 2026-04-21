"""Helpers for working with Pega CDH-style data.

This package preserves the public surface of the previous ``cdh_utils`` module
while splitting the implementation across several focused private submodules:

* ``_dates`` — Pega date-time parsing and start/end-date resolution.
* ``_namespacing`` — Pega field-name normalisation (``_capitalize``) and
  predictor-categorisation defaults.
* ``_polars`` — Polars expression / frame helpers (queries, sampling,
  schema casting, list-overlap utilities, weighted averages).
* ``_metrics`` — Performance metrics: AUC, lift, log-odds, gains tables and
  feature importance.
* ``_io`` — File, temp-directory, logger setup and version-check helpers.
* ``_misc`` — Small standalone helpers (list flattening, plot legend colors).

Submodule names are underscore-prefixed; only this ``__init__`` is the
supported import surface. Imports such as
``from pdstools.utils.cdh_utils import safe_int`` continue to resolve
unchanged.
"""

# ruff: noqa: F401
# Re-export the module-level imports that the legacy single-file module
# exposed via ``dir(cdh_utils)``. Some downstream code relied on these
# being addressable as attributes, so we keep them here verbatim.
import datetime
import io
import logging
import math
import re
import tempfile
import zipfile
from collections.abc import Iterable, Sequence
from functools import partial
from io import StringIO
from operator import is_not
from os import PathLike
from pathlib import Path
from typing import TypeVar, overload

import polars as pl
from polars._typing import PolarsTemporalType

from ..types import QUERY
from ._common import F, logger
from ._dates import (
    _get_start_end_date_args,
    from_prpc_date_time,
    parse_pega_date_time_formats,
    to_prpc_date_time,
)
from ._io import (
    _read_pdc,
    create_working_and_temp_dir,
    get_latest_pdstools_version,
    process_files_to_bytes,
    setup_logger,
)
from ._metrics import (
    auc_from_bincounts,
    auc_from_probs,
    auc_to_gini,
    aucpr_from_bincounts,
    aucpr_from_probs,
    bin_log_odds,
    feature_importance,
    gains_table,
    lift,
    log_odds_polars,
    safe_range_auc,
    z_ratio,
)
from ._misc import legend_color_order, safe_flatten_list
from ._namespacing import _capitalize, default_predictor_categorization
from ._polars import (
    POLARS_DURATION_PATTERN,
    _apply_query,
    _apply_schema_types,
    _combine_queries,
    _extract_keys,
    _polars_capitalize,
    is_valid_polars_duration,
    lazy_sample,
    overlap_lists_polars,
    overlap_matrix,
    weighted_average_polars,
    weighted_performance_polars,
)

__all__ = [
    # constants / shared types
    "F",
    "POLARS_DURATION_PATTERN",
    "QUERY",
    "logger",
    # date / time
    "from_prpc_date_time",
    "parse_pega_date_time_formats",
    "to_prpc_date_time",
    # naming / categorisation
    "default_predictor_categorization",
    # polars helpers
    "is_valid_polars_duration",
    "lazy_sample",
    "overlap_lists_polars",
    "overlap_matrix",
    "weighted_average_polars",
    "weighted_performance_polars",
    # metrics
    "auc_from_bincounts",
    "auc_from_probs",
    "auc_to_gini",
    "aucpr_from_bincounts",
    "aucpr_from_probs",
    "bin_log_odds",
    "feature_importance",
    "gains_table",
    "lift",
    "log_odds_polars",
    "safe_range_auc",
    "z_ratio",
    # I/O / misc
    "create_working_and_temp_dir",
    "get_latest_pdstools_version",
    "legend_color_order",
    "process_files_to_bytes",
    "safe_flatten_list",
    "setup_logger",
]
