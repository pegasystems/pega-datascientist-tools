"""Helpers for generating Quarto-based pdstools reports.

This package preserves the public surface of the previous ``report_utils``
module while splitting the implementation across several focused private
submodules:

* ``_common`` — shared module-level state (logger).
* ``_filenames`` — output filename composition and report-resource copying.
* ``_quarto`` — Quarto execution: rendering, command-line wrappers,
  Quarto/Pandoc version detection, callouts, and the ``show_credits`` block.
* ``_html`` — HTML post-processing: CSS inlining, zip bundling of Quarto
  resource folders, and report error scanning.
* ``_tables`` — RAG-coloured metric tables (``itables`` and ``great_tables``).
* ``_polars_helpers`` — small Polars helpers and aggregations used by
  Quarto report templates (gains table, hierarchy summarisations, etc.).
* ``_query`` — serialise/deserialise pdstools ``QUERY`` objects so they can
  travel through Quarto YAML parameters.

Submodule names are underscore-prefixed; only this ``__init__`` is the
supported import surface. Imports such as
``from pdstools.utils.report_utils import quarto_render_to_html`` continue
to resolve unchanged.
"""

# ruff: noqa: F401
# Re-export the module-level imports that the legacy single-file module
# exposed via ``dir(report_utils)``. Some downstream code (and Quarto
# templates) relied on these being addressable as attributes, so we keep
# them here verbatim.
import datetime
import html
import io
import json
import logging
import os
import re
import shutil
import subprocess
import traceback
import zipfile
from pathlib import Path
from typing import Any

import polars as pl

from ..types import QUERY

# Re-export RAG functions from metric_limits for convenience in Quarto reports.
from ..metric_limits import (
    MetricFormats,
    MetricLimits,
    exclusive_0_1_range_rag,
    positive_values,
    standard_NBAD_channels_rag,
    standard_NBAD_configurations_rag,
    standard_NBAD_directions_rag,
    standard_NBAD_predictions_rag,
    strict_positive_values,
)

# Re-export NumberFormat for external use
from ..number_format import NumberFormat

from ._common import logger
from ._filenames import (
    copy_quarto_file,
    copy_report_resources,
    get_output_filename,
)
from ._html import (
    _inline_css,
    bundle_quarto_resources,
    check_report_for_errors,
    generate_zipped_report,
)
from ._polars_helpers import (
    avg_by_hierarchy,
    gains_table,
    max_by_hierarchy,
    n_unique_values,
    polars_col_exists,
    polars_subset_to_existing_cols,
    sample_values,
)
from ._quarto import (
    _get_cmd_output,
    _get_version_only,
    _set_command_options,
    _write_params_files,
    get_pandoc_with_version,
    get_quarto_with_version,
    quarto_callout_important,
    quarto_callout_info,
    quarto_callout_no_prediction_data_warning,
    quarto_callout_no_predictor_data_warning,
    quarto_plot_exception,
    quarto_print,
    run_quarto,
    show_credits,
)
from ._query import deserialize_query, serialize_query
from ._tables import create_metric_gttable, create_metric_itable

__all__ = [
    # shared types / state
    "QUERY",
    "logger",
    # re-exports from sibling modules
    "MetricFormats",
    "MetricLimits",
    "NumberFormat",
    "exclusive_0_1_range_rag",
    "positive_values",
    "standard_NBAD_channels_rag",
    "standard_NBAD_configurations_rag",
    "standard_NBAD_directions_rag",
    "standard_NBAD_predictions_rag",
    "strict_positive_values",
    # filenames / resource copying
    "copy_quarto_file",
    "copy_report_resources",
    "get_output_filename",
    # quarto execution / callouts / credits
    "get_pandoc_with_version",
    "get_quarto_with_version",
    "quarto_callout_important",
    "quarto_callout_info",
    "quarto_callout_no_prediction_data_warning",
    "quarto_callout_no_predictor_data_warning",
    "quarto_plot_exception",
    "quarto_print",
    "run_quarto",
    "show_credits",
    # html post-processing
    "bundle_quarto_resources",
    "check_report_for_errors",
    "generate_zipped_report",
    # rag-coloured metric tables
    "create_metric_gttable",
    "create_metric_itable",
    # polars helpers / aggregations
    "avg_by_hierarchy",
    "gains_table",
    "max_by_hierarchy",
    "n_unique_values",
    "polars_col_exists",
    "polars_subset_to_existing_cols",
    "sample_values",
    # query serialisation
    "deserialize_query",
    "serialize_query",
]
