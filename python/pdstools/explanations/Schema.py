"""Polars schemas for explanations input parquet files and aggregate outputs.

Mirrors the pattern used by ``pdstools.adm.Schema``: each class is a
collection of class-level attributes naming the expected columns and
their polars dtypes. Apply with ``cdh_utils._apply_schema_types``.

The raw explanation parquet schema is the public contract between Pega
and the Explanations module. Validating against it up front (in
``Preprocess.generate``) turns malformed inputs into a clear
``ValueError`` instead of a cryptic DuckDB error mid-processing.
"""

from __future__ import annotations

import polars as pl


class RawExplanationData:
    """Schema for a single explanation parquet file produced by Pega.

    Each row is one (sample, predictor) shap-coefficient observation.
    Context columns (``pyChannel``, ``pyDirection``, ``pyIssue``,
    ``pyGroup``, ``pyName``, ``pyTreatment``) are user-configurable and
    not all of them are required to be present, so they are not part of
    the strict required-columns check. The ``partition`` column
    (JSON-encoded context dict) is required because every downstream
    SQL aggregation groups by it.
    """

    pySubjectID = pl.Utf8
    pyInteractionID = pl.Utf8
    predictor_name = pl.Utf8
    predictor_type = pl.Utf8
    symbolic_value = pl.Utf8
    numeric_value = pl.Float64
    shap_coeff = pl.Float64
    score = pl.Float64
    partition = pl.Utf8


REQUIRED_RAW_COLUMNS: tuple[str, ...] = (
    "pyInteractionID",
    "predictor_name",
    "predictor_type",
    "shap_coeff",
    "partition",
)
"""Columns that must be present in every raw explanation parquet file.

``symbolic_value`` and ``numeric_value`` are technically optional per
row (one is null depending on ``predictor_type``), but at least one
must exist as a column or the SQL queries fail. We check this
separately in ``_validate_raw_data``.
"""


class ContextualAggregate:
    """Schema for the per-context aggregate parquet (``*_BATCH_*.parquet``).

    Produced by ``Preprocess._parquet_in_batches`` from
    ``resources/queries/numeric.sql`` or ``symbolic.sql``.
    """

    partition = pl.Utf8
    predictor_name = pl.Utf8
    predictor_type = pl.Utf8
    bin_contents = pl.Utf8
    bin_order = pl.Int64
    contribution_abs = pl.Float64
    contribution = pl.Float64
    contribution_min = pl.Float64
    contribution_max = pl.Float64
    frequency = pl.Int64


class OverallAggregate:
    """Schema for the per-model aggregate parquet (``*_OVERALL.parquet``).

    Same shape as ``ContextualAggregate`` but ``partition`` is always the
    literal string ``'whole_model'``.
    """

    partition = pl.Utf8
    predictor_name = pl.Utf8
    predictor_type = pl.Utf8
    bin_contents = pl.Utf8
    bin_order = pl.Int64
    contribution_abs = pl.Float64
    contribution = pl.Float64
    contribution_min = pl.Float64
    contribution_max = pl.Float64
    frequency = pl.Int64
