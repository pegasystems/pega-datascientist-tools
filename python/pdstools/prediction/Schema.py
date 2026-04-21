"""Schema definitions for Prediction Studio data.

Mirrors the layout of :mod:`pdstools.adm.Schema` — each class is a
plain attribute container mapping column names to their target
Polars dtypes. Used by :class:`pdstools.prediction.Prediction`'s
internal validators to coerce raw input into a consistent shape.
"""

from __future__ import annotations

import polars as pl


class PredictionSnapshot:
    """Raw column dtypes for the ``PR_DATA_DM_SNAPSHOTS`` Pega export.

    These are the columns produced by Pega's Prediction Studio dataset
    export (or equivalent PDC payload) before any pdstools renaming or
    pivoting is applied.
    """

    pySnapShotTime = pl.Utf8
    pyModelId = pl.Utf8
    pyModelType = pl.Categorical
    pySnapshotType = pl.Categorical
    pyDataUsage = pl.Categorical
    pyPositives = pl.Float64
    pyNegatives = pl.Float64
    pyCount = pl.Float64
    pyValue = pl.Float64
    pyName = pl.Utf8


class PredictionData:
    """Processed/output column dtypes for ``Prediction.predictions``.

    The shape produced after :class:`Prediction`'s validator pivots
    raw rows into per-(``pyModelId``, ``SnapshotTime``) records with
    Test/Control/NBA suffixes.
    """

    pyModelId = pl.Utf8
    SnapshotTime = pl.Date
    Class = pl.Utf8
    ModelName = pl.Utf8
    Performance = pl.Float32
    Positives = pl.Float32
    Negatives = pl.Float32
    ResponseCount = pl.Float32
    Positives_Test = pl.Float32
    Positives_Control = pl.Float32
    Positives_NBA = pl.Float32
    Negatives_Test = pl.Float32
    Negatives_Control = pl.Float32
    Negatives_NBA = pl.Float32
    ResponseCount_Test = pl.Float32
    ResponseCount_Control = pl.Float32
    ResponseCount_NBA = pl.Float32
    CTR = pl.Float64
    CTR_Test = pl.Float64
    CTR_Control = pl.Float64
    CTR_NBA = pl.Float64
    CTR_Lift = pl.Float64
    isValidPrediction = pl.Boolean
