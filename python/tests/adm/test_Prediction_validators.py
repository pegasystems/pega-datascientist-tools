"""Exact-value unit tests for Prediction validator helpers and Schema.

These cover the small, pure helpers extracted in the gold-standard
refactor: ``_normalize_performance_scale``, ``_parse_snapshot_time``,
and ``_validate_prediction_data``. Assertions use exact values
traceable back to the inputs — no structural-only checks.
"""

from __future__ import annotations

import datetime

import polars as pl
import pytest
from pdstools import Prediction
from pdstools.prediction.Schema import PredictionData, PredictionSnapshot


def test_normalize_performance_scale_pega_range():
    """Performance values on Pega's 50-100 scale are divided by 100."""
    df = pl.LazyFrame({"Performance": [55.0, 70.0, 100.0]})
    out = Prediction._normalize_performance_scale(df).collect()
    assert out["Performance"].to_list() == pytest.approx([0.55, 0.70, 1.00])


def test_normalize_performance_scale_already_unit():
    """Performance values already in 0-1 range are left untouched."""
    df = pl.LazyFrame({"Performance": [0.55, 0.70, 1.00]})
    out = Prediction._normalize_performance_scale(df).collect()
    assert out["Performance"].to_list() == pytest.approx([0.55, 0.70, 1.00])


def test_normalize_performance_scale_missing_column():
    """Missing Performance column is a no-op (returns the frame as-is)."""
    df = pl.LazyFrame({"Other": [1, 2, 3]})
    out = Prediction._normalize_performance_scale(df).collect()
    assert out.columns == ["Other"]
    assert out["Other"].to_list() == [1, 2, 3]


def test_normalize_performance_scale_all_null():
    """All-null Performance returns frame unchanged (no scaling)."""
    df = pl.LazyFrame({"Performance": [None, None]}, schema={"Performance": pl.Float64})
    out = Prediction._normalize_performance_scale(df).collect()
    assert out["Performance"].to_list() == [None, None]


def test_parse_snapshot_time_from_pega_string():
    """A Pega string timestamp is parsed into a Date column."""
    # 20400401T000000.000 GMT — Pega format, 15-char prefix used downstream
    df = pl.LazyFrame({"pySnapShotTime": ["20400401T000000.000 GMT"]})
    out = Prediction._parse_snapshot_time(df).collect()
    assert out.schema["SnapshotTime"] == pl.Date
    assert out["SnapshotTime"].to_list() == [datetime.date(2040, 4, 1)]


def test_parse_snapshot_time_from_temporal_dtype():
    """An already-temporal pySnapShotTime is cast directly to Date."""
    df = pl.LazyFrame(
        {"pySnapShotTime": [datetime.datetime(2040, 4, 1, 12, 0)]},
    )
    out = Prediction._parse_snapshot_time(df).collect()
    assert out.schema["SnapshotTime"] == pl.Date
    assert out["SnapshotTime"].to_list() == [datetime.date(2040, 4, 1)]


def test_validate_prediction_data_pivots_test_control_nba():
    """Validator joins Control/Test/NBA usage counts onto each Daily row.

    The current implementation filters the "main" frame to
    ``pySnapshotType == "Daily"`` then joins on (pyModelId, SnapshotTime).
    With three Daily rows (Control/Test/NBA) and matching Test/Control/NBA
    counts, the cartesian-style join yields three output rows where the
    main Positives/Negatives vary but the ``_Test``/``_Control``/``_NBA``
    suffixes are identical across rows.
    """
    raw = pl.LazyFrame(
        {
            "pySnapShotTime": ["20400401T000000.000 GMT"] * 4,
            "pyModelId": ["DATA-DECISION-REQUEST!MYPRED"] * 4,
            "pyModelType": ["PREDICTION"] * 4,
            # The "" usage row carries the per-day aggregate but is tagged
            # with a NULL snapshot type so it's excluded from the main path.
            "pySnapshotType": ["Daily", "Daily", "Daily", None],
            "pyDataUsage": ["Control", "Test", "NBA", ""],
            "pyPositives": [100.0, 400.0, 500.0, 1000.0],
            "pyNegatives": [1000.0, 2000.0, 3000.0, 6000.0],
            "pyCount": [1100.0, 2400.0, 3500.0, 7000.0],
            "pyValue": [0.65, 0.65, 0.65, 0.65],
        },
    )
    out = Prediction._validate_prediction_data(raw).collect()

    assert out.height == 3
    rows = out.sort("Positives").to_dicts()

    # Main Positives/Negatives come from each Control/Test/NBA row in turn.
    assert [r["Positives"] for r in rows] == [100.0, 400.0, 500.0]
    assert [r["Negatives"] for r in rows] == [1000.0, 2000.0, 3000.0]

    # Suffix columns are the same across all three rows (the join on
    # (pyModelId, SnapshotTime) duplicates the single Test/Control/NBA row).
    for r in rows:
        assert r["Positives_Test"] == 400.0
        assert r["Negatives_Test"] == 2000.0
        assert r["Positives_Control"] == 100.0
        assert r["Negatives_Control"] == 1000.0
        assert r["Positives_NBA"] == 500.0
        assert r["Negatives_NBA"] == 3000.0
        assert r["CTR_Test"] == pytest.approx(400.0 / 2400.0)
        assert r["CTR_Control"] == pytest.approx(100.0 / 1100.0)
        assert r["CTR_Lift"] == pytest.approx(
            (r["CTR_Test"] - r["CTR_Control"]) / r["CTR_Control"],
        )
        assert r["Performance"] == pytest.approx(0.65)
        assert r["Class"] == "DATA-DECISION-REQUEST"
        assert r["ModelName"] == "MYPRED"
        assert r["isValidPrediction"] is True


def test_validate_prediction_data_filters_non_prediction_rows():
    """Rows with pyModelType != 'PREDICTION' are dropped before the pivot."""
    raw = pl.LazyFrame(
        {
            "pySnapShotTime": ["20400401T000000.000 GMT"] * 4,
            "pyModelId": ["DATA-DECISION-REQUEST!P"] * 3 + ["DATA-DECISION-REQUEST!Q"],
            "pyModelType": ["PREDICTION"] * 3 + ["ADM"],
            "pySnapshotType": ["Daily", "Daily", "Daily", "Daily"],
            "pyDataUsage": ["Control", "Test", "NBA", "Control"],
            "pyPositives": [10.0, 40.0, 50.0, 999.0],
            "pyNegatives": [100.0, 200.0, 300.0, 999.0],
            "pyCount": [110.0, 240.0, 350.0, 1998.0],
            "pyValue": [0.6] * 4,
        },
    )
    out = Prediction._validate_prediction_data(raw).collect()
    # Only the PREDICTION rows survive; the ADM model is gone entirely.
    assert set(out["pyModelId"].unique().to_list()) == {"DATA-DECISION-REQUEST!P"}


def test_validate_prediction_data_invalid_when_zero_counts():
    """isValidPrediction is False when any group has zero positives."""
    raw = pl.LazyFrame(
        {
            "pySnapShotTime": ["20400401T000000.000 GMT"] * 3,
            "pyModelId": ["DATA-DECISION-REQUEST!P"] * 3,
            "pyModelType": ["PREDICTION"] * 3,
            "pySnapshotType": ["Daily", "Daily", "Daily"],
            "pyDataUsage": ["Control", "Test", "NBA"],
            # Test group has 0 positives => invalid across all output rows.
            "pyPositives": [10.0, 0.0, 50.0],
            "pyNegatives": [100.0, 200.0, 300.0],
            "pyCount": [110.0, 200.0, 350.0],
            "pyValue": [0.6] * 3,
        },
    )
    out = Prediction._validate_prediction_data(raw).collect()
    assert out["isValidPrediction"].to_list() == [False] * out.height
    assert out.height >= 1


def test_validate_prediction_data_normalizes_high_performance():
    """Performance > 1 is normalised by /100 across the whole frame."""
    raw = pl.LazyFrame(
        {
            "pySnapShotTime": ["20400401T000000.000 GMT"] * 3,
            "pyModelId": ["DATA-DECISION-REQUEST!P"] * 3,
            "pyModelType": ["PREDICTION"] * 3,
            "pySnapshotType": ["Daily"] * 3,
            "pyDataUsage": ["Control", "Test", "NBA"],
            "pyPositives": [10.0, 40.0, 50.0],
            "pyNegatives": [100.0, 200.0, 300.0],
            "pyCount": [110.0, 240.0, 350.0],
            "pyValue": [70.0, 70.0, 70.0],
        },
    )
    out = Prediction._validate_prediction_data(raw).collect()
    assert out["Performance"].to_list() == pytest.approx([0.70] * out.height)


def test_schema_classes_have_expected_dtypes():
    """Schema modules expose dtype attributes mirroring adm/Schema.py style."""
    assert PredictionSnapshot.pyValue == pl.Float64
    assert PredictionSnapshot.pyModelType == pl.Categorical
    assert PredictionData.SnapshotTime == pl.Date
    assert PredictionData.isValidPrediction == pl.Boolean
    assert PredictionData.CTR_Lift == pl.Float64
