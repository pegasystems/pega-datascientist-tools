import time

import polars as pl
import pytest
from pdstools import Prediction
from pdstools.sample_data._prediction import PREDICTIONS, prediction_snapshots
from pdstools.sample_data._world import sample_world


@pytest.fixture(scope="module")
def raw():
    return prediction_snapshots(sample_world(end="canonical")).collect()


def test_every_day_has_daily_test_and_control_rows(raw):
    counts = (
        raw.with_columns(Usage=pl.coalesce("pyDataUsage", "pySnapshotType"))
        .group_by("pyModelId", "Usage")
        .agg(pl.col("pySnapShotTime").str.slice(0, 8).n_unique())
    )
    per_model = {(m, u): n for m, u, n in counts.iter_rows()}
    for channel, name in PREDICTIONS.items():
        model = f"DATA-DECISION-REQUEST-CUSTOMER!{name.upper()}"
        assert per_model[(model, "Daily")] == 182
        assert per_model[(model, "Test")] == 182
        assert per_model[(model, "Control")] == 182
        assert per_model.get((model, "NBA"), 0) == (0 if channel == "Call Center" else 182)


def test_daily_rows_carry_totals(raw):
    daily = raw.filter(pl.col("pySnapshotType") == "Daily")
    assert daily["pyName"].unique().to_list() == ["auc"]
    assert daily["pyValue"].min() >= 50 and daily["pyValue"].max() <= 100
    assert (daily["pyCount"] == daily["pyPositives"] + daily["pyNegatives"]).all()
    arms = (
        raw.filter(pl.col("pyDataUsage").is_not_null())
        .group_by("pyModelId", pl.col("pySnapShotTime").str.slice(0, 8))
        .agg(pl.col("pyPositives").sum())
    )
    joined = daily.with_columns(pl.col("pySnapShotTime").str.slice(0, 8)).join(
        arms, on=["pyModelId", "pySnapShotTime"], suffix="_arms"
    )
    non_call_center = joined.filter(~pl.col("pyModelId").str.contains("CALLCENTER"))
    assert (non_call_center["pyPositives"] == non_call_center["pyPositives_arms"]).all()


def test_deterministic_per_seed(raw):
    assert prediction_snapshots(sample_world(end="canonical")).collect().equals(raw)
    other = prediction_snapshots(sample_world(seed=7, end="canonical")).collect()
    assert not other.equals(raw)


def test_auc_is_noisy_around_world_curve(raw):
    world = sample_world(end="canonical")
    expected = [d.auc for d in world.channel_days("Web")]
    observed = (
        raw.filter(pl.col("pySnapshotType") == "Daily", pl.col("pyModelId").str.contains("WEB"))
        .sort("pySnapShotTime")["pyValue"]
        .to_list()
    )
    assert observed != pytest.approx(expected, abs=1e-6)
    assert sum(abs(o - e) for o, e in zip(observed, expected, strict=True)) / 182 < 6


def test_mock_data_lift_and_validity():
    summary = Prediction.from_mock_data(end="canonical").summary_by_channel().collect()
    lift = dict(zip(summary["Channel"], summary["Lift"], strict=True))
    assert lift["Web"] > lift["E-mail"] > 0
    assert lift["Mobile"] > 0
    valid = dict(zip(summary["Channel"], summary["isValid"], strict=True))
    assert not valid["Call Center"]
    assert valid["Web"] and valid["Mobile"] and valid["E-mail"] and valid["Multi-channel"]


def _raw(test_pos: int, control_pos: int) -> pl.LazyFrame:
    rows = [
        ("20240101T120000.000 GMT", "Daily", None, "auc", 70.0, 500.0, 19500.0),
        ("20240101T000000.000 GMT", None, "Test", None, None, float(test_pos), 10_000.0 - test_pos),
        ("20240101T000000.000 GMT", None, "Control", None, None, float(control_pos), 10_000.0 - control_pos),
        ("20240101T000000.000 GMT", None, "NBA", None, None, 5000.0, 1.0),
    ]
    return pl.LazyFrame(
        [
            {
                "pySnapShotTime": t,
                "pyModelId": "DATA-DECISION-REQUEST-CUSTOMER!PREDICTWEBPROPENSITY",
                "pyModelType": "PREDICTION",
                "pySnapshotType": s,
                "pyDataUsage": u,
                "pyName": n,
                "pyValue": v,
                "pyPositives": p,
                "pyNegatives": neg,
                "pyCount": p + neg,
            }
            for t, s, u, n, v, p, neg in rows
        ]
    )


def test_lift_is_test_over_control_and_ignores_nba():
    summary = Prediction(_raw(160, 100)).summary_by_channel().collect()
    assert summary["Lift"].item() == pytest.approx(0.6)


def test_zero_control_positives_give_undefined_lift():
    summary = Prediction(_raw(160, 0)).summary_by_channel().collect()
    assert not summary["isValid"].item()
    assert summary["Lift"].item() == float("inf")


def test_sixty_days_and_budget():
    start = time.perf_counter()
    pred = Prediction.from_mock_data(days=60, seed=None)
    assert pred.predictions.select(pl.col("SnapshotTime").n_unique()).collect().item() == 60
    assert time.perf_counter() - start < 1.0
