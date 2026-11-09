"""Tests for the ADM model snapshot view of the sample world."""

import polars as pl
import pytest
from pdstools import ADMDatamart
from pdstools.sample_data._adm import (
    NO_BREAK_SPACE,
    ZERO_WIDTH_SPACE,
    adm_models,
    default_report_model,
    model_id,
    model_snapshots,
    seeded_cases,
    summary_model_id,
)
from pdstools.sample_data._spec import DEFAULT_SPEC
from pdstools.sample_data._world import sample_world

VALID_CHANNELS = ["Web", "Mobile", "Email"]


@pytest.fixture(scope="module")
def world():
    return sample_world()


@pytest.fixture(scope="module")
def raw(world):
    return model_snapshots(world).collect()


@pytest.fixture(scope="module")
def datamart(raw):
    return ADMDatamart(model_df=raw.lazy())


@pytest.fixture(scope="module")
def last_active(datamart):
    analysis = datamart.analysis
    return analysis._get_last_data().filter(analysis.health_check_active_filter())


def test_model_ids_are_stable_uuids():
    assert model_id("Cfg", "A1", "Web", "T1") == model_id("Cfg", "A1", "Web", "T1")
    assert model_id("Cfg", "A1", "Web", None) != model_id("Cfg", "A1", "Web", "T1")
    assert summary_model_id("Cfg") != model_id("Cfg", "", "", None)
    assert len(model_id("Cfg", "A1", "Web", "T1")) == 36


def test_one_model_per_context():
    models = adm_models(DEFAULT_SPEC)
    assert len(models) == len(DEFAULT_SPEC.model_contexts) == 864
    assert len({m.model_id for m in models}) == 864


def test_one_row_per_model_per_day(world, raw):
    agb_configurations = {c.name for c in DEFAULT_SPEC.configurations if c.technique == "GradientBoost"}
    assert raw.select(pl.col("pyModelID").n_unique()).item() == 864 + len(agb_configurations)
    assert raw.select(pl.struct("pyModelID", "pySnapshotTime").is_unique().all()).item()
    assert raw.select(pl.col("pySnapshotTime").n_unique()).item() == world.days
    first = raw.sort("pySnapshotTime")["pySnapshotTime"][0]
    assert first == f"{world.dates[0]:%Y%m%d}T220349.000 GMT"


def test_same_seed_is_reproducible(raw):
    assert model_snapshots(sample_world()).collect().equals(raw)
    other = model_snapshots(sample_world(seed=7)).collect()
    assert not other.equals(raw)
    assert set(other["pyModelID"]) == set(raw["pyModelID"])


def test_agb_summary_rows_are_mapped_and_dropped(datamart):
    agb_configurations = sorted(c.name for c in DEFAULT_SPEC.configurations if c.technique == "GradientBoost")
    assert sorted(datamart._agb_summary_model_ids) == sorted(summary_model_id(c) for c in agb_configurations)
    model_ids = datamart.model_data.select("ModelID").unique().collect()["ModelID"]
    assert not set(datamart._agb_summary_model_ids) & set(model_ids)
    assert model_ids.len() == 864


def test_maturity_budget(datamart, last_active):
    valid = last_active.filter(pl.col("Channel").is_in(VALID_CHANNELS))
    shares = {
        key: valid.filter(expression).height / valid.height
        for key, _, expression in datamart.analysis.health_check_maturity_criteria()
    }
    assert 0.82 <= shares["mature_decent"] <= 0.90
    assert 0.05 <= shares["immature"] <= 0.10
    assert 0.01 <= shares["stuck_at_50"] <= 0.03
    assert 0.01 <= shares["low_performance"] <= 0.03
    assert shares["never_used"] + shares["responses_no_positives"] <= 0.01


def test_call_center_is_not_a_valid_channel(last_active):
    totals = last_active.group_by("Channel").agg(pl.col("Positives").sum(), pl.col("ResponseCount").sum())
    call_center = totals.filter(pl.col("Channel") == "Call Center").row(0, named=True)
    assert call_center["Positives"] < 200
    others = totals.filter(pl.col("Channel").is_in(VALID_CHANNELS))
    assert others.height == 3
    assert (others["Positives"] >= 200).all()
    assert (others["ResponseCount"] >= 1000).all()


def test_positive_only_weeks(datamart):
    weekly = (
        datamart.model_data.sort("SnapshotTime")
        .with_columns(
            pl.col("Positives").diff().over("ModelID").alias("NewPositives"),
            pl.col("ResponseCount").diff().over("ModelID").alias("NewResponses"),
        )
        .group_by("Channel", "ModelTechnique", pl.col("SnapshotTime").dt.truncate("1w").alias("Week"))
        .agg(pl.col("NewPositives").sum(), pl.col("NewResponses").sum())
        .filter(pl.col("NewResponses") > 0, pl.col("NewPositives") == pl.col("NewResponses"))
        .select(pl.col("Channel").cast(pl.Utf8), "ModelTechnique")
        .collect()
    )
    assert sorted(weekly.rows()) == [
        ("Email", "NaiveBayes"),
        ("Mobile", "NaiveBayes"),
        ("Web", "GradientBoost"),
    ]


def test_seeded_quality_cases(world, datamart):
    cases = seeded_cases(world)
    assert sorted(cases.values()).count("stuck") == 12
    assert sorted(cases.values()).count("weak") == 12
    assert sorted(cases.values()).count("overfit") == 4
    last = (
        datamart.aggregates.last(table="model_data")
        .filter(pl.col("ModelID").is_in(list(cases)))
        .select("ModelID", "Performance", "ActivePredictors", "Positives")
        .collect()
    )
    by_case = {
        case: last.filter(pl.col("ModelID").is_in([k for k, v in cases.items() if v == case]))
        for case in cases.values()
    }
    assert (by_case["stuck"]["Performance"] == 0.5).all()
    assert (by_case["stuck"]["ActivePredictors"] == 0).all()
    assert by_case["weak"]["Performance"].is_between(0.506, 0.518).all()
    assert by_case["overfit"]["Performance"].is_between(0.86, 0.92).all()
    assert (last["Positives"] >= 200).all()


def test_naming_review(datamart):
    review = {row["Column"]: row for row in datamart.aggregates.naming_review().collect().iter_rows(named=True)}
    assert review["Group"]["Models with Empty Value"] == 3
    assert review["Group"]["Values with Hidden Characters"] == 1
    assert review["Group"]["Models with Hidden Characters"] == 3
    assert review["Name"]["Values with Hidden Characters"] == 1
    assert review["Name"]["Models with Hidden Characters"] == 2
    assert review["Treatment"]["Models with Empty Value"] == 262
    hidden = datamart.aggregates.hidden_character_names().collect()
    assert sorted(hidden["Column"]) == ["Group", "Name"]
    assert (hidden["Name Clashes"] > 0).all()


def test_naming_cases_use_expected_characters():
    models = adm_models(DEFAULT_SPEC)
    hidden_names = [m for m in models if m.naming_case == "hidden_name"]
    hidden_groups = [m for m in models if m.naming_case == "hidden_group"]
    assert len(hidden_names) == 2
    assert all(m.name.endswith(ZERO_WIDTH_SPACE) for m in hidden_names)
    assert {m.context.configuration for m in hidden_groups} == {
        "Mobile_Click_Through_Rate",
        "Mobile_Click_Through_Rate_GB",
        "OmniAdaptiveModel",
    }
    assert all(m.group.endswith(NO_BREAK_SPACE) for m in hidden_groups)
    assert sorted(m.context.channel for m in models if m.naming_case == "empty_group") == ["Email", "Web", "Web"]


def test_default_report_model(datamart):
    model = default_report_model(DEFAULT_SPEC)
    assert model.context.configuration == "Mobile_Click_Through_Rate"
    last = (
        datamart.aggregates.last(table="model_data")
        .filter(pl.col("ModelID") == model.model_id)
        .select("ActivePredictors", "TotalPredictors", "Positives", "Performance")
        .collect()
        .row(0, named=True)
    )
    assert last["TotalPredictors"] == 136
    assert 27 <= last["ActivePredictors"] <= 30
    assert last["Positives"] >= 200
    assert last["Performance"] > 0.6


def test_stopped_models_keep_their_last_counts(world, raw):
    stopped = next(m for m in adm_models(DEFAULT_SPEC) if m.context.stop_day is not None and m.context.stop_day < 182)
    rows = raw.filter(pl.col("pyModelID") == stopped.model_id).sort("pySnapshotTime")
    stop = world.window_day(stopped.context.stop_day)
    tail = rows[stop - 1 :]
    assert tail["pyResponseCount"].n_unique() == 1
    assert rows["pyResponseCount"][stop - 2] <= rows["pyResponseCount"][stop - 1]


def test_mobile_models_learn_from_zero(world, raw):
    first_day = raw.filter(pl.col("pySnapshotTime") == raw["pySnapshotTime"].min())
    mobile = first_day.filter(
        pl.col("pyConfigurationName").str.starts_with("Mobile_"),
        pl.col("pyName").is_not_null(),
    )
    last_day = raw.filter(
        pl.col("pySnapshotTime") == raw["pySnapshotTime"].max(),
        pl.col("pyConfigurationName").str.starts_with("Mobile_"),
        pl.col("pyName").is_not_null(),
    )
    assert mobile.height > 0
    assert mobile["pyPositives"].sum() < 0.02 * last_day["pyPositives"].sum()
    web = first_day.filter(pl.col("pyConfigurationName") == "Web_Click_Through_Rate_GB")
    assert web["pyPositives"].median() > 200


@pytest.mark.parametrize("days", [56, 365])
def test_other_window_lengths(days):
    world = sample_world(days=days)
    raw = model_snapshots(world).collect()
    assert raw.select(pl.col("pySnapshotTime").n_unique()).item() == days
    assert raw.select(pl.col("pyModelID").n_unique()).item() == 867


def test_from_mock_data_window_and_views():
    dm = ADMDatamart.from_mock_data(days=56, end="canonical", seed=7, scale=0.5)
    models = dm.model_data.collect()
    assert models["SnapshotTime"].dt.date().n_unique() == 56
    assert str(models["SnapshotTime"].max().date()) == "2000-07-02"
    assert models["ModelID"].n_unique() == 864
    predictors = dm.predictor_data.collect()
    assert set(predictors["ModelID"].unique()) <= set(models["ModelID"].unique())
    assert set(predictors["EntryType"].unique().cast(pl.String)) == {"Active", "Inactive", "Classifier"}
