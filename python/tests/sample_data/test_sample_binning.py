"""Tests for the ADM predictor binning view of the sample world."""

import polars as pl
import pytest
from pdstools import ADMDatamart
from pdstools.sample_data._adm import model_snapshots, summary_model_id
from pdstools.sample_data._binning import _allocate, _auc, _expected_auc, _solve_scale, predictor_binning
from pdstools.sample_data._spec import DEFAULT_SPEC
from pdstools.sample_data._world import sample_world

MOBILE_SHARED = {
    "Param.ContentVariant",
    "Param.MessageLengthBand",
    "IH.Mobile.Inbound.Clicked.pxLastOutcomeTime.DaysSince",
}


@pytest.fixture(scope="module")
def world():
    return sample_world()


@pytest.fixture(scope="module")
def models(world):
    return model_snapshots(world).collect()


@pytest.fixture(scope="module")
def raw(world):
    return predictor_binning(world).collect()


@pytest.fixture(scope="module")
def datamart(models, raw):
    return ADMDatamart(model_df=models.lazy(), predictor_df=raw.lazy())


@pytest.fixture(scope="module")
def predictors(datamart):
    return datamart.predictor_data.collect()


@pytest.fixture(scope="module")
def last_models(datamart):
    return datamart.model_data.group_by("ModelID").agg(pl.all().sort_by("SnapshotTime").last()).collect()


def test_auc_matches_known_values():
    assert _auc([10, 0], [0, 10]) == pytest.approx(1.0)
    assert _auc([5, 5], [5, 5]) == pytest.approx(0.5)
    assert _auc([0, 10], [10, 0]) == pytest.approx(1.0)


def test_solve_scale_hits_target_auc():
    shares, z = [0.2, 0.3, 0.5], [-1.0, 0.0, 1.0]
    scale = _solve_scale(shares, z, 0.6)
    assert _expected_auc(shares, z, scale) == pytest.approx(0.6, abs=1e-3)


def test_allocate_preserves_total_and_caps():
    assert sum(_allocate(101, [1.0, 2.0, 3.0])) == 101
    assert _allocate(10, [1.0, 1.0], caps=[3, 100]) == [3, 7]


def test_is_deterministic(world, raw):
    again = predictor_binning(world).collect()
    assert again.equals(raw)


def test_loader_keeps_all_categories(predictors):
    categories = set(predictors["PredictorCategory"].drop_nulls().unique().to_list())
    assert categories == {"Customer", "Account", "IH", "Param", "Context Keys"}
    assert predictors.filter(pl.col("PredictorCategory").is_null())["EntryType"].unique().to_list() == ["Classifier"]


def test_active_rows_match_model_snapshots(predictors, last_models):
    active = (
        predictors.filter(pl.col("EntryType") == "Active")
        .group_by("ModelID")
        .agg(pl.col("PredictorName").n_unique().alias("n"))
    )
    joined = last_models.join(active, on="ModelID", how="left").with_columns(pl.col("n").fill_null(0))
    assert joined.filter(pl.col("n") != pl.col("ActivePredictors")).height == 0


def test_bins_reconcile_with_predictor_totals(raw):
    sums = raw.group_by("pyModelID", "pyPredictorName").agg(
        (pl.sum("pyBinPositives") - pl.first("pyPositives")).abs().alias("dp"),
        (pl.sum("pyBinNegatives") - pl.first("pyNegatives")).abs().alias("dn"),
    )
    nb = raw.filter(pl.col("pyBinType") != "NONE").select("pyModelID", "pyPredictorName").unique()
    sums = sums.join(nb, on=["pyModelID", "pyPredictorName"])
    assert sums["dp"].max() < 1e-6
    assert sums["dn"].max() < 1e-6


def test_predictor_totals_match_model(raw, last_models):
    totals = raw.filter(pl.col("pyEntryType") == "Classifier").select("pyModelID", "pyPositives", "pyNegatives")
    joined = totals.join(last_models, left_on="pyModelID", right_on="ModelID")
    assert joined.height > 0
    assert (joined["pyPositives"] - joined["Positives"]).abs().max() < 1e-6
    assert (joined["pyNegatives"] - joined["Negatives"]).abs().max() < 1e-6


def test_stored_performance_is_bin_auc(raw):
    sample = (
        raw.filter(pl.col("pyBinType") != "NONE", pl.col("pyResponseCount") > 0)
        .group_by("pyModelID", "pyPredictorName")
        .agg(pl.first("pyPerformance"), pl.col("pyBinPositives"), pl.col("pyBinNegatives"), pl.col("pyBinIndex"))
    )
    for row in sample.head(300).iter_rows(named=True):
        order = sorted(range(len(row["pyBinIndex"])), key=lambda i: row["pyBinIndex"][i])
        positives = [int(row["pyBinPositives"][i]) for i in order]
        negatives = [int(row["pyBinNegatives"][i]) for i in order]
        assert row["pyPerformance"] == pytest.approx(_auc(positives, negatives), abs=1e-9)


def test_naive_bayes_univariate_auc_and_bin_counts(predictors):
    active = (
        predictors.filter(
            pl.col("EntryType") == "Active",
            pl.col("BinType") != "NONE",
            pl.col("ResponseCount") > 2000,
        )
        .group_by("ModelID", "PredictorName")
        .agg(pl.first("Performance"), pl.len().alias("bins"))
    )
    auc = active["Performance"]
    assert 0.51 < auc.quantile(0.1) < 0.53
    assert 0.54 < auc.median() < 0.58
    assert 0.59 < auc.quantile(0.9) < 0.66
    assert active["bins"].min() >= 4
    assert active["bins"].max() <= 9
    assert 4.5 < active["bins"].mean() < 6


def test_missing_shares(predictors):
    missing = (
        predictors.filter(
            pl.col("BinType") == "MISSING",
            ~pl.col("PredictorName").cast(pl.String).str.starts_with("IH."),
        )
        .group_by("PredictorName")
        .agg((pl.sum("BinResponseCount") / pl.sum("ResponseCount")).alias("share"))
    )
    assert 0.15 < (missing["share"] > 0.1).mean() < 0.35


def test_classifier_auc_tracks_model_performance(raw, last_models):
    classifier = raw.filter(pl.col("pyEntryType") == "Classifier", pl.col("pyBinType") != "NONE")
    classifier = classifier.group_by("pyModelID").agg(pl.first("pyPerformance"), pl.first("pyResponseCount"))
    joined = classifier.join(last_models, left_on="pyModelID", right_on="ModelID").filter(
        pl.col("pyResponseCount") >= 500,
    )
    assert joined.height > 500
    assert (joined["pyPerformance"] - joined["Performance"]).abs().max() < 0.03


def test_gradient_boost_rows(raw):
    gb = raw.filter(pl.col("pyBinType") == "NONE", pl.col("pyEntryType") != "Classifier")
    assert gb.height > 0
    assert gb["pyPerformance"].unique().to_list() == [0.0]
    summaries = [summary_model_id(c.name) for c in DEFAULT_SPEC.configurations if c.technique == "GradientBoost"]
    keys = gb.filter(pl.col("pyPredictorName").str.starts_with("py"), ~pl.col("pyModelID").is_in(summaries))
    assert keys["pyEntryType"].unique().to_list() == ["Inactive"]
    assert keys["pyFeatureImportance"].max() == 0
    summary_keys = gb.filter(pl.col("pyPredictorName").str.starts_with("py"), pl.col("pyModelID").is_in(summaries))
    assert summary_keys["pyFeatureImportance"].max() > 0


def _top(datamart, models, configuration, n=10):
    ids = models.filter(pl.col("pyConfigurationName") == configuration)["pyModelID"].unique().implode()
    importance = (
        datamart.aggregates._predictor_importance_data()
        .filter(pl.col("ModelID").is_in(ids), pl.col("EntryType") != "Classifier")
        .group_by("PredictorName")
        .agg(pl.mean("FeatureImportance"))
        .sort("FeatureImportance", descending=True)
        .collect()
    )
    return importance.head(n)["PredictorName"].cast(pl.String).to_list()


@pytest.mark.parametrize("configuration", ["Mobile_Click_Through_Rate", "Mobile_Click_Through_Rate_GB"])
def test_mobile_shared_predictors_lead(datamart, models, configuration):
    assert MOBILE_SHARED <= set(_top(datamart, models, configuration))


def test_gradient_boost_summary_ranks_context_keys(datamart, models):
    top = _top(datamart, models, "Web_Click_Through_Rate_GB")
    assert len({name for name in top if name.startswith("py")}) >= 2


def test_gradient_boost_favours_ih_and_param(datamart, models):
    ids = models.filter(pl.col("pyConfigurationName") == "Web_Click_Through_Rate_GB")["pyModelID"].unique().implode()
    by_category = (
        datamart.aggregates._predictor_importance_data()
        .filter(pl.col("ModelID").is_in(ids), pl.col("EntryType") != "Classifier")
        .group_by("PredictorCategory")
        .agg(pl.mean("FeatureImportance"))
        .collect()
    )
    importance = dict(
        zip(by_category["PredictorCategory"].cast(pl.String), by_category["FeatureImportance"], strict=True)
    )
    assert importance["IH"] > 3 * importance["Customer"]
    assert importance["IH"] > 3 * importance["Account"]
