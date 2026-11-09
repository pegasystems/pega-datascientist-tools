"""Tests for the ADM predictor binning view of the sample world."""

from dataclasses import replace
from random import Random

import polars as pl
import pytest
from pdstools import ADMDatamart
from pdstools.sample_data._adm import model_snapshots, summary_model_id
from pdstools.sample_data._binning import (
    _allocate,
    _auc,
    _counts,
    _expected_auc,
    _importance,
    _inactive_bins,
    _missing_share,
    _solve_scale,
    predictor_binning,
)
from pdstools.sample_data._spec import DEFAULT_SPEC
from pdstools.sample_data._world import sample_world

MOBILE_SHARED = {
    "Param.ContentVariant",
    "Param.MessageLengthBand",
    "IH.Mobile.Inbound.Clicked.pxLastOutcomeTime.DaysSince",
}
CHURN_SCORE = "Customer.ChurnRiskModelScore"


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
    assert categories == {"Customer", "Account", "IH", "Param", "Context Keys", "External Model"}
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
    assert 0.53 < auc.quantile(0.1) < 0.55
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


def test_missing_probability_endpoints():
    predictor = next(p for p in DEFAULT_SPEC.predictors if p.id == "Param.CallToActionStyle")
    assert predictor.missing_probability == 1.0
    placement = next(p for p in DEFAULT_SPEC.predictors if p.id == "Param.Placement")
    assert placement.missing_probability == 0.01
    for seed in (7, 42):
        assert _missing_share(predictor, Random(seed)) == 1.0
        assert _missing_share(replace(predictor, missing_probability=0.0), Random(seed)) == 0.0


def test_partially_missing_parameter(predictors):
    parameter = predictors.filter(
        pl.col("PredictorName") == "Param.CreativeFormat",
        pl.col("BinType") == "MISSING",
    )
    share = parameter["BinResponseCount"].sum() / parameter["ResponseCount"].sum()
    assert 0.1 < share < 0.3


def test_fully_missing_numeric_predictor_has_one_uninformative_bin():
    predictor = next(p for p in DEFAULT_SPEC.predictors if p.id == "Customer.CreditScore")
    bins, contents = _inactive_bins(replace(predictor, missing_probability=1.0), Random(42))
    positives, negatives = _counts(bins, positives=37, negatives=63, scale=0.0)
    assert contents == "Range : []"
    assert [(b.kind, b.symbol, b.share) for b in bins] == [("MISSING", "MISSING", 1.0)]
    assert positives == [37]
    assert negatives == [63]
    assert _auc(positives, negatives) == 0.5
    assert _importance(positives, negatives) == 0.0


def test_fully_missing_parameter_is_inactive_and_reconciles(raw):
    parameter = raw.filter(pl.col("pyPredictorName") == "Param.CallToActionStyle")
    nb = parameter.filter(pl.col("pyBinType") != "NONE")
    assert nb.height > 0
    assert nb["pyBinSymbol"].unique().to_list() == ["MISSING"]
    assert nb["pyEntryType"].unique().to_list() == ["Inactive"]
    assert nb["pyPerformance"].unique().to_list() == [0.5]
    assert nb["pyFeatureImportance"].unique().to_list() == [0.0]
    assert nb["pyBinResponseCount"].sum() > 0
    assert (nb["pyBinResponseCount"] - nb["pyResponseCount"]).abs().max() == 0

    agb = parameter.filter(pl.col("pyBinType") == "NONE")
    assert agb.height > 0
    assert agb["pyEntryType"].unique().to_list() == ["Inactive"]
    assert agb["pyFeatureImportance"].unique().to_list() == [0.0]


def test_classifier_auc_tracks_model_performance(raw, last_models):
    classifier = raw.filter(pl.col("pyEntryType") == "Classifier", pl.col("pyBinType") != "NONE")
    classifier = classifier.group_by("pyModelID").agg(pl.first("pyPerformance"), pl.first("pyResponseCount"))
    joined = classifier.join(last_models, left_on="pyModelID", right_on="ModelID").filter(
        pl.col("pyResponseCount") >= 500,
    )
    assert joined.height > 500
    assert (joined["pyPerformance"] - joined["Performance"]).abs().max() < 0.03
    for channel, issue, technique in [
        ("Email", "Service", "NaiveBayes"),
        ("Web", "Risk", "GradientBoost"),
    ]:
        context = joined.filter(
            pl.col("Channel") == channel,
            pl.col("Issue") == issue,
            pl.col("ModelTechnique") == technique,
        )
        assert context.height > 10
        assert 0.52 < context["Performance"].median() < 0.55


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


@pytest.mark.parametrize("seed", [42, 7])
def test_customer_predictors_are_mixed_into_each_health_check_importance_plot(datamart, seed):
    dm = datamart
    if seed != 42:
        world = sample_world(seed=seed)
        dm = ADMDatamart(model_df=model_snapshots(world), predictor_df=predictor_binning(world))

    facets = [
        ("Mobile_Click_Through_Rate", "Mobile", "NaiveBayes"),
        ("Email_Click_Through_Rate", "Email", "NaiveBayes"),
        ("OmniAdaptiveModel", "Web", "NaiveBayes"),
        ("OmniAdaptiveModel", "Mobile", "NaiveBayes"),
        ("OmniAdaptiveModel", "Email", "NaiveBayes"),
        ("Web_Click_Through_Rate_GB", "Web", "GradientBoost"),
        ("Web_Loans_Click_Through_Rate_GB", "Web", "GradientBoost"),
        ("Mobile_Click_Through_Rate_GB", "Mobile", "GradientBoost"),
    ]
    for configuration, channel, technique in facets:
        query = (
            (pl.col("Configuration") == configuration)
            & (pl.col("Channel") == channel)
            & dm.analysis.health_check_active_filter()
        )
        if technique == "NaiveBayes":
            data = dm.plot.predictor_performance(
                query=query & (pl.col("ResponseCountBin") > 200), top_n=30, return_df=True
            )
        else:
            data = dm.plot.predictor_importance(query=query, top_n=20, return_df=True)
        categories = data["PredictorCategory"].cast(pl.String).to_list()[:20]
        customer_positions = [i for i, category in enumerate(categories) if category == "Customer"]
        scope = f"seed={seed}, configuration={configuration}, channel={channel}"
        assert len(customer_positions) >= 3, scope
        assert customer_positions[0] < 13, scope
        assert any(
            category != "Customer" for category in categories[customer_positions[0] + 1 : customer_positions[-1]]
        ), scope
        assert {"IH", "Param"} & set(categories[:5]), scope


def test_external_churn_score_ranks_in_health_check_plots(datamart):
    facets = [
        ("Mobile_Click_Through_Rate", "Mobile", "NaiveBayes"),
        ("Email_Click_Through_Rate", "Email", "NaiveBayes"),
        ("Mobile_Click_Through_Rate_GB", "Mobile", "GradientBoost"),
        ("Web_Click_Through_Rate_GB", "Web", "GradientBoost"),
    ]
    mean_ranks, display_ranks = [], []
    for configuration, channel, technique in facets:
        query = (
            (pl.col("Configuration") == configuration)
            & (pl.col("Channel") == channel)
            & datamart.analysis.health_check_active_filter()
        )
        if technique == "NaiveBayes":
            query &= pl.col("ResponseCountBin") > 200
            data = datamart.plot.predictor_performance(query=query, top_n=30, return_df=True)
            fig = datamart.plot.predictor_performance(query=query, top_n=30)
            mean_order = data.sort("mean", descending=True)["PredictorName"].cast(pl.String).to_list()
        else:
            data = datamart.plot.predictor_importance(query=query, top_n=20, return_df=True)
            fig = datamart.plot.predictor_importance(query=query, top_n=20)
            mean_order = data["PredictorName"].cast(pl.String).to_list()

        display_order = data["PredictorName"].cast(pl.String).to_list()
        assert CHURN_SCORE in mean_order
        mean_ranks.append(mean_order.index(CHURN_SCORE) + 1)
        display_ranks.append(display_order.index(CHURN_SCORE) + 1)
        assert data.filter(pl.col("PredictorName") == CHURN_SCORE)["PredictorCategory"].to_list() == ["External Model"]
        traces = [trace for trace in fig.data if trace.name == "External Model"]
        assert len(traces) == 1
        assert CHURN_SCORE in traces[0].y
        assert traces[0].marker.color == "#DE4342"
        assert traces[0].showlegend
    assert 10 <= sum(mean_ranks) / len(mean_ranks) <= 20
    assert 10 <= sum(display_ranks) / len(display_ranks) <= 20


def test_external_churn_score_has_numeric_bins_and_never_enters_loans(raw, models):
    score = raw.filter(pl.col("pyPredictorName") == CHURN_SCORE).join(
        models.select("pyModelID", "pyConfigurationName", "pyModelTechnique").unique(),
        on="pyModelID",
    )
    assert set(score["pyModelTechnique"].unique()) == {"NaiveBayes", "GradientBoost"}
    assert score["pyType"].unique().to_list() == ["numeric"]
    assert "Web_Loans_Click_Through_Rate_GB" not in set(score["pyConfigurationName"])
    nb = score.filter(pl.col("pyModelTechnique") == "NaiveBayes", pl.col("pyEntryType") == "Active")
    assert nb["pyBinLowerBound"].min() == 0
    assert nb["pyBinUpperBound"].max() == 1
