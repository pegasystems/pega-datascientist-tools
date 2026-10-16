"""Success rate intervals, hidden characters and predictor importance."""

import polars as pl
import pytest
from pdstools import ADMDatamart
from pdstools.adm.Analysis import Analysis
from pdstools.utils import cdh_utils

ZWSP = "\u200b"


def _models(rows):
    return pl.DataFrame(
        {
            "ModelID": [r[0] for r in rows],
            "Name": [r[1] for r in rows],
            "Positives": [r[2] for r in rows],
            "ResponseCount": [r[3] for r in rows],
            "ModelTechnique": [r[4] for r in rows],
            "SnapshotTime": "20250101",
            "Configuration": "Cfg",
            "Channel": "Web",
            "Direction": "Inbound",
            "Performance": 0.6,
        }
    ).lazy()


def test_wilson_interval_exact():
    df = pl.DataFrame({"p": [50, 0, 0], "n": [100, 10, 0]}).with_columns(
        lo=cdh_utils.wilson_interval(pl.col("p"), pl.col("n"))[0],
        hi=cdh_utils.wilson_interval(pl.col("p"), pl.col("n"))[1],
    )
    assert df["lo"][0] == pytest.approx(0.4038, abs=1e-3)
    assert df["hi"][0] == pytest.approx(0.5962, abs=1e-3)
    assert df["lo"][1] == pytest.approx(0.0, abs=1e-9)
    assert df["hi"][1] == pytest.approx(0.2775, abs=1e-3)
    assert df["lo"][2] is None


@pytest.fixture
def rates_dm():
    rows = [
        ("m1", "very_low", 1, 100000, "NaiveBayes"),
        ("m2", "below", 5000, 100000, "NaiveBayes"),
        ("m3", "same", 21600, 100000, "NaiveBayes"),
        ("m4", "above", 60000, 100000, "NaiveBayes"),
        ("m5", "empty", 0, 0, "NaiveBayes"),
        ("m6", "almost_all", 200, 210, "NaiveBayes"),
    ]
    return ADMDatamart(model_df=_models(rows))


def test_success_rate_intervals_categories(rates_dm):
    df = rates_dm.aggregates.success_rate_intervals().collect()
    categories = dict(zip(df["Name"], df["SuccessRateCategory"], strict=True))
    assert "empty" not in categories
    assert categories["very_low"] == "Very low"
    assert categories["below"] == "Below average"
    assert categories["same"] == "Consistent with average"
    assert categories["above"] == "Above average"
    assert df["AverageSuccessRate"][0] == pytest.approx(86801 / 400210)
    row = df.filter(pl.col("Name") == "almost_all").row(0, named=True)
    assert row["Negatives"] == 10
    assert row["SuccessRate"] == pytest.approx(200 / 210)


def test_success_rate_intervals_nothing_in_scope(rates_dm):
    df = rates_dm.aggregates.success_rate_intervals(query=pl.col("Name") == "nothing").collect()
    assert df.height == 0
    assert rates_dm.plot.success_rate_intervals(query=pl.col("Name") == "nothing") is None


def test_success_rate_plot(rates_dm):
    fig = rates_dm.plot.success_rate_intervals()
    assert {t.name for t in fig.data if t.mode == "markers"} >= {"Very low", "Above average"}
    assert rates_dm.plot.success_rate_intervals(return_df=True).height == 5


def test_hidden_characters():
    rows = [
        ("m1", "offer", 1, 10, "NaiveBayes"),
        ("m2", f"offer{ZWSP}", 1, 10, "NaiveBayes"),
        ("m3", " lonely", 1, 10, "NaiveBayes"),
        ("m4", "clean", 1, 10, "NaiveBayes"),
    ]
    dm = ADMDatamart(model_df=_models(rows))
    df = dm.aggregates.hidden_character_names().collect()
    assert df.sort("Value").select("Column", "Value", "Cleaned", "Models", "Lookalike Models").rows() == [
        ("Name", '" lonely"', "lonely", 1, 0),
        ("Name", '"offer<U+200B>"', "offer", 1, 1),
    ]


def test_hidden_characters_none():
    dm = ADMDatamart(model_df=_models([("m1", "clean", 1, 10, "NaiveBayes")]))
    df = dm.aggregates.hidden_character_names().collect()
    assert df.height == 0
    assert df.columns == ["Column", "Value", "Cleaned", "Models", "Lookalike Models"]


def _importance_dm():
    models = _models(
        [
            ("a1", "x", 10, 100, "GradientBoost"),
            ("a2", "y", 10, 100, "GradientBoost"),
            ("a3", "untrained", 10, 100, "GradientBoost"),
        ]
    )
    rows = [
        ("a1", "Age", "Customer", "Active", 30.0),
        ("a1", "Age", "Customer", "Active", 30.0),
        ("a1", "Param.Tier", "Param", "Active", 10.0),
        ("a1", "Classifier", "Classifier", "Classifier", None),
        ("a2", "Age", "Customer", "Active", 50.0),
        ("a2", "IH.Clicks", "IH", "Active", 50.0),
        ("a3", "Age", "Customer", "Active", 0.0),
    ]
    predictors = pl.DataFrame(
        {
            "ModelID": [r[0] for r in rows],
            "PredictorName": [r[1] for r in rows],
            "PredictorCategory": [r[2] for r in rows],
            "EntryType": [r[3] for r in rows],
            "FeatureImportance": [r[4] for r in rows],
            "SnapshotTime": "20250101",
            "BinIndex": 1,
            "BinType": "NONE",
            "BinSymbol": "",
            "BinResponseCount": 0,
            "BinPositives": 0,
            "Type": "numeric",
            "Performance": 0.5,
            "ResponseCount": 0,
            "Positives": 0,
        }
    ).lazy()
    return ADMDatamart(model_df=models, predictor_df=predictors)


def test_predictor_importance_normalisation():
    dm = _importance_dm()
    df = dm.aggregates.predictor_importance().collect()
    shares = {r["PredictorName"]: r for r in df.iter_rows(named=True)}
    assert set(shares) == {"Age", "Param.Tier", "IH.Clicks"}
    assert shares["Age"]["Models"] == 2
    # a1: Age 30/40=75%, Tier 25%; a2: Age 50%, Clicks 50%; untrained model left out
    assert shares["Age"]["Importance %"] == pytest.approx((75 + 50) / 2)
    assert shares["Param.Tier"]["Importance %"] == pytest.approx(12.5)
    assert shares["IH.Clicks"]["Importance %"] == pytest.approx(25.0)
    assert sum(r["Importance %"] for r in shares.values()) == pytest.approx(100)
    assert shares["Age"]["Active Predictors"] == pytest.approx(1.0)


def test_predictor_importance_by_category_and_plots():
    dm = _importance_dm()
    cats = dm.aggregates.predictor_importance(by="PredictorCategory").collect()
    assert dict(zip(cats["PredictorCategory"], cats["Importance %"], strict=True)) == pytest.approx(
        {"Customer": 62.5, "Param": 12.5, "IH": 25.0}
    )
    assert dm.plot.predictor_importance().data[0].x[0] == pytest.approx(62.5)
    assert [t.name for t in dm.plot.predictor_category_importance().data] == ["Customer", "IH", "Param"]
    assert dm.plot.predictor_importance(query=pl.col("Name") == "nothing") is None


def test_predictor_importance_without_column():
    dm = ADMDatamart(model_df=_models([("m1", "x", 1, 10, "NaiveBayes")]))
    with pytest.raises(ValueError):
        dm.aggregates.predictor_importance()


def test_findings_hidden_and_implausible():
    rows = [
        ("m1", f"offer{ZWSP}", 200, 210, "NaiveBayes"),
        ("m2", "other", 100, 10000, "NaiveBayes"),
    ]
    dm = ADMDatamart(model_df=_models(rows))
    analysis = Analysis(dm)
    hidden = analysis._check_hidden_characters()
    assert len(hidden) == 1 and hidden[0].data["value_count"] == 1
    implausible = analysis._check_implausible_success_rates()
    assert len(implausible) == 1 and implausible[0].data["model_count"] == 1

    clean = Analysis(ADMDatamart(model_df=_models([("m1", "x", 100, 10000, "NaiveBayes")])))
    assert clean._check_hidden_characters() == []
    assert clean._check_implausible_success_rates() == []
