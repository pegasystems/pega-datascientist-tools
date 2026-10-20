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


def test_success_rate_average_is_per_channel_and_direction():
    models = pl.concat(
        [
            _models([("a", "web_a", 1000, 100000, "NaiveBayes"), ("b", "web_b", 3000, 100000, "NaiveBayes")]),
            _models(
                [("c", "mail_c", 50, 100000, "NaiveBayes"), ("d", "mail_d", 150, 100000, "NaiveBayes")]
            ).with_columns(Channel=pl.lit("Email"), Direction=pl.lit("Outbound")),
        ]
    )
    dm = ADMDatamart(model_df=models)
    df = dm.aggregates.success_rate_intervals().collect()
    averages = dict(zip(df["Name"], df["AverageSuccessRate"], strict=True))
    assert averages["web_a"] == pytest.approx(0.02)
    assert averages["mail_c"] == pytest.approx(0.001)
    fig = dm.plot.success_rate_intervals()
    assert [a.text for a in fig.layout.annotations] == ["Email/Outbound", "Web/Inbound"]
    assert sum(1 for t in fig.data if t.mode == "lines") == 2


def test_hidden_characters():
    rows = [
        ("m1", "offer", 1, 10, "NaiveBayes"),
        ("m2", f"offer{ZWSP}", 1, 10, "NaiveBayes"),
        ("m3", " lonely", 1, 10, "NaiveBayes"),
        ("m4", "clean", 1, 10, "NaiveBayes"),
    ]
    dm = ADMDatamart(model_df=_models(rows))
    df = dm.aggregates.hidden_character_names().collect()
    assert df.sort("Value").select(
        "Column", "Value", "Hidden Characters", "Cleaned", "Models", "Name Clashes"
    ).rows() == [
        ("Name", '" lonely"', "leading or trailing whitespace", '"lonely"', 1, 0),
        ("Name", '"offer[U+200B]"', "U+200B zero width space", '"offer"', 1, 1),
    ]


@pytest.mark.parametrize("char", ["\u2003", "\u3000", "\u00a0", "\u200b", "\ufeff", "\t"])
def test_hidden_characters_inside_value(char):
    dm = ADMDatamart(model_df=_models([("m1", f"a{char}b", 1, 10, "NaiveBayes")]))
    df = dm.aggregates.hidden_character_names().collect()
    assert df["Cleaned"].to_list() == ['"ab"']


def test_hidden_characters_none():
    dm = ADMDatamart(model_df=_models([("m1", "clean", 1, 10, "NaiveBayes")]))
    df = dm.aggregates.hidden_character_names().collect()
    assert df.height == 0
    assert df.columns == ["Column", "Value", "Hidden Characters", "Cleaned", "Models", "Name Clashes"]


def test_naming_review():
    rows = [
        ("m1", "offer", 1, 10, "NaiveBayes"),
        ("m2", f"offer{ZWSP}", 1, 10, "NaiveBayes"),
        ("m3", " ", 1, 10, "NaiveBayes"),
        ("m4", "clean", 1, 10, "NaiveBayes"),
    ]
    dm = ADMDatamart(model_df=_models(rows))
    df = dm.aggregates.naming_review().collect()
    name = df.filter(pl.col("Column") == "Name").row(0, named=True)
    assert name["Distinct Values"] == 3
    assert name["Models with Empty Value"] == 1
    assert name["Values with Hidden Characters"] == 1
    assert name["Models with Hidden Characters"] == 1
    assert df.filter(pl.col("Column") == "Channel")["Models with Empty Value"].item() == 0


def test_naming_review_no_columns():
    dm = ADMDatamart(model_df=_models([("m1", "clean", 1, 10, "NaiveBayes")]))
    df = dm.aggregates.naming_review(columns=("Nonexistent",)).collect()
    assert df.height == 0
    assert df.columns[0] == "Column"


def _importance_dm(bin_responses=0):
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
            "BinResponseCount": bin_responses,
            "BinPositives": bin_responses // 10,
            "Type": "numeric",
            "Performance": 0.5,
            "ResponseCount": bin_responses,
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


@pytest.fixture
def configuration_importance_dm():
    models = _models(
        [
            ("a1", "first", 10, 100, "GradientBoost"),
            ("a2", "second", 10, 100, "GradientBoost"),
            ("summary_a", "", 20, 200, "GradientBoost"),
            ("b1", "third", 10, 100, "GradientBoost"),
            ("summary_b", "", 10, 100, "GradientBoost"),
            ("nb", "fourth", 10, 100, "NaiveBayes"),
        ],
    ).with_columns(
        Configuration=pl.when(pl.col("ModelID").is_in(["b1", "summary_b"])).then(pl.lit("B")).otherwise(pl.lit("A")),
        Channel=pl.when(pl.col("Name") == "").then(pl.lit("")).otherwise(pl.col("Channel")),
    )
    rows = []
    for model_id, primary in [
        ("a1", 0.0),
        ("a2", 0.0),
        ("summary_a", 20.0),
        ("b1", 0.0),
        ("summary_b", 30.0),
        ("nb", 0.0),
    ]:
        rows.extend(
            [
                (model_id, "pyGroup", primary, "Active" if primary else "Inactive"),
                (model_id, "Customer.Age", 100.0 - primary, "Active"),
                (model_id, "Classifier", 0.0, "Classifier"),
            ],
        )
    # Importance repeats across bins and must be counted only once.
    rows.append(("summary_a", "pyGroup", 20.0, "Active"))
    predictors = (
        pl.DataFrame(
            rows,
            schema=["ModelID", "PredictorName", "FeatureImportance", "EntryType"],
            orient="row",
        )
        .lazy()
        .with_columns(
            SnapshotTime=pl.lit("20250102"),
            BinIndex=pl.lit(1),
            BinPositives=pl.lit(0),
            BinResponseCount=pl.lit(0),
            Performance=pl.lit(0.5),
        )
    )
    return ADMDatamart(model_df=models, predictor_df=predictors)


def test_agb_configuration_importance_preserves_model_and_bin_data(configuration_importance_dm):
    dm = configuration_importance_dm
    assert dm.model_data.select("ModelID").collect()["ModelID"].sort().to_list() == ["a1", "a2", "b1", "nb"]
    assert dm.model_data.select(pl.col("ResponseCount").sum(), pl.col("Performance").mean()).collect().row(0) == (
        400.0,
        0.6,
    )
    # The original action-level predictor values remain intact for other consumers.
    assert dm.combined_data.filter(pl.col("PredictorName") == "pyGroup").select(
        pl.col("FeatureImportance").sum(),
        (pl.col("EntryType") == "Active").sum(),
    ).collect().row(0) == (0.0, 0)
    result = dm.aggregates.predictor_importance(by="PredictorCategory").collect()
    assert result.select(
        "Configuration", "ModelTechnique", "PredictorCategory", "Models", "Importance %", "Active Predictors"
    ).sort(
        "Configuration",
        "ModelTechnique",
        "PredictorCategory",
    ).rows() == [
        ("A", "GradientBoost", "Customer", 2, 80.0, 1.0),
        ("A", "GradientBoost", "Primary", 2, 20.0, 1.0),
        ("A", "NaiveBayes", "Customer", 1, 100.0, 1.0),
        ("A", "NaiveBayes", "Primary", 1, 0.0, 0.0),
        ("B", "GradientBoost", "Customer", 1, 70.0, 1.0),
        ("B", "GradientBoost", "Primary", 1, 30.0, 1.0),
    ]
    selected = dm.aggregates.predictor_importance(query=pl.col("Name") == "second").collect()
    assert selected.select("PredictorName", "Models", "Importance %").sort("PredictorName").rows() == [
        ("Customer.Age", 1, 80.0),
        ("pyGroup", 1, 20.0),
    ]


def test_agb_configuration_importance_survives_report_cache(configuration_importance_dm, tmp_path):
    dm = configuration_importance_dm
    model_path, predictor_path = dm.save_data(tmp_path, selected_model_ids=["a1", "b1"])
    restored = ADMDatamart.from_ds_export(model_filename=str(model_path), predictor_filename=str(predictor_path))
    # Report sections reconstruct filtered datamarts again after cache loading.
    filtered = ADMDatamart(
        model_df=restored.model_data.filter(pl.col("Configuration") == "B"),
        predictor_df=restored.predictor_data,
    )
    result = filtered.aggregates.predictor_importance(by="PredictorCategory").collect()
    assert result.select("PredictorCategory", "Models", "Importance %").sort("PredictorCategory").rows() == [
        ("Customer", 1, 70.0),
        ("Primary", 1, 30.0),
    ]


def test_agb_configuration_importance_missing_predictor_rows(configuration_importance_dm):
    dm = configuration_importance_dm
    predictors = dm.predictor_data.drop("ConfigurationImportance", "ConfigurationActive")
    # Recreate a summary model whose predictor snapshot is absent.
    summary = dm.model_data.filter(pl.col("ModelID") == "a1").with_columns(
        ModelID=pl.lit("missing_summary"),
        Name=pl.lit(""),
    )
    restored = ADMDatamart(model_df=pl.concat([dm.model_data, summary]), predictor_df=predictors)
    result = restored.aggregates.predictor_importance(query=pl.col("ModelID") == "a1").collect()
    assert result.select("PredictorName", "Importance %").sort("PredictorName").rows() == [
        ("Customer.Age", 100.0),
        ("pyGroup", 0.0),
    ]


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


def test_finding_very_low_success_rate_relative_to_channel():
    rows = [
        ("m1", "dead", 1, 100000, "NaiveBayes"),
        ("m2", "fine", 20000, 100000, "NaiveBayes"),
    ]
    findings = Analysis(ADMDatamart(model_df=_models(rows)))._check_very_low_success_rates()
    assert len(findings) == 1 and findings[0].data["model_count"] == 1

    # the same low rate is unremarkable when its own channel has a similarly low average
    web = _models([("m1", "a", 1, 100000, "NaiveBayes"), ("m2", "b", 2, 100000, "NaiveBayes")])
    assert Analysis(ADMDatamart(model_df=web))._check_very_low_success_rates() == []


def test_category_plots_share_colors_and_legend_title():
    dm = _importance_dm(bin_responses=100)
    expected = dm.predictor_category_color_map
    assert expected["Customer"] == "#001F5F"
    figs = {
        "importance": dm.plot.predictor_importance(),
        "category_importance": dm.plot.predictor_category_importance(),
        "contribution": dm.plot.predictor_contribution(),
        "performance": dm.plot.predictor_performance(),
        "category_performance": dm.plot.predictor_category_performance(),
        "count": dm.plot.predictor_count(by=["PredictorCategory"]),
    }
    for name, fig in figs.items():
        assert type(fig).__name__ == "Figure", name
        assert fig.layout.legend.title.text == "Predictor Category", name
        colors = {t.name: t.marker.color for t in fig.data if t.name in expected}
        assert colors, name
        assert all(colors[c] == expected[c] for c in colors), name
