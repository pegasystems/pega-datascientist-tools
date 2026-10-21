"""Success rate intervals, hidden characters and predictor importance."""

import datetime
from pathlib import Path
from unittest.mock import MagicMock

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


_ACTION_SNAPSHOT = datetime.datetime(2025, 1, 2)

# Mirrors how AGB writes importance: action models hold their context keys
# constant (zero, Inactive) and rescale the rest to 100%; the configuration
# summary model (empty context keys) carries the global importance of all
# predictors.
_IMPORTANCE = {
    # ModelID: (pyGroup, Customer.Age, Customer.Income)
    "a1": (0.0, 75.0, 25.0),
    "a2": (0.0, 50.0, 50.0),
    "summary_a": (10.0, 60.0, 30.0),
    "b1": (0.0, 100.0, 0.0),
    "summary_b": (20.0, 80.0, 0.0),
    "nb": (0.0, 40.0, 60.0),
}


def _configuration_importance_inputs(importance=None, snapshot_times=None):
    importance = importance or _IMPORTANCE
    snapshot_times = snapshot_times or {}
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
    for model_id, values in importance.items():
        for predictor, value in zip(("pyGroup", "Customer.Age", "Customer.Income"), values, strict=True):
            for snapshot in snapshot_times.get(model_id, [_ACTION_SNAPSHOT]):
                rows.append((model_id, predictor, value, "Active" if value else "Inactive", snapshot))
        rows.append((model_id, "Classifier", 0.0, "Classifier", _ACTION_SNAPSHOT))
    # Importance repeats across bins and must be counted only once.
    rows.append(("summary_a", "pyGroup", importance["summary_a"][0], "Active", _ACTION_SNAPSHOT))
    predictors = pl.DataFrame(
        rows,
        schema=["ModelID", "PredictorName", "FeatureImportance", "EntryType", "SnapshotTime"],
        orient="row",
    ).lazy()
    predictors = predictors.with_columns(
        BinIndex=pl.lit(1),
        BinPositives=pl.lit(0),
        BinResponseCount=pl.lit(0),
        Performance=pl.lit(0.5),
    )
    return models, predictors


@pytest.fixture
def configuration_importance_dm():
    models, predictors = _configuration_importance_inputs()
    return ADMDatamart(model_df=models, predictor_df=predictors)


def _category_importance(dm, **kwargs):
    result = dm.aggregates.predictor_importance(by="PredictorCategory", **kwargs).collect()
    return (
        result.select(
            "Configuration", "ModelTechnique", "PredictorCategory", "Models", "Importance %", "Active Predictors"
        )
        .sort("Configuration", "ModelTechnique", "PredictorCategory")
        .rows()
    )


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
    assert dm.combined_data.filter(pl.col("ConfigurationImportance").is_not_null()).select(
        "ModelID", "PredictorName", "ConfigurationImportance", "ConfigurationActive"
    ).sort("ModelID", "PredictorName").collect().rows() == [
        ("a1", "Customer.Age", 60.0, True),
        ("a1", "Customer.Income", 30.0, True),
        ("a1", "pyGroup", 10.0, True),
        ("a2", "Customer.Age", 60.0, True),
        ("a2", "Customer.Income", 30.0, True),
        ("a2", "pyGroup", 10.0, True),
        ("b1", "Customer.Age", 80.0, True),
        ("b1", "Customer.Income", 0.0, False),
        ("b1", "pyGroup", 20.0, True),
    ]


def test_agb_predictor_importance_reports_configuration_importance(configuration_importance_dm):
    dm = configuration_importance_dm
    # AGB configurations report exactly their summary (global) importance;
    # Naive Bayes keeps its own values.
    assert _category_importance(dm) == [
        ("A", "GradientBoost", "Context Keys", 2, 10.0, 1.0),
        ("A", "GradientBoost", "Customer", 2, 90.0, 2.0),
        ("A", "NaiveBayes", "Context Keys", 1, 0.0, 0.0),
        ("A", "NaiveBayes", "Customer", 1, 100.0, 2.0),
        ("B", "GradientBoost", "Context Keys", 1, 20.0, 1.0),
        ("B", "GradientBoost", "Customer", 1, 80.0, 1.0),
    ]
    selected = dm.aggregates.predictor_importance(query=pl.col("Name") == "second").collect()
    assert selected.select("PredictorName", "PredictorCategory", "Models", "Importance %").sort(
        "PredictorName"
    ).rows() == [
        ("Customer.Age", "Customer", 1, 60.0),
        ("Customer.Income", "Customer", 1, 30.0),
        ("pyGroup", "Context Keys", 1, 10.0),
    ]


def test_agb_configuration_importance_uses_nearest_summary_snapshot():
    later = _ACTION_SNAPSHOT + datetime.timedelta(seconds=1)
    older = _ACTION_SNAPSHOT - datetime.timedelta(days=30)
    models, predictors = _configuration_importance_inputs(snapshot_times={"summary_a": [older, later]})
    stale = predictors.filter((pl.col("ModelID") == "summary_a") & (pl.col("SnapshotTime") == older))
    current = predictors.filter(~((pl.col("ModelID") == "summary_a") & (pl.col("SnapshotTime") == older)))
    stale = stale.with_columns(
        FeatureImportance=pl.when(pl.col("PredictorName") == "pyGroup").then(50.0).otherwise(25.0),
        EntryType=pl.when(pl.col("PredictorName") == "Classifier")
        .then(pl.lit("Classifier"))
        .otherwise(pl.lit("Active")),
    )
    dm = ADMDatamart(model_df=models, predictor_df=pl.concat([current, stale]))
    assert _category_importance(dm, query=pl.col("Configuration") == "A")[:2] == [
        ("A", "GradientBoost", "Context Keys", 2, 10.0, 1.0),
        ("A", "GradientBoost", "Customer", 2, 90.0, 2.0),
    ]


def test_agb_configuration_importance_survives_report_cache(configuration_importance_dm, tmp_path):
    model_path, predictor_path = configuration_importance_dm.save_data(tmp_path, selected_model_ids=["a1", "b1"])
    restored = ADMDatamart.from_ds_export(model_filename=str(model_path), predictor_filename=str(predictor_path))
    # Report sections reconstruct filtered datamarts again after cache loading.
    filtered = ADMDatamart(
        model_df=restored.model_data.filter(pl.col("Configuration") == "B"),
        predictor_df=restored.predictor_data,
    )
    assert _category_importance(filtered) == [
        ("B", "GradientBoost", "Context Keys", 1, 20.0, 1.0),
        ("B", "GradientBoost", "Customer", 1, 80.0, 1.0),
    ]


def test_agb_configuration_importance_missing_summary_predictors():
    models, predictors = _configuration_importance_inputs()
    dm = ADMDatamart(model_df=models, predictor_df=predictors.filter(pl.col("ModelID") != "summary_b"))
    # Without summary predictor rows, configuration B keeps action-level values.
    assert _category_importance(dm, query=pl.col("Configuration") == "B") == [
        ("B", "GradientBoost", "Context Keys", 1, 0.0, 0.0),
        ("B", "GradientBoost", "Customer", 1, 100.0, 1.0),
    ]


def test_agb_configuration_importance_ambiguous_summary(caplog):
    models, predictors = _configuration_importance_inputs()
    # A configuration with summary models per channel has no single global row.
    second_summary = models.filter(pl.col("ModelID") == "summary_a").with_columns(ModelID=pl.lit("summary_a2"))
    second_predictors = predictors.filter(pl.col("ModelID") == "summary_a").with_columns(
        ModelID=pl.lit("summary_a2"),
    )
    with caplog.at_level("WARNING", logger="pdstools.adm.ADMDatamart"):
        dm = ADMDatamart(
            model_df=pl.concat([models, second_summary]),
            predictor_df=pl.concat([predictors, second_predictors]),
        )
    assert "multiple summary models per configuration" in caplog.text
    selected = dm.aggregates.predictor_importance(query=pl.col("ModelID") == "a1").collect()
    assert selected.select("PredictorName", "Importance %").sort("PredictorName").rows() == [
        ("Customer.Age", 75.0),
        ("Customer.Income", 25.0),
        ("pyGroup", 0.0),
    ]
    assert _category_importance(dm, query=pl.col("Configuration") == "B") == [
        ("B", "GradientBoost", "Context Keys", 1, 20.0, 1.0),
        ("B", "GradientBoost", "Customer", 1, 80.0, 1.0),
    ]


@pytest.fixture
def unused_importance_dm():
    models, predictors = _configuration_importance_inputs()
    models = models.with_columns(
        ResponseCount=pl.when(pl.col("ModelID") == "b1").then(300.0).otherwise(pl.col("ResponseCount")),
    )
    # Action-level values of a1/a2 are zero: the global importance still
    # makes them trained and their predictors used.
    predictors = predictors.with_columns(
        FeatureImportance=pl.when(pl.col("ModelID").is_in(["a1", "a2"]))
        .then(0.0)
        .otherwise(pl.col("FeatureImportance")),
    )
    unused_customer = predictors.filter(pl.col("PredictorName") == "Customer.Age").with_columns(
        PredictorName=pl.lit("Customer.Unused"),
        FeatureImportance=pl.when(pl.col("ModelID") == "nb").then(100.0).otherwise(0.0),
    )
    unused_context_key = predictors.filter(pl.col("PredictorName") == "pyGroup").with_columns(
        PredictorName=pl.lit("pyTreatment"),
        FeatureImportance=pl.lit(0.0),
    )
    cold_model = models.filter(pl.col("ModelID") == "a1").with_columns(
        ModelID=pl.lit("cold"),
        Configuration=pl.lit("Cold"),
        ResponseCount=pl.lit(9999.0),
    )
    cold_predictors = predictors.filter(pl.col("ModelID") == "a1").with_columns(
        ModelID=pl.lit("cold"),
        # Classifier rows must not make an untrained model appear trained.
        FeatureImportance=pl.when(pl.col("EntryType") == "Classifier").then(100.0).otherwise(0.0),
    )
    cold_only = cold_predictors.filter(pl.col("PredictorName") == "Customer.Age").with_columns(
        PredictorName=pl.lit("Customer.ColdOnly"),
    )
    return ADMDatamart(
        model_df=pl.concat([models, cold_model], how="vertical_relaxed"),
        predictor_df=pl.concat(
            [predictors, unused_customer, unused_context_key, cold_predictors, cold_only],
            how="vertical_relaxed",
        ),
    )


_UNUSED_IN_A1 = [
    ("Customer.Unused", "Customer", 1, 100.0, 100.0),
    ("pyTreatment", "Context Keys", 1, 100.0, 100.0),
]


def test_unused_agb_predictors_configuration_importance(unused_importance_dm):
    dm = unused_importance_dm
    assert dm.aggregates.unused_agb_predictors().collect().rows() == [
        ("Customer.Unused", "Customer", 2, 100.0, 300.0),
        ("pyTreatment", "Context Keys", 2, 100.0, 300.0),
    ]
    assert dm.aggregates.unused_agb_predictors(query=pl.col("ModelID") == "a1").collect().rows() == _UNUSED_IN_A1
    assert dm.aggregates.unused_agb_predictors(query=pl.col("ModelID") == "cold").collect().rows() == []
    assert dm.aggregates.unused_agb_predictors(query=pl.col("ModelID") == "nb").collect().rows() == []
    assert dm.aggregates.unused_agb_predictors(query=pl.col("ModelID") == "absent").collect().rows() == []


def test_unused_agb_predictors_without_configuration_importance(configuration_importance_dm):
    dm = ADMDatamart(
        model_df=configuration_importance_dm.model_data,
        predictor_df=configuration_importance_dm.predictor_data.drop("ConfigurationImportance", "ConfigurationActive"),
    )
    # Action-level only: context keys look unused within every action model.
    assert dm.aggregates.unused_agb_predictors().collect().rows() == [("pyGroup", "Context Keys", 2, 100.0, 100.0)]


def test_poor_predictors_report_uses_configuration_importance(unused_importance_dm):
    report = Path(__file__).parents[2] / "pdstools" / "reports" / "HealthCheck.qmd"
    section = report.read_text(encoding="utf-8").split("## Poor Predictors Globally", 1)[1]
    cell = section.split("```{python}", 1)[1].split("```", 1)[0]
    report_utils = MagicMock()
    display = MagicMock()
    scope = {
        "datamart": unused_importance_dm,
        "pl": pl,
        "agb_query": pl.col("ModelTechnique") == "GradientBoost",
        "active_models_filter_expr": pl.col("ModelID") == "a1",
        "active_models_threshold_date_string": "2025-01-01",
        "agb_importance": unused_importance_dm.aggregates.predictor_importance().collect(),
        "report_utils": report_utils,
        "display": display,
        "style": MagicMock(),
        "loc": MagicMock(),
    }
    exec(compile(cell, str(report), "exec"), scope)
    report_utils.quarto_plot_exception.assert_not_called()
    display.assert_called_once()
    assert report_utils.create_metric_gttable.call_args.args[0].rows() == _UNUSED_IN_A1


@pytest.fixture(params=["NaiveBayes", None], ids=["explicit_nb", "legacy_nb"])
def nb_performance_dm(request):
    models = _models(
        [
            ("agb", "boosted", 10, 100, "GradientBoost"),
            ("nb", "bayes", 10, 100, request.param),
            ("old", "old_bayes", 10, 100, request.param),
        ],
    )
    rows = [
        ("agb", "pyGroup", 0.5),
        ("agb", "Customer.Age", 0.5),
        ("agb", "AgbOnly", 0.5),
        ("nb", "pyGroup", 0.8),
        ("nb", "Customer.Age", 0.5),
        ("old", "pyGroup", 0.5),
        ("old", "Customer.Age", 0.5),
        ("old", "OldOnly", 0.5),
    ]
    predictors = pl.DataFrame(rows, schema=["ModelID", "PredictorName", "Performance"], orient="row").lazy()
    predictors = predictors.with_columns(
        SnapshotTime=pl.lit("20250101"),
        EntryType=pl.lit("Active"),
        Type=pl.lit("numeric"),
        # Positive AGB predictor response counts must not admit AGB into NB plots.
        ResponseCount=pl.lit(100),
        Positives=pl.lit(10),
        BinIndex=pl.lit(1),
        BinResponseCount=pl.lit(100),
        BinPositives=pl.lit(10),
        BinSymbol=pl.lit("value"),
    )
    return ADMDatamart(model_df=models, predictor_df=predictors)


@pytest.mark.parametrize("agb_only", [False, True])
def test_nb_category_report_excludes_agb(nb_performance_dm, monkeypatch, agb_only):
    dm = nb_performance_dm
    active_filter = pl.col("ModelID") == "agb" if agb_only else pl.col("ModelID") != "old"
    report = Path(__file__).parents[2] / "pdstools" / "reports" / "HealthCheck.qmd"
    section = report.read_text(encoding="utf-8").split("## Importance of Naive Bayes Predictors by Category", 1)[1]
    cell = section.split("```{python}", 1)[1].split("```", 1)[0]
    plot = dm.plot.predictor_category_performance
    captured = []

    def capture_data(**kwargs):
        captured.append(plot(return_df=True, **kwargs))

    monkeypatch.setattr(dm.plot, "predictor_category_performance", capture_data)
    report_utils = MagicMock()
    exec(
        compile(cell, str(report), "exec"),
        {
            "datamart": dm,
            "active_models_filter_expr": active_filter,
            "nb_query": pl.col("ModelTechnique").fill_null("").ne("GradientBoost"),
            "report_utils": report_utils,
        },
    )
    report_utils.quarto_plot_exception.assert_not_called()
    assert len(captured) == 1
    if agb_only:
        assert captured[0].rows() == []
    else:
        assert captured[0].sort("PredictorCategory").select("PredictorCategory", "mean").rows() == [
            ("Context Keys", 80.0),
            ("Customer", 50.0),
        ]


@pytest.mark.parametrize("agb_only", [False, True])
def test_markdown_predictor_performance_is_nb_only(nb_performance_dm, agb_only):
    dm = nb_performance_dm
    active_filter = pl.col("ModelID") == "agb" if agb_only else pl.col("ModelID") != "old"
    preaggregates = dm.analysis.compute_health_check_preaggregates(active_filter=active_filter)
    # The estate-wide count still includes AgbOnly, while OldOnly is out of scope.
    assert preaggregates.predictor_count == 3
    overview = preaggregates.predictor_overview.select("PredictorName", "Mean", "Active in Models").sort(
        "PredictorName"
    )
    assert overview.rows() == ([] if agb_only else [("Customer.Age", 50.0, 1), ("pyGroup", 80.0, 1)])
    categories = preaggregates.predictor_categories.sort("PredictorCategory")
    assert categories.rows() == ([] if agb_only else [("Context Keys", 1, 80.0), ("Customer", 1, 50.0)])
    findings = dm.analysis.findings(active_filter=active_filter, preaggregates=preaggregates)
    poor = [f for f in findings if "poor_predictor_count" in f.data]
    assert [f.data for f in poor] == (
        [] if agb_only else [{"poor_predictor_count": 1, "total_predictors": 2, "threshold": 52.0}]
    )
    markdown = dm.analysis.markdown(active_filter=active_filter, preaggregates=preaggregates)
    assert "| Predictors | 3 |" in markdown
    if agb_only:
        assert "Naive Bayes Predictor Categories" not in markdown
        assert "poor performance" not in markdown
    else:
        assert "### Naive Bayes Predictor Categories" in markdown
        assert "1 of 2 Naive Bayes predictors (50%) have poor performance" in markdown


def test_predictor_findings_fallback_excludes_agb(nb_performance_dm):
    dm = nb_performance_dm
    # Exercise the direct path without precomputed summaries. Both NB models
    # are in scope here; AGB must not lower pyGroup's mean AUC or add AgbOnly.
    poor = [f for f in dm.analysis._check_predictors() if "poor_predictor_count" in f.data]
    assert [f.data for f in poor] == [{"poor_predictor_count": 2, "total_predictors": 3, "threshold": 52.0}]
    agb = ADMDatamart(
        model_df=dm.model_data.filter(pl.col("ModelTechnique") == "GradientBoost"),
        predictor_df=dm.predictor_data,
    )
    assert [f for f in agb.analysis._check_predictors() if "poor_predictor_count" in f.data] == []


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
