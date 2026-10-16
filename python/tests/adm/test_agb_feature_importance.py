"""Gradient Boost feature importance: overall-model split, aggregates, plots and report cells."""

import datetime
import re
from pathlib import Path
from unittest.mock import Mock

import plotly.graph_objects as go
import polars as pl
import pytest
from pdstools import ADMDatamart

SNAPSHOT = datetime.datetime(2025, 1, 1)


def _models() -> pl.LazyFrame:
    """Two GB configurations with an overall (empty context) row each, plus one NB model."""
    rows = [
        # ModelID, Configuration, Channel, Direction, Issue, Group, Name, Treatment, Technique
        ("g1", "AGBConfig", "Web", "Inbound", "Sales", "Cards", "A", "T1", "GradientBoost"),
        ("g2", "AGBConfig", "Web", "Inbound", "Sales", "Cards", "B", "T2", "GradientBoost"),
        ("g_overall", "AGBConfig", "", "", "", "", "", "", "GradientBoost"),
        ("e1", "EmailAGB", "Email", "Outbound", "Sales", "Cards", "A", "T1", "GradientBoost"),
        ("e_overall", "EmailAGB", "", "", "", "", "", "", "GradientBoost"),
        ("n1", "NBConfig", "Web", "Inbound", "Sales", "Cards", "A", None, "NaiveBayes"),
    ]
    columns = [
        "ModelID",
        "Configuration",
        "Channel",
        "Direction",
        "Issue",
        "Group",
        "Name",
        "Treatment",
        "ModelTechnique",
    ]
    df = pl.DataFrame(rows, schema=columns, orient="row").with_columns(
        SnapshotTime=pl.lit(SNAPSHOT),
        Positives=pl.lit(10.0),
        ResponseCount=pl.lit(100.0),
        Performance=pl.lit(0.7),
    )
    return df.lazy()


# FeatureImportance per (ModelID, PredictorName); GB bins carry no response counts.
_IMPORTANCE = {
    "g1": {"Param.Type": 70.0, "Customer.Age": 30.0},
    "g2": {"Param.Type": 60.0, "Customer.Age": 40.0},
    "g_overall": {"Param.Type": 80.0, "Customer.Age": 15.0, "IH.Clicks": 5.0},
    "e1": {"Param.Type": 50.0, "Customer.Age": 50.0},
    "e_overall": {"Param.Type": 0.0, "Customer.Age": 0.0},
    "n1": {"Customer.Age": 2.5},
}


def _predictors(with_importance: bool = True) -> pl.LazyFrame:
    rows = []
    for model_id, importances in _IMPORTANCE.items():
        is_nb = model_id == "n1"
        for name, importance in [*importances.items(), ("Classifier", 0.0)]:
            rows.append(
                {
                    "ModelID": model_id,
                    "PredictorName": name,
                    "EntryType": "Classifier" if name == "Classifier" else ("Active" if importance > 0 else "Inactive"),
                    "BinIndex": 1,
                    "BinPositives": 5.0 if is_nb else 0.0,
                    "BinNegatives": 95.0 if is_nb else 0.0,
                    "BinResponseCount": 100.0 if is_nb else 0.0,
                    "ResponseCount": 100.0 if is_nb else 0.0,
                    "Performance": 0.6,
                    "SnapshotTime": SNAPSHOT,
                    "Type": "symbolic",
                    "FeatureImportance": importance,
                }
            )
    df = pl.DataFrame(rows)
    if not with_importance:
        df = df.drop("FeatureImportance")
    return df.lazy()


def _ids(lf: pl.LazyFrame | None) -> set[str]:
    if lf is None:
        return set()
    return set(lf.select("ModelID").unique().collect()["ModelID"].to_list())


@pytest.fixture
def dm() -> ADMDatamart:
    return ADMDatamart(model_df=_models(), predictor_df=_predictors())


# ---- data split -------------------------------------------------------------


def test_overall_rows_split_from_model_data(dm):
    assert _ids(dm.model_data) == {"g1", "g2", "e1", "n1"}
    assert _ids(dm.agb_overall_model_data) == {"g_overall", "e_overall"}


def test_overall_predictor_rows_moved_out_of_predictor_data(dm):
    assert _ids(dm.predictor_data) == {"g1", "g2", "e1", "n1"}
    assert _ids(dm.agb_overall_predictor_data) == {"g_overall", "e_overall"}
    assert "g_overall" not in _ids(dm.combined_data)


def test_query_keeps_overall_rows_of_remaining_configurations():
    dm = ADMDatamart(model_df=_models(), predictor_df=_predictors(), query=pl.col("Channel") == "Web")
    assert _ids(dm.model_data) == {"g1", "g2", "n1"}
    assert _ids(dm.agb_overall_model_data) == {"g_overall"}
    assert _ids(dm.agb_overall_predictor_data) == {"g_overall"}


def test_no_overall_rows_when_model_technique_unknown():
    dm = ADMDatamart(model_df=_models().drop("ModelTechnique"), predictor_df=_predictors())
    assert _ids(dm.model_data) == {"g1", "g2", "g_overall", "e1", "e_overall", "n1"}
    assert dm.agb_overall_model_data.collect().height == 0
    assert dm.agb_overall_predictor_data.collect().height == 0


def test_overall_rows_dropped_for_configurations_without_instances():
    models = _models().filter(~pl.col("ModelID").is_in(["e1"]))
    dm = ADMDatamart(model_df=models, predictor_df=_predictors())
    assert _ids(dm.agb_overall_model_data) == {"g_overall"}
    assert _ids(dm.agb_overall_predictor_data) == {"g_overall"}


def test_rebuild_with_overall_rows_follows_remaining_configurations(dm):
    """The Health Check rebuilds the datamart from filtered model data via with_model_data."""
    rebuilt = dm.with_model_data(dm.model_data.filter(pl.col("Configuration") != "EmailAGB"))
    assert _ids(rebuilt.model_data) == {"g1", "g2", "n1"}
    assert _ids(rebuilt.agb_overall_model_data) == {"g_overall"}
    assert rebuilt.aggregates.agb_feature_importance().select(
        "PredictorName", "FeatureImportance"
    ).collect().rows() == [
        ("Param.Type", 80.0),
        ("Customer.Age", 15.0),
        ("IH.Clicks", 5.0),
    ]


def test_with_model_data_without_predictor_data():
    dm = ADMDatamart(model_df=_models())
    rebuilt = dm.with_model_data(dm.model_data.filter(pl.col("Channel") == "Web"))
    assert _ids(rebuilt.model_data) == {"g1", "g2", "n1"}
    assert _ids(rebuilt.agb_overall_model_data) == {"g_overall"}
    assert rebuilt.predictor_data is None
    assert rebuilt.agb_overall_predictor_data is None


def test_save_data_round_trip_keeps_overall_rows_of_selected_configurations(dm, tmp_path):
    model_path, predictor_path = dm.save_data(tmp_path, selected_model_ids=["g1", "g2"])
    reloaded = ADMDatamart.from_ds_export(model_filename=str(model_path), predictor_filename=str(predictor_path))
    assert _ids(reloaded.model_data) == {"g1", "g2"}
    assert _ids(reloaded.agb_overall_model_data) == {"g_overall"}
    assert _ids(reloaded.agb_overall_predictor_data) == {"g_overall"}
    assert _ids(reloaded.predictor_data) == {"g1", "g2"}


def test_save_data_without_selection_keeps_all_overall_rows(dm, tmp_path):
    model_path, predictor_path = dm.save_data(tmp_path)
    reloaded = ADMDatamart.from_ds_export(model_filename=str(model_path), predictor_filename=str(predictor_path))
    assert _ids(reloaded.model_data) == {"g1", "g2", "e1", "n1"}
    assert _ids(reloaded.agb_overall_model_data) == {"g_overall", "e_overall"}


def test_apply_predictor_categorization_updates_overall_predictors(dm):
    dm.apply_predictor_categorization({"Interaction History": "IH."})
    categories = dict(
        dm.agb_overall_predictor_data.filter(pl.col("ModelID") == "g_overall")
        .select(pl.col("PredictorName").cast(pl.String), pl.col("PredictorCategory").cast(pl.String))
        .unique()
        .collect()
        .iter_rows()
    )
    assert categories["IH.Clicks"] == "Interaction History"
    assert categories["Param.Type"] == "Param"


# ---- aggregates -------------------------------------------------------------


def test_agb_feature_importance_values(dm):
    out = dm.aggregates.agb_feature_importance(query=pl.col("Configuration") == "AGBConfig").collect()
    assert out.select("ModelID", "PredictorName", "PredictorCategory", "EntryType", "FeatureImportance").rows() == [
        ("g_overall", "Param.Type", "Param", "Active", 80.0),
        ("g_overall", "Customer.Age", "Customer", "Active", 15.0),
        ("g_overall", "IH.Clicks", "IH", "Active", 5.0),
    ]


def test_agb_feature_importance_all_configurations(dm):
    out = dm.aggregates.agb_feature_importance().collect()
    assert out.group_by("Configuration").agg(pl.col("FeatureImportance").sum()).sort("Configuration").rows() == [
        ("AGBConfig", 100.0),
        ("EmailAGB", 0.0),
    ]


def test_agb_feature_importance_empty_without_feature_importance():
    dm = ADMDatamart(model_df=_models(), predictor_df=_predictors(with_importance=False))
    out = dm.aggregates.agb_feature_importance().collect()
    assert out.height == 0
    assert out.columns == [
        "ModelID",
        "Configuration",
        "PredictorName",
        "PredictorCategory",
        "EntryType",
        "Type",
        "FeatureImportance",
    ]


# ---- box plot over model instances ----------------------------------------


def test_feature_importance_boxplot_uses_gradient_boost_instances_only(dm):
    out = dm.plot.predictor_performance(metric="FeatureImportance", return_df=True)
    stats = {
        row["PredictorName"]: (row["median"], row["mean"], row["min"], row["max"]) for row in out.iter_rows(named=True)
    }
    # g1, g2 and e1 only: the NB model (2.5) and the overall models (80/15/5, 0/0) are excluded
    assert stats == {
        "Param.Type": (60.0, 60.0, 50.0, 70.0),
        "Customer.Age": (40.0, 40.0, 30.0, 50.0),
    }


def test_feature_importance_boxplot_query_and_top_n(dm):
    out = dm.plot.predictor_performance(
        metric="FeatureImportance",
        top_n=1,
        query=pl.col("Configuration") == "AGBConfig",
        return_df=True,
    )
    assert out.select("PredictorName", "median", "min", "max").rows() == [("Param.Type", 65.0, 60.0, 70.0)]


def test_feature_importance_boxplot_figure(dm):
    fig = dm.plot.predictor_performance(metric="FeatureImportance")
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "Feature Importance (%)"
    assert sorted(trace.y[0] for trace in fig.data) == ["Customer.Age", "Param.Type"]


def test_feature_importance_boxplot_requires_feature_importance():
    dm = ADMDatamart(model_df=_models(), predictor_df=_predictors(with_importance=False))
    with pytest.raises(ValueError, match="FeatureImportance"):
        dm.plot.predictor_performance(metric="FeatureImportance")


# ---- overall model bar chart ----------------------------------------------


def test_agb_feature_importance_plot_data(dm):
    out = dm.plot.agb_feature_importance(top_n=2, return_df=True).collect()
    # EmailAGB has only zero importances and is left out
    assert out.select("Configuration", "PredictorName", "FeatureImportance").rows() == [
        ("AGBConfig", "Param.Type", 80.0),
        ("AGBConfig", "Customer.Age", 15.0),
    ]
    assert dm.plot.agb_feature_importance(top_n=None, return_df=True).collect().height == 3


def test_agb_feature_importance_plot_figure(dm):
    fig = dm.plot.agb_feature_importance(query=pl.col("Configuration") == "AGBConfig")
    assert isinstance(fig, go.Figure)
    assert all(trace.orientation == "h" for trace in fig.data)
    assert {trace.name for trace in fig.data} == {"Param", "Customer", "IH"}
    assert "AGBConfig: Param 80.0% · Customer 15.0% · IH 5.0%" in fig.layout.title.text


def test_agb_feature_importance_plot_none_without_importance(dm):
    assert dm.plot.agb_feature_importance(query=pl.col("Configuration") == "EmailAGB") is None


# ---- Health Check report cells ----------------------------------------------


def _report_cell(marker: str):
    template = Path(__file__).parents[2] / "pdstools" / "reports" / "HealthCheck.qmd"
    cells = re.findall(r"```\{python\}\n(.*?)\n```", template.read_text(encoding="utf-8"), re.DOTALL)
    matches = [cell for cell in cells if marker in cell]
    assert len(matches) == 1
    return compile(matches[0], str(template), "exec")


@pytest.fixture
def shown_figures(monkeypatch):
    shown: list[go.Figure] = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: shown.append(self))
    return shown


def test_report_gradient_boost_feature_importance_cell(dm, shown_figures):
    report_utils = Mock()
    exec(
        _report_cell("datamart.aggregates.agb_feature_importance()"),
        {"datamart": dm, "pl": pl, "report_utils": report_utils, "configurations_max": 10},
    )
    assert len(shown_figures) == 1
    assert "AGBConfig" in shown_figures[0].layout.title.text
    report_utils.quarto_callout_info.assert_called_once_with(
        "The overall Gradient Boost model has no active predictors for: EmailAGB."
    )


def test_report_predictor_importance_cell_includes_gradient_boost(dm, shown_figures):
    report_utils = Mock()
    exec(
        _report_cell("gb_feature_importance_available"),
        {
            "datamart": dm,
            "pl": pl,
            "report_utils": report_utils,
            "configurations_max": 10,
            "datamart_all_columns": dm.combined_data.collect_schema().names(),
            "active_models_filter_expr": pl.lit(True),
            "predictor_analysis_query": pl.col("ResponseCountBin") > 0,
        },
    )
    report_utils.quarto_plot_exception.assert_not_called()
    titles = sorted(fig.layout.title.text for fig in shown_figures)
    assert len(titles) == 3
    assert sum("Feature importance of the Gradient Boost model instances" in t for t in titles) == 2
