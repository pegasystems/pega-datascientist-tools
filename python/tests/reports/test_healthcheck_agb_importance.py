"""Exact-value regressions for AGB importance partitioning in the report."""

import re
from html import unescape
from pathlib import Path
from unittest.mock import Mock

import plotly.graph_objects as go
import polars as pl
import pytest
from pdstools import ADMDatamart


@pytest.fixture
def report_cells():
    template = Path(__file__).parents[2] / "pdstools" / "reports" / "HealthCheck.qmd"
    return re.findall(r"```\{python\}\n(.*?)\n```", template.read_text(encoding="utf-8"), re.DOTALL)


@pytest.fixture
def title_helper_cell(report_cells):
    return next(cell for cell in report_cells if "def predictor_importance_title(" in cell)


@pytest.fixture
def agb_report_cell(report_cells, title_helper_cell):
    cell = next(cell for cell in report_cells if 'agb_query = pl.col("ModelTechnique")' in cell)
    return compile(title_helper_cell + "\n" + cell, "HealthCheck.qmd", "exec")


@pytest.fixture
def nb_report_cell(report_cells, title_helper_cell):
    cell = next(cell for cell in report_cells if "nb_importance_groups = " in cell)
    return compile(title_helper_cell + "\n" + cell, "HealthCheck.qmd", "exec")


@pytest.fixture
def importance_dm():
    models = pl.DataFrame(
        {
            "ModelID": ["a1", "a2", "b1", "c1", "untrained", "inactive", "nb"],
            "Name": ["a1", "a2", "b1", "c1", "untrained", "inactive", "nb"],
            "Configuration": ["A", "A", "B", "C", "D", "E", "N"],
            "Channel": ["Web", "Web", "Web", "Email", "Web", "Web", "Web"],
            "Direction": ["Inbound", "Inbound", "Inbound", None, "Inbound", "Inbound", "Inbound"],
            "ModelTechnique": ["GradientBoost"] * 6 + ["NaiveBayes"],
            "SnapshotTime": "20260101",
            "ResponseCount": 1000,
            "Positives": 100,
            "Performance": 0.6,
        }
    ).lazy()
    predictors = pl.DataFrame(
        {
            "ModelID": ["a1", "a1", "a2", "a2", "b1", "b1", "c1", "c1", "untrained", "inactive", "nb"],
            "PredictorName": [
                "Age",
                "Param.Tier",
                "Age",
                "Param.Tier",
                "Age",
                "IH.Clicks",
                "Age",
                "IH.Clicks",
                "Age",
                "Age",
                "Age",
            ],
            "FeatureImportance": [90.0, 10.0, 50.0, 50.0, 10.0, 90.0, 20.0, 80.0, 0.0, 100.0, 100.0],
            "EntryType": "Active",
            "SnapshotTime": "20260101",
            "BinIndex": 1,
            "BinType": "NONE",
            "BinSymbol": "",
            "BinResponseCount": 0,
            "BinPositives": 0,
            "Type": "numeric",
            "Performance": 0.5,
            "ResponseCount": 1000,
            "Positives": 0,
        }
    ).lazy()
    return ADMDatamart(model_df=models, predictor_df=predictors)


@pytest.mark.parametrize("configurations_max", [3, 2])
def test_agb_report_partitioning(agb_report_cell, importance_dm, configurations_max, monkeypatch):
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self: shown.append(self))
    report_utils = Mock()
    namespace = {
        "pl": pl,
        "datamart": importance_dm,
        "trend_datamart": importance_dm,
        "active_models_filter_expr": pl.col("Name") != "inactive",
        "configurations_max": configurations_max,
        "report_utils": report_utils,
    }

    exec(agb_report_cell, namespace)

    report_utils.quarto_plot_exception.assert_not_called()
    # The first figure is the existing category overview; the rest are top predictors.
    shares = [
        {name: value for trace in fig.data for name, value in zip(trace.y, trace.x, strict=True)} for fig in shown[1:]
    ]
    titles = [fig.layout.title.text for fig in shown[1:]]
    assert all(fig.layout.title.font.size == 14 for fig in shown[1:])
    assert all(
        fig.layout.margin.t == 80 + 20 * title.count("<br>") for fig, title in zip(shown[1:], titles, strict=True)
    )
    if configurations_max == 3:
        assert shares == [
            pytest.approx({"Age": 70.0, "Param.Tier": 30.0}),
            pytest.approx({"Age": 10.0, "IH.Clicks": 90.0}),
            pytest.approx({"Age": 20.0, "IH.Clicks": 80.0}),
        ]
        assert titles == [
            f"Predictor Importance: {configuration}<br>"
            f"Channel={channel}, Direction={direction}, ModelTechnique=GradientBoost"
            for configuration, channel, direction in [
                ("A", "Web", "Inbound"),
                ("B", "Web", "Inbound"),
                ("C", "Email", None),
            ]
        ]
        report_utils.quarto_callout_info.assert_not_called()
    else:
        assert shares == [
            pytest.approx({"Age": 20.0, "IH.Clicks": 80.0}),
            pytest.approx({"Age": 50.0, "Param.Tier": 20.0, "IH.Clicks": 30.0}),
        ]
        assert titles == [
            "Predictor Importance: C<br>Channel=Email, Direction=None, ModelTechnique=GradientBoost",
            "Predictor Importance (configurations combined)<br>"
            "Channel=Web, Direction=Inbound, ModelTechnique=GradientBoost",
        ]
        report_utils.quarto_callout_info.assert_called_once()
        assert "(3) AGB model configurations" in report_utils.quarto_callout_info.call_args.args[0]


@pytest.mark.parametrize("configurations_max", [3, 2])
def test_nb_report_titles_and_fallback(nb_report_cell, importance_dm, configurations_max, monkeypatch):
    dm = ADMDatamart(
        model_df=importance_dm.model_data.filter(~pl.col("Name").is_in(["nb", "inactive", "untrained"])).with_columns(
            pl.lit("NaiveBayes").alias("ModelTechnique"),
        ),
        predictor_df=importance_dm.predictor_data,
    )
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self: shown.append(self))
    report_utils = Mock()
    namespace = {
        "pl": pl,
        "datamart": dm,
        "active_models_filter_expr": ~pl.col("Name").is_in(["inactive", "untrained"]),
        "predictor_analysis_query": pl.col("ResponseCountBin") > 200,
        "configurations_max": configurations_max,
        "report_utils": report_utils,
    }

    exec(nb_report_cell, namespace)

    report_utils.quarto_plot_exception.assert_not_called()
    titles = [fig.layout.title.text for fig in shown]
    if configurations_max == 3:
        assert titles == [
            f"Predictor Importance: {configuration}<br>"
            f"Channel={channel}, Direction={direction}, ModelTechnique=NaiveBayes"
            for configuration, channel, direction in [
                ("A", "Web", "Inbound"),
                ("B", "Web", "Inbound"),
                ("C", "Email", None),
            ]
        ]
        report_utils.quarto_callout_info.assert_not_called()
    else:
        assert titles == [
            "Predictor Importance: C<br>Channel=Email, Direction=None, ModelTechnique=NaiveBayes",
            "Predictor Importance (configurations combined)<br>"
            "Channel=Web, Direction=Inbound, ModelTechnique=NaiveBayes",
        ]
        report_utils.quarto_callout_info.assert_called_once()
    assert all(fig.layout.title.font.size == 14 for fig in shown)


def test_importance_title_wraps_and_escapes_configuration_names(title_helper_cell):
    namespace = {}
    exec(title_helper_cell, namespace)
    title = namespace["predictor_importance_title"](
        {"Channel": "Web", "Direction": "Inbound", "ModelTechnique": "GradientBoost"},
        ["A<test>" + "B" * 100],
    )
    assert "A&lt;test&gt;" in title
    assert "<sup>" not in title
    assert all(len(unescape(line)) <= 65 for line in title.split("<br>"))


@pytest.mark.parametrize("model_name", ["untrained", "nb", "missing"])
def test_agb_report_no_importance(agb_report_cell, importance_dm, model_name, monkeypatch):
    show = Mock()
    monkeypatch.setattr(go.Figure, "show", show)
    report_utils = Mock()

    exec(
        agb_report_cell,
        {
            "pl": pl,
            "datamart": importance_dm,
            "trend_datamart": importance_dm,
            "active_models_filter_expr": pl.col("Name") == model_name,
            "configurations_max": 3,
            "report_utils": report_utils,
        },
    )

    show.assert_not_called()
    report_utils.quarto_plot_exception.assert_not_called()
    report_utils.quarto_callout_info.assert_called_once()
