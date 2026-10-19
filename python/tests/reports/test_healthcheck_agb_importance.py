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
        # A/B/C become NB; D/E stay AGB-only and must not count towards the NB facets.
        model_df=importance_dm.model_data.filter(pl.col("Name") != "nb").with_columns(
            pl.when(pl.col("Configuration").is_in(["A", "B", "C"]))
            .then(pl.lit("NaiveBayes"))
            .otherwise(pl.col("ModelTechnique"))
            .alias("ModelTechnique"),
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
        assert "(3) Naive Bayes model configurations" in report_utils.quarto_callout_info.call_args.args[0]
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


@pytest.fixture
def poor_predictor_cells(report_cells):
    nb_cell = next(cell for cell in report_cells if 'title="Overview of Poor Naive Bayes Predictors"' in cell)
    agb_cell = next(cell for cell in report_cells if 'title="Unused Adaptive Gradient Boosting Predictors"' in cell)
    return compile(nb_cell, "HealthCheck.qmd", "exec"), compile(agb_cell, "HealthCheck.qmd", "exec")


@pytest.fixture
def poor_predictor_dm(importance_dm):
    extra = pl.DataFrame(
        {
            "ModelID": ["a1", "a2", "b1", "a1", "c1", "inactive", "nb"],
            "PredictorName": ["Unused.X", "Unused.X", "Unused.X", "Partly.Z", "Partly.Z", "Unused.Y", "Poor.NB"],
            "FeatureImportance": [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
            "EntryType": ["Inactive", "Inactive", "Inactive", "Inactive", "Active", "Inactive", "Active"],
        }
    )
    predictors = (
        pl.concat(
            [
                importance_dm.predictor_data.collect()
                .select("ModelID", "PredictorName", "FeatureImportance", "EntryType")
                .cast({"ModelID": pl.String, "PredictorName": pl.String, "EntryType": pl.String}),
                extra,
            ],
            how="vertical_relaxed",
        )
        .with_columns(
            # AGB predictors carry Performance 0 (no univariate AUC); NB "Age" is strong, "Poor.NB" weak.
            pl.when(pl.col("ModelID") != "nb")
            .then(0.0)
            .when(pl.col("PredictorName") == "Poor.NB")
            .then(0.505)
            .otherwise(0.7)
            .alias("Performance"),
            SnapshotTime=pl.lit("20260101"),
            BinIndex=pl.lit(1),
            BinType=pl.lit("NONE"),
            BinSymbol=pl.lit(""),
            BinResponseCount=pl.lit(0),
            BinPositives=pl.lit(0),
            Type=pl.lit("numeric"),
            ResponseCount=pl.lit(1000),
            Positives=pl.lit(0),
        )
        .lazy()
    )
    return ADMDatamart(model_df=importance_dm.model_data, predictor_df=predictors)


def _run_poor_predictor_cells(cells, dm, active_models_filter_expr):
    from great_tables import html, loc, style
    from pdstools.utils import report_utils as real_report_utils
    from pdstools.utils.metric_limits import MetricLimits

    displayed = []
    report_utils = Mock(create_metric_gttable=real_report_utils.create_metric_gttable)
    namespace = {
        "pl": pl,
        "datamart": dm,
        "active_models_filter_expr": active_models_filter_expr,
        "active_models_threshold_date_string": "2026-01-01",
        "agb_query": pl.col("ModelTechnique") == "GradientBoost",
        "agb_importance": pl.DataFrame({"x": [1]}),
        "agb_unavailable": "no AGB importance",
        "MetricLimits": MetricLimits,
        "report_utils": report_utils,
        "display": displayed.append,
        "html": html,
        "loc": loc,
        "style": style,
    }
    for cell in cells:
        exec(cell, namespace)
    report_utils.quarto_plot_exception.assert_not_called()
    return displayed, report_utils


def test_poor_predictors_split_by_technique(poor_predictor_cells, poor_predictor_dm):
    displayed, report_utils = _run_poor_predictor_cells(
        poor_predictor_cells, poor_predictor_dm, pl.col("Name") != "inactive"
    )

    nb_table, agb_table = (gt._tbl_data for gt in displayed)
    # AGB zeros no longer drag NB "Age" (AUC 70) or AGB-only predictors into the NB table.
    assert nb_table.select("PredictorName", "Mean").to_dicts() == [{"PredictorName": "Poor.NB", "Mean": 50.5}]
    # Unused.Y is only in the inactive model; untrained models carry no importance.
    assert agb_table.to_dicts() == [{"PredictorName": "Unused.X", "PredictorCategory": "Unused", "Configurations": 2}]
    report_utils.quarto_callout_info.assert_not_called()


def test_unused_agb_predictors_none(poor_predictor_cells, poor_predictor_dm):
    displayed, report_utils = _run_poor_predictor_cells(
        poor_predictor_cells[1:], poor_predictor_dm, pl.col("Name").is_in(["c1", "nb"])
    )

    assert displayed == []
    report_utils.quarto_callout_info.assert_called_once_with(
        "Every predictor is used by at least one Adaptive Gradient Boosting model."
    )


def test_unused_agb_predictors_without_agb_importance(poor_predictor_cells, poor_predictor_dm):
    agb_cell = poor_predictor_cells[1]
    report_utils = Mock()
    exec(
        agb_cell,
        {"pl": pl, "agb_importance": None, "agb_unavailable": "no AGB importance", "report_utils": report_utils},
    )
    report_utils.quarto_callout_info.assert_called_once_with("no AGB importance")


def test_predictor_count_is_naive_bayes_only(report_cells, poor_predictor_dm, monkeypatch):
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self: shown.append(self))
    count_cell = next(cell for cell in report_cells if "datamart.plot.predictor_count(" in cell)
    report_utils = Mock()
    exec(
        compile(count_cell, "HealthCheck.qmd", "exec"),
        {
            "pl": pl,
            "datamart": poor_predictor_dm,
            "active_models_filter_expr": pl.lit(True),
            "active_models_threshold_date_string": "2026-01-01",
            "nb_query": pl.col("ModelTechnique").fill_null("").ne("GradientBoost"),
            "report_utils": report_utils,
        },
    )

    report_utils.quarto_plot_exception.assert_not_called()
    (fig,) = shown
    assert fig.layout.title.text.startswith("Naive Bayes Predictor Counts")
    labels = {label for trace in fig.data for label in trace.y}
    assert labels == {"N / Active / numeric", "N / Active / Overall"}
