"""Verify sample and supplied data stay separate in the Health Check template."""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import plotly.express as px
import polars as pl
import pytest
from pdstools.adm.Plots._helpers import add_metric_limit_lines
from pdstools.utils import pega_template
from pdstools.utils.plot_utils import fig_update_facet


@pytest.fixture(scope="module")
def data_cells():
    template = Path(__file__).parents[2] / "pdstools" / "reports" / "HealthCheck.qmd"
    cells = re.findall(r"```\{python\}\n(.*?)\n```", template.read_text(encoding="utf-8"), re.DOTALL)
    return [
        compile(next(cell for cell in cells if marker in cell), str(template), "exec")
        for marker in ("# Load datamart", "prediction = None\nif prediction_file_path:")
    ]


@pytest.fixture(scope="module")
def action_hierarchy_cell():
    template = Path(__file__).parents[2] / "pdstools" / "reports" / "HealthCheck.qmd"
    source = template.read_text(encoding="utf-8")
    start = source.index('if "Issue" in datamart_all_columns and "Group" in datamart_all_columns:')
    end = source.index("\n```", start)
    return compile(source[start:end], str(template), "exec")


@pytest.fixture(scope="module")
def bubble_chart_cell():
    template = Path(__file__).parents[2] / "pdstools" / "reports" / "HealthCheck.qmd"
    cells = re.findall(r"```\{python\}\n(.*?)\n```", template.read_text(encoding="utf-8"), re.DOTALL)
    cell = next(cell for cell in cells if "# Check number of unique Configuration/Channel/Direction" in cell)
    return compile(cell, str(template), "exec")


@pytest.mark.parametrize(
    ("model_path", "prediction_path", "raw_prediction", "expected_prediction"),
    [
        (None, None, False, sentinel.sample_prediction),
        ("models.parquet", None, False, None),
        ("models.parquet", "predictions.parquet", True, sentinel.supplied_prediction),
        ("models.parquet", "predictions.parquet", False, sentinel.supplied_prediction),
        (None, "predictions.parquet", True, sentinel.supplied_prediction),
    ],
    ids=["sample", "model-only", "raw-prediction", "processed-prediction", "sample-with-explicit-prediction"],
)
def test_healthcheck_prediction_source(data_cells, model_path, prediction_path, raw_prediction, expected_prediction):
    bundle = SimpleNamespace(datamart=sentinel.sample_datamart, prediction=sentinel.sample_prediction)
    demo_data = Mock(return_value=bundle)
    datamart_cls = Mock()
    datamart_cls.from_ds_export.return_value = sentinel.supplied_datamart
    prediction_cls = Mock(return_value=sentinel.supplied_prediction)
    prediction_cls.from_processed_data.return_value = sentinel.supplied_prediction
    prediction_frame = Mock()
    prediction_frame.collect_schema.return_value.names.return_value = ["pyModelType"] if raw_prediction else []
    read_ds_export = Mock(return_value=prediction_frame)
    predictor_path = "predictors.parquet" if model_path else None
    namespace = {
        "ADMDatamart": datamart_cls,
        "Prediction": prediction_cls,
        "demo_data": demo_data,
        "model_file_path": model_path,
        "predictor_file_path": predictor_path,
        "prediction_file_path": prediction_path,
        "query": None,
        "read_ds_export": read_ds_export,
    }

    for cell in data_cells:
        exec(cell, namespace)

    assert namespace["datamart"] is (sentinel.supplied_datamart if model_path else sentinel.sample_datamart)
    assert namespace["prediction"] is expected_prediction
    if model_path:
        demo_data.assert_not_called()
        datamart_cls.from_ds_export.assert_called_once_with(
            model_filename=model_path,
            predictor_filename=predictor_path,
            base_path=".",
            query=None,
        )
    else:
        demo_data.assert_called_once_with()
        datamart_cls.from_ds_export.assert_not_called()

    if prediction_path:
        read_ds_export.assert_called_once_with(prediction_path)
        if raw_prediction:
            prediction_cls.assert_called_once_with(prediction_frame)
            prediction_cls.from_processed_data.assert_not_called()
        else:
            prediction_cls.assert_not_called()
            prediction_cls.from_processed_data.assert_called_once_with(prediction_frame)
    else:
        read_ds_export.assert_not_called()
        prediction_cls.assert_not_called()
        prediction_cls.from_processed_data.assert_not_called()


def test_action_hierarchy_collapses_whitespace_group_variants(action_hierarchy_cell):
    report_utils = Mock()
    namespace = {
        "pl": pl,
        "datamart": SimpleNamespace(
            model_data=pl.DataFrame(
                {
                    "Issue": ["Sales", "Sales", "Sales"],
                    "Group": ["CreditCards", "CreditCards\u00a0", "Loans"],
                    "ModelID": ["model-1", "model-2", "model-3"],
                    "Name": ["EverydayCard", "TravelCard", "PersonalLoan"],
                }
            )
            .with_columns(pl.col("Group").cast(pl.Categorical))
            .lazy()
        ),
        "datamart_all_columns": ["Issue", "Group"],
        "active_models_filter_expr": pl.lit(True),
        "active_models_threshold_date_string": "2026-01-01",
        "report_utils": report_utils,
        "display": Mock(),
    }

    exec(action_hierarchy_cell, namespace)

    hierarchy = report_utils.create_metric_gttable.call_args.args[0]
    assert hierarchy.select("Group", "Models").rows() == [("CreditCards", 2), ("Loans", 1)]


@pytest.mark.parametrize(
    ("facet_count", "expected_height", "configurations_max", "technique", "title_detail"),
    [
        (1, 465, 100, "GradientBoost", "Configuration and Channel/Direction"),
        (3, 680, 100, "GradientBoost", "Configuration and Channel/Direction"),
        (8, 1110, 100, "GradientBoost", "Configuration and Channel/Direction"),
        (3, 680, 1, "GradientBoost", "Channel/Direction and Model Technique"),
        (3, 680, 1, None, "Channel/Direction"),
    ],
)
def test_bubble_chart_compacts_facets_without_overlapping_labels(
    bubble_chart_cell, monkeypatch, facet_count, expected_height, configurations_max, technique, title_detail
):
    configs = [f"Configuration{i}" for i in range(facet_count)]
    channels = [f"Channel{i}" for i in range(facet_count)]
    plot_data = pl.DataFrame(
        {
            "Performance": [60 + i for i in range(facet_count)],
            "SuccessRate": [0.01] * facet_count,
            "Facet": [f"{config}/{channel}/Inbound" for config, channel in zip(configs, channels, strict=True)],
        }
    )
    fig = px.scatter(
        plot_data,
        x="Performance",
        y="SuccessRate",
        color="Performance",
        facet_col="Facet",
        facet_col_wrap=2,
    )
    add_metric_limit_lines(fig)
    monkeypatch.setattr(type(fig), "show", lambda self: None)
    report_utils = Mock()
    datamart = SimpleNamespace(
        model_data=pl.DataFrame(
            {
                "Configuration": configs,
                "Channel": channels,
                "Direction": ["Inbound"] * facet_count,
                "ModelTechnique": [technique] * facet_count,
            }
        ).lazy(),
        plot=SimpleNamespace(bubble_chart=Mock(return_value=fig)),
    )
    namespace = {
        "pl": pl,
        "datamart": datamart,
        "active_models_filter_expr": pl.lit(True),
        "configurations_max": configurations_max,
        "active_models_threshold_date_string": "2026-01-01",
        "report_utils": report_utils,
        "fig_update_facet": fig_update_facet,
        "pega_template": pega_template,
    }

    exec(bubble_chart_cell, namespace)

    report_utils.quarto_plot_exception.assert_not_called()
    assert fig.layout.height == expected_height
    assert fig.layout.width == 900
    assert fig.layout.title.text.startswith(f"ADM Models by {title_detail}")
    assert len(fig.data) == facet_count
    assert len(fig.layout.shapes) == 7 * facet_count
    html = fig.to_html(full_html=False, include_plotlyjs=False)
    graph_heights = re.findall(r'class="plotly-graph-div"[^>]*style="[^"]*\bheight:\s*(\d+)px', html)
    assert graph_heights == [str(expected_height)]
    titles = [a for a in fig.layout.annotations if a.xref == "paper" and a.yref == "paper"]
    assert len(titles) == facet_count
    assert all(a.visible is not False for a in titles)

    limits = [a for a in fig.layout.annotations if a.yref and a.yref.endswith(" domain")]
    assert len(limits) == 4 * facet_count
    bottom_labels = 0
    for annotation in limits:
        axis_name = annotation.yref.split(" ", 1)[0].replace("y", "yaxis", 1)
        if fig.layout[axis_name].domain[0] == 0:
            assert annotation.visible is not False
            bottom_labels += 1
        else:
            assert annotation.visible is False
    assert bottom_labels == 4 * (1 if facet_count % 2 else 2)
