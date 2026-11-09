"""Verify sample and supplied data stay separate in the Health Check template."""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import polars as pl
import pytest


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
