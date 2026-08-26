"""Integration test for the batch report generator (scripts/batch_healthcheck.py).

Runs the actual script as a subprocess with sample data, verifying it
produces valid HTML healthcheck reports, model reports, and Excel exports.
"""

import importlib.util
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
SCRIPT = Path(__file__).parent.parent.parent.parent / "scripts" / "batch_healthcheck.py"
SPEC = importlib.util.spec_from_file_location("batch_healthcheck", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load batch healthcheck script from {SCRIPT}")
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def test_find_data_directories_discovers_canonical_prediction(tmp_path):
    hc_dir = tmp_path / "Dataset" / "HC"
    hc_dir.mkdir(parents=True)
    model = hc_dir / "PR_DATA_DM_ADMMART_MDL_FACT.parquet"
    predictor = hc_dir / "PR_DATA_DM_ADMMART_PRED.parquet"
    prediction = hc_dir / "PR_DATA_DM_SNAPSHOTS.parquet"
    for path in (model, predictor, prediction):
        path.touch()

    datasets = batch.find_data_directories(tmp_path)

    assert datasets == [
        {
            "name": "Dataset",
            "data_dir": hc_dir,
            "model_file": model,
            "predictor_file": predictor,
            "prediction_file": prediction,
        }
    ]


def test_print_report_size_comparison_highlights_smaller_embed(capsys):
    batch._print_report_size_comparison("HC", cdn_mb=2.0, embed_mb=1.0)

    captured = capsys.readouterr()

    assert "HC size: CDN 2.0 MB vs embed 1.0 MB (0.5x)" in captured.out
    assert "HC full-embed output is smaller than CDN output" in captured.out


def test_select_interesting_models_excludes_empty_active_ranges():
    datamart = MagicMock()
    datamart.predictor_data = pl.LazyFrame(
        {
            "ModelID": ["invalid", "invalid", "valid", "valid"],
            "BinType": ["SYMBOLIC", "NONE", "SYMBOLIC", "NONE"],
            "EntryType": ["Active", "Classifier", "Active", "Classifier"],
        }
    )
    datamart.combined_data = pl.LazyFrame(
        {
            "ModelID": ["invalid", "valid"],
            "Channel": ["Web", "Web"],
            "Direction": ["Inbound", "Inbound"],
            "Issue": ["Offer", "Offer"],
            "Positives": [300, 300],
            "ResponseCount": [1500, 1500],
            "Performance": [0.9, 0.8],
        }
    )
    datamart.active_ranges.return_value = pl.LazyFrame(
        {
            "ModelID": ["invalid", "valid"],
            "AUC_ActiveRange": [None, 0.7],
            "idx_min": [4, 1],
            "idx_max": [4, 3],
        }
    )

    assert batch.select_interesting_models(datamart) == ["valid"]


def test_print_dataset_paths_omits_missing_optional_files(capsys):
    batch._print_dataset_paths(
        {
            "Data_Dir": "/tmp/data/HC",
            "Model_File": "/tmp/data/HC/PR_DATA_DM_ADMMART_MDL_FACT.parquet",
            "Predictor_File": None,
            "Prediction_File": "/tmp/data/HC/PR_DATA_DM_SNAPSHOTS.parquet",
        }
    )

    captured = capsys.readouterr()

    assert "Data directory: /tmp/data/HC" in captured.out
    assert "Model file: /tmp/data/HC/PR_DATA_DM_ADMMART_MDL_FACT.parquet" in captured.out
    assert "Prediction file: /tmp/data/HC/PR_DATA_DM_SNAPSHOTS.parquet" in captured.out
    assert "Predictor file" not in captured.out


def test_process_dataset_passes_prediction_and_canonical_paths(tmp_path):
    model = tmp_path / "PR_DATA_DM_ADMMART_MDL_FACT.parquet"
    predictor = tmp_path / "PR_DATA_DM_ADMMART_PRED.parquet"
    prediction_file = tmp_path / "PR_DATA_DM_SNAPSHOTS.parquet"
    for path in (model, predictor, prediction_file):
        path.write_bytes(b"data")
    dataset = {
        "name": "Dataset",
        "data_dir": tmp_path,
        "model_file": model,
        "predictor_file": predictor,
        "prediction_file": prediction_file,
    }
    datamart = MagicMock()
    datamart.model_data = pl.LazyFrame({"ModelID": ["model-1"]})
    datamart.generate.health_check = MagicMock()
    datamart.generate.excel_report.return_value = (None, [])
    prediction = MagicMock()

    with (
        patch.object(batch.ADMDatamart, "from_ds_export", return_value=datamart),
        patch.object(batch.Prediction, "from_ds_export", return_value=prediction),
        patch.object(batch, "select_interesting_models", return_value=[]),
        patch.object(batch, "is_esbuild_available", return_value=True),
        patch.object(batch, "_generate_quarto_report", return_value=(1.0, "Success", None)) as generate,
    ):
        result = batch.process_dataset(dataset, tmp_path / "reports")

    assert result["Prediction_File_MB"] > 0
    assert generate.call_count == 2
    for call in generate.call_args_list:
        assert call.kwargs["prediction"] is prediction
        assert call.kwargs["model_file_path"] == model
        assert call.kwargs["predictor_file_path"] == predictor
        assert call.kwargs["prediction_file_path"] == prediction_file


def test_process_dataset_defaults_output_to_dataset_directory_without_cleaning(tmp_path):
    model = tmp_path / "PR_DATA_DM_ADMMART_MDL_FACT.parquet"
    model.write_bytes(b"data")
    unrelated_report = tmp_path / "HealthCheck (1).html"
    unrelated_report.write_text("stale")
    dataset = {
        "name": "Dataset",
        "data_dir": tmp_path,
        "model_file": model,
        "predictor_file": None,
        "prediction_file": None,
    }
    datamart = MagicMock()
    datamart.model_data = pl.LazyFrame({"ModelID": ["model-1"]})
    datamart.generate.excel_report.return_value = (None, [])

    with (
        patch.object(batch.ADMDatamart, "from_ds_export", return_value=datamart),
        patch.object(batch, "select_interesting_models", return_value=[]),
        patch.object(batch, "is_esbuild_available", return_value=True),
        patch.object(batch, "_generate_quarto_report", return_value=(1.0, "Success", None)) as generate,
    ):
        batch.process_dataset(dataset, None)

    assert unrelated_report.read_text() == "stale"
    assert {call.args[2] for call in generate.call_args_list} == {tmp_path}
    datamart.generate.excel_report.assert_called_once_with(
        name=tmp_path / "dataset.xlsx",
        predictor_binning=True,
    )


def test_main_defaults_to_per_dataset_output(tmp_path, monkeypatch):
    dataset = {
        "name": "Dataset",
        "data_dir": tmp_path / "Dataset" / "HC",
        "model_file": tmp_path / "Dataset" / "HC" / "PR_DATA_DM_ADMMART_MDL_FACT.parquet",
        "predictor_file": None,
        "prediction_file": None,
    }
    result = {
        "Dataset": "Dataset",
        "Model_File_MB": 1.0,
        "Predictor_File_MB": 0.0,
        "Prediction_File_MB": 0.0,
        "HC_CDN_MB": 1.0,
        "HC_CDN_Status": "Success",
        "HC_CDN_Errors": None,
        "HC_Embed_MB": 2.0,
        "HC_Embed_Status": "Success",
        "HC_Embed_Errors": None,
        "ModelReport_Models": 0,
        "ModelReport_CDN_MB": 0.0,
        "ModelReport_CDN_Status": "Skipped",
        "ModelReport_CDN_Errors": None,
        "ModelReport_Embed_MB": 0.0,
        "ModelReport_Embed_Status": "Skipped",
        "ModelReport_Embed_Errors": None,
        "Excel_MB": 1.0,
        "Excel_Status": "Success",
    }
    monkeypatch.setattr(sys, "argv", ["batch_healthcheck.py", str(tmp_path)])

    with (
        patch.object(batch, "find_data_directories", return_value=[dataset]),
        patch.object(batch, "process_dataset", return_value=result) as process_dataset,
    ):
        batch.main()

    process_dataset.assert_called_once_with(
        dataset,
        None,
        max_models=3,
        active_window_days=30,
        positives_maturity_threshold=200,
        ci_maturity_dataset_rows=[],
        ci_maturity_model_rows=[],
        auc_rollup_rows=[],
    )
    assert (tmp_path / "summary.csv").exists()


def test_main_error_summary_includes_data_paths(tmp_path, monkeypatch, capsys):
    dataset = {
        "name": "Dataset",
        "data_dir": tmp_path / "Dataset" / "HC",
        "model_file": tmp_path / "Dataset" / "HC" / "PR_DATA_DM_ADMMART_MDL_FACT.parquet",
        "predictor_file": tmp_path / "Dataset" / "HC" / "PR_DATA_DM_ADMMART_PRED.parquet",
        "prediction_file": None,
    }
    result = {
        "Dataset": "Dataset",
        "Data_Dir": str(dataset["data_dir"]),
        "Model_File": str(dataset["model_file"]),
        "Predictor_File": str(dataset["predictor_file"]),
        "Prediction_File": None,
        "Model_File_MB": 1.0,
        "Predictor_File_MB": 1.0,
        "Prediction_File_MB": 0.0,
        "HC_CDN_MB": 0.0,
        "HC_CDN_Status": "Error",
        "HC_CDN_Errors": "render failed",
        "HC_Embed_MB": 0.0,
        "HC_Embed_Status": "Skipped",
        "HC_Embed_Errors": None,
        "ModelReport_Models": 0,
        "ModelReport_CDN_MB": 0.0,
        "ModelReport_CDN_Status": "Skipped",
        "ModelReport_CDN_Errors": None,
        "ModelReport_Embed_MB": 0.0,
        "ModelReport_Embed_Status": "Skipped",
        "ModelReport_Embed_Errors": None,
        "Excel_MB": 0.0,
        "Excel_Status": "Skipped",
    }
    monkeypatch.setattr(sys, "argv", ["batch_healthcheck.py", str(tmp_path)])

    with (
        patch.object(batch, "find_data_directories", return_value=[dataset]),
        patch.object(batch, "process_dataset", return_value=result),
        pytest.raises(SystemExit) as exit_info,
    ):
        batch.main()

    captured = capsys.readouterr()

    assert exit_info.value.code == 1
    assert "Report Errors Detected (HC CDN)" in captured.out
    assert f"Data directory: {dataset['data_dir']}" in captured.out
    assert f"Model file: {dataset['model_file']}" in captured.out
    assert f"Predictor file: {dataset['predictor_file']}" in captured.out
    assert "  - render failed" in captured.out


def test_process_dataset_generates_individual_model_reports(tmp_path):
    model = tmp_path / "PR_DATA_DM_ADMMART_MDL_FACT.parquet"
    model.write_bytes(b"data")
    dataset = {
        "name": "Dataset",
        "data_dir": tmp_path,
        "model_file": model,
        "predictor_file": None,
        "prediction_file": None,
    }
    datamart = MagicMock()
    datamart.model_data = pl.LazyFrame({"ModelID": ["model-1", "model-2"]})
    datamart.generate.excel_report.return_value = (None, [])

    with (
        patch.object(batch.ADMDatamart, "from_ds_export", return_value=datamart),
        patch.object(batch, "select_interesting_models", return_value=["model-1", "model-2"]),
        patch.object(batch, "is_esbuild_available", return_value=True),
        patch.object(batch, "_generate_quarto_report", return_value=(1.0, "Success", None)) as generate,
    ):
        result = batch.process_dataset(dataset, tmp_path / "reports")

    model_report_calls = [call for call in generate.call_args_list if call.args[1].startswith("ModelReport")]
    assert [call.kwargs["model_ids"] for call in model_report_calls] == [
        "model-1",
        "model-2",
        "model-1",
        "model-2",
    ]
    assert all(isinstance(call.kwargs["model_ids"], str) for call in model_report_calls)
    assert result["ModelReport_CDN_MB"] == 2.0
    assert result["ModelReport_Embed_MB"] == 2.0
    assert result["ModelReport_CDN_Status"] == "Success"
    assert result["ModelReport_Embed_Status"] == "Success"


def test_process_dataset_skips_full_embed_when_esbuild_unavailable(tmp_path):
    """Full-embed reports need esbuild; without it they are skipped, not failed.

    Regression test for the DJS "no esbuild" environment (issue #620): the
    batch runner must not attempt (and therefore fail) full-embed generation
    when esbuild is absent, otherwise the CDN-only run exits non-zero.
    """
    model = tmp_path / "PR_DATA_DM_ADMMART_MDL_FACT.parquet"
    model.write_bytes(b"data")
    dataset = {
        "name": "Dataset",
        "data_dir": tmp_path,
        "model_file": model,
        "predictor_file": None,
        "prediction_file": None,
    }
    datamart = MagicMock()
    datamart.model_data = pl.LazyFrame({"ModelID": ["model-1"]})
    datamart.generate.excel_report.return_value = (None, [])

    with (
        patch.object(batch.ADMDatamart, "from_ds_export", return_value=datamart),
        patch.object(batch, "select_interesting_models", return_value=["model-1"]),
        patch.object(batch, "is_esbuild_available", return_value=False),
        patch.object(batch, "_generate_quarto_report", return_value=(1.0, "Success", None)) as generate,
    ):
        result = batch.process_dataset(dataset, tmp_path / "reports")

    # Only the CDN variants are generated (HealthCheck + model report).
    assert generate.call_count == 2
    for call in generate.call_args_list:
        assert call.kwargs["full_embed"] is False

    # Full-embed variants are marked Skipped so they don't count as failures.
    assert result["HC_CDN_Status"] == "Success"
    assert result["HC_Embed_Status"] == "Skipped"
    assert result["ModelReport_CDN_Status"] == "Success"
    assert result["ModelReport_Embed_Status"] == "Skipped"


def test_compute_ci_maturity_analysis_returns_expected_metrics():
    now = datetime(2026, 8, 3, 12, 0, 0)
    datamart = MagicMock()
    datamart.predictor_data = pl.LazyFrame(
        {
            "ModelID": ["m1", "m1", "m2", "m2"],
            "EntryType": ["Active", "Classifier", "Active", "Classifier"],
            "BinType": ["SYMBOLIC", "NONE", "SYMBOLIC", "NONE"],
        }
    )
    datamart.model_data = pl.LazyFrame(
        {
            "ModelID": ["m1", "m1", "m2", "m2"],
            "Positives": [250, 100, 50, 25],
            "ResponseCount": [1200, 800, 300, 200],
            "SnapshotTime": [now - timedelta(days=2), now - timedelta(days=10), now - timedelta(days=1), now],
        }
    )
    datamart.active_ranges.return_value = pl.LazyFrame(
        {
            "ModelID": ["m1", "m2"],
            "AUC_ActiveRange": [0.72, 0.68],
            "AUC_ActiveRange_CI_Lower": [0.69, 0.63],
            "AUC_ActiveRange_CI_Upper": [0.75, 0.73],
            "AUC_ActiveRange_CI_Available": [True, True],
            "AUC_ActiveRange_CI_Reason": [None, None],
        }
    )

    metrics, model_level = batch._compute_ci_maturity_analysis(
        datamart,
        active_window_days=30,
        positives_maturity_threshold=200,
    )

    assert metrics["Active_NB_Models"] == 2
    assert metrics["Active_NB_Models_With_CI"] == 2
    assert metrics["Maturity_Pct_Above_Threshold"] == 50.0
    assert model_level.height == 2
    assert "CI_Width" in model_level.columns
    assert "PositivesSegment" in model_level.columns


def test_compute_ci_maturity_analysis_retains_rows_when_ci_fails_for_some_models():
    now = datetime(2026, 8, 3, 12, 0, 0)
    datamart = MagicMock()
    datamart.predictor_data = pl.LazyFrame(
        {
            "ModelID": ["m1", "m1", "m2", "m2"],
            "EntryType": ["Active", "Classifier", "Active", "Classifier"],
            "BinType": ["SYMBOLIC", "NONE", "SYMBOLIC", "NONE"],
        }
    )
    datamart.model_data = pl.LazyFrame(
        {
            "ModelID": ["m1", "m2"],
            "Positives": [250, 50],
            "ResponseCount": [1200, 300],
            "SnapshotTime": [now - timedelta(days=1), now],
        }
    )

    def active_ranges_side_effect(model_id):
        if model_id == "m1":
            return pl.LazyFrame(
                {
                    "ModelID": ["m1"],
                    "AUC_ActiveRange": [0.72],
                    "AUC_ActiveRange_CI_Lower": [0.69],
                    "AUC_ActiveRange_CI_Upper": [0.75],
                    "AUC_ActiveRange_CI_Available": [True],
                    "AUC_ActiveRange_CI_Reason": [None],
                }
            )
        raise ValueError("pos and neg must be non-empty")

    datamart.active_ranges.side_effect = active_ranges_side_effect

    metrics, model_level = batch._compute_ci_maturity_analysis(
        datamart,
        active_window_days=30,
        positives_maturity_threshold=200,
    )

    assert metrics["Active_NB_Models"] == 2
    assert metrics["Active_NB_Models_With_CI"] == 1
    assert model_level.height == 2
    failing_model = model_level.filter(pl.col("ModelID") == "m2")
    assert failing_model.height == 1
    assert failing_model["AUC_ActiveRange_CI_Available"][0] is False
    assert failing_model["AUC_ActiveRange_CI_Reason"][0] == "analysis_error"


def test_compute_ci_maturity_analysis_splits_agb_and_defaults_missing_technique():
    now = datetime(2026, 8, 3, 12, 0, 0)
    datamart = MagicMock()
    datamart.predictor_data = pl.LazyFrame(
        {
            "ModelID": ["nb", "agb"],
            "EntryType": ["Classifier", "Classifier"],
        }
    )
    datamart.model_data = pl.LazyFrame(
        {
            "ModelID": ["nb", "agb"],
            "ModelTechnique": [None, "GradientBoost"],
            "Positives": [250, 500],
            "ResponseCount": [1200, 2000],
            "SnapshotTime": [now, now],
        }
    )

    def active_ranges(model_id):
        return pl.LazyFrame(
            {
                "ModelID": [model_id],
                "AUC_ActiveRange": [0.7],
                "AUC_ActiveRange_CI_Lower": [0.65],
                "AUC_ActiveRange_CI_Upper": [0.75],
                "AUC_ActiveRange_CI_Available": [True],
                "AUC_ActiveRange_CI_Reason": [None],
            }
        )

    datamart.active_ranges.side_effect = active_ranges

    metrics, model_level = batch._compute_ci_maturity_analysis(
        datamart,
        active_window_days=30,
        positives_maturity_threshold=200,
    )

    assert metrics["Active_NB_Models"] == 1
    assert metrics["Active_AGB_Models"] == 1
    assert metrics["NB_CI_Width_Mean"] == pytest.approx(0.1)
    assert metrics["AGB_CI_Width_Mean"] == pytest.approx(0.1)
    assert model_level.select("ModelID", "ModelTechnique").sort("ModelID").to_dicts() == [
        {"ModelID": "agb", "ModelTechnique": "GradientBoost"},
        {"ModelID": "nb", "ModelTechnique": "NaiveBayes"},
    ]


def test_main_writes_ci_maturity_outputs_when_enabled(tmp_path, monkeypatch):
    dataset = {
        "name": "Dataset",
        "data_dir": tmp_path / "Dataset" / "HC",
        "model_file": tmp_path / "Dataset" / "HC" / "PR_DATA_DM_ADMMART_MDL_FACT.parquet",
        "predictor_file": None,
        "prediction_file": None,
    }
    result = {
        "Dataset": "Dataset",
        "Model_File_MB": 1.0,
        "Predictor_File_MB": 0.0,
        "Prediction_File_MB": 0.0,
        "HC_CDN_MB": 1.0,
        "HC_CDN_Status": "Success",
        "HC_CDN_Errors": None,
        "HC_Embed_MB": 0.0,
        "HC_Embed_Status": "Skipped",
        "HC_Embed_Errors": None,
        "ModelReport_Models": 0,
        "ModelReport_CDN_MB": 0.0,
        "ModelReport_CDN_Status": "Skipped",
        "ModelReport_CDN_Errors": None,
        "ModelReport_Embed_MB": 0.0,
        "ModelReport_Embed_Status": "Skipped",
        "ModelReport_Embed_Errors": None,
        "Excel_MB": 1.0,
        "Excel_Status": "Success",
        "Active_NB_Models": 1,
        "Active_NB_Models_With_CI": 1,
        "Maturity_Pct_Above_Threshold": 100.0,
        "CI_Width_Mean": 0.05,
        "CI_Width_Median": 0.05,
        "CI_Width_P90": 0.05,
        "CI_Width_Mean_AboveThreshold": 0.05,
        "CI_Width_Mean_AtOrBelowThreshold": None,
        "CI_Width_Ratio_AtOrBelow_over_Above": None,
        "Positives_vs_CI_Width_Spearman": None,
    }

    def fake_process_dataset(*args, **kwargs):
        kwargs["ci_maturity_dataset_rows"].append(
            {
                "Dataset": "Dataset",
                "Active_NB_Models": 1,
                "Active_NB_Models_With_CI": 1,
                "Maturity_Pct_Above_Threshold": 100.0,
            }
        )
        kwargs["ci_maturity_model_rows"].append(
            {
                "Dataset": "Dataset",
                "ModelID": "m1",
                "Positives": 250,
                "ResponseCount": 1200,
                "CI_Width": 0.05,
            }
        )
        return result

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_healthcheck.py",
            str(tmp_path),
            "--output",
            str(tmp_path / "reports"),
        ],
    )

    with (
        patch.object(batch, "find_data_directories", return_value=[dataset]),
        patch.object(batch, "process_dataset", side_effect=fake_process_dataset),
    ):
        batch.main()

    assert (tmp_path / "reports" / "ci_maturity_dataset_summary.csv").exists()
    assert (tmp_path / "reports" / "ci_maturity_model_level.csv").exists()


def test_generate_ci_maturity_plots_returns_empty_for_no_ci_rows(tmp_path):
    empty_plot_df = pl.DataFrame(
        {
            "ModelID": ["m1"],
            "Positives": [250],
            "ResponseCount": [1200],
            "CI_Width": [None],
        }
    )

    outputs = batch._generate_ci_maturity_plots(
        empty_plot_df,
        output_dir=tmp_path,
        positives_maturity_threshold=200,
    )

    assert outputs == []


def test_generate_ci_maturity_plots_writes_one_log_log_plot(tmp_path):
    model_level_df = pl.DataFrame(
        {
            "ModelID": ["m1", "m2", "m3"],
            "ModelTechnique": ["NaiveBayes", "GradientBoost", "NaiveBayes"],
            "Positives": [100.0, 250.0, 1000.0],
            "CI_Width": [0.4, 0.25, 0.12],
        }
    )

    outputs = batch._generate_ci_maturity_plots(
        model_level_df,
        output_dir=tmp_path,
        positives_maturity_threshold=200,
    )

    assert outputs == [tmp_path / "ci_maturity_vs_confidence_intervals.png"]
    assert outputs[0].exists()
    assert not (tmp_path / "ci_maturity_vs_confidence_intervals_logx.png").exists()
    assert not (tmp_path / "ci_maturity_vs_confidence_intervals_cap10k.png").exists()


def test_generate_cross_dataset_ci_width_plot(tmp_path):
    model_level_df = pl.DataFrame(
        {
            "Dataset": ["Dataset A", "Dataset A", "Dataset B", "Dataset B"],
            "Positives": [100.0, 1000.0, 10000.0, 100000.0],
            "CI_Width": [0.4, 0.15, 0.05, 0.015],
        }
    )

    output = batch._generate_cross_dataset_ci_width_plot(
        model_level_df,
        output_dir=tmp_path,
    )

    assert output == tmp_path / "ci_width_vs_positives_all_datasets.png"
    assert output.exists()
    assert output.stat().st_size > 0


def test_generate_cross_dataset_ci_width_plot_returns_none_without_valid_rows(tmp_path):
    model_level_df = pl.DataFrame(
        {
            "Dataset": ["Dataset"],
            "Positives": [100.0],
            "CI_Width": [None],
        }
    )

    assert batch._generate_cross_dataset_ci_width_plot(model_level_df, output_dir=tmp_path) is None


def test_process_dataset_writes_ci_plots_to_dataset_data_dir(tmp_path):
    data_dir = tmp_path / "Dataset" / "HC"
    data_dir.mkdir(parents=True)
    model = data_dir / "PR_DATA_DM_ADMMART_MDL_FACT.parquet"
    model.write_bytes(b"data")
    dataset = {
        "name": "Dataset",
        "data_dir": data_dir,
        "model_file": model,
        "predictor_file": None,
        "prediction_file": None,
    }
    datamart = MagicMock()
    datamart.model_data = pl.LazyFrame({"ModelID": ["model-1"]})
    datamart.generate.excel_report.return_value = (None, [])

    ci_metrics = {
        "Active_NB_Models": 1,
        "Active_NB_Models_With_CI": 1,
        "Maturity_Pct_Above_Threshold": 100.0,
        "CI_Width_Mean": 0.05,
        "CI_Width_Median": 0.05,
        "CI_Width_P90": 0.05,
        "CI_Width_Mean_AboveThreshold": 0.05,
        "CI_Width_Mean_AtOrBelowThreshold": None,
        "CI_Width_Ratio_AtOrBelow_over_Above": None,
        "Positives_vs_CI_Width_Spearman": None,
    }
    ci_model_df = pl.DataFrame(
        {
            "ModelID": ["model-1"],
            "Positives": [250.0],
            "ResponseCount": [1200.0],
            "CI_Width": [0.05],
        }
    )

    with (
        patch.object(batch.ADMDatamart, "from_ds_export", return_value=datamart),
        patch.object(batch, "select_interesting_models", return_value=[]),
        patch.object(batch, "is_esbuild_available", return_value=True),
        patch.object(batch, "_generate_quarto_report", return_value=(1.0, "Success", None)),
        patch.object(batch, "_compute_ci_maturity_analysis", return_value=(ci_metrics, ci_model_df)),
        patch.object(
            batch, "_generate_ci_maturity_plots", return_value=[data_dir / "ci_maturity_vs_confidence_intervals.png"]
        ) as plot_gen,
    ):
        batch.process_dataset(dataset, tmp_path / "reports")

    plot_gen.assert_called_once()
    assert plot_gen.call_args.kwargs["output_dir"] == data_dir
    assert plot_gen.call_args.kwargs["positives_maturity_threshold"] == 200


def test_generate_quarto_report_fails_when_rendered_html_contains_errors(tmp_path):
    def generate_error_report(output_dir, full_embed):
        report = Path(output_dir) / "HealthCheck.html"
        report.write_text("<html><body>Error rendering Predictor Importance plot: TypeError:</body></html>")
        return report

    size_mb, status, errors = batch._generate_quarto_report(
        generate_error_report,
        "HealthCheck",
        tmp_path,
        full_embed=False,
    )

    assert size_mb > 0
    assert status == "Error"
    assert "Plot rendering error" in errors
    assert "TypeError exception" in errors


@pytest.fixture
def hc_layout(tmp_path):
    """Create a realistic HC/ directory from the repo's sample CSVs."""
    model_csv = DATA_DIR / "pr_data_dm_admmart_mdl_fact.csv"
    pred_csv = DATA_DIR / "pr_data_dm_admmart_pred.csv"
    if not model_csv.exists():
        pytest.skip("Sample CSV data not available")

    hc_dir = tmp_path / "SampleCustomer" / "HC"
    hc_dir.mkdir(parents=True)

    pl.read_csv(model_csv).write_parquet(hc_dir / "PR_DATA_DM_ADMMART_MDL_FACT.parquet")
    if pred_csv.exists():
        pl.read_csv(pred_csv).write_parquet(hc_dir / "PR_DATA_DM_ADMMART_PRED.parquet")

    return tmp_path


def _run_batch(hc_layout, output_dir, extra_args=None):
    """Run batch_healthcheck.py and return the subprocess result."""
    cmd = [
        sys.executable,
        str(SCRIPT),
        str(hc_layout),
        "--output",
        str(output_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.slow
def test_batch_healthcheck_cdn(hc_layout, tmp_path):
    """Verify the CDN healthcheck report is produced."""
    output_dir = tmp_path / "reports"
    result = _run_batch(hc_layout, output_dir)

    assert result.returncode == 0, f"batch_healthcheck.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    # Verify CDN HTML report was created
    cdn_files = list(output_dir.glob("*_cdn.html"))
    assert len(cdn_files) >= 1, f"No CDN report found, files: {[f.name for f in output_dir.glob('*.html')]}"

    for html_file in cdn_files:
        size_kb = html_file.stat().st_size / 1024
        assert size_kb > 100, f"{html_file.name} is suspiciously small: {size_kb:.1f} KB"

    # Verify summary CSV
    summary = output_dir / "summary.csv"
    assert summary.exists(), "summary.csv was not created"
    df = pl.read_csv(summary)
    assert len(df) == 1
    assert df["HC_CDN_Status"][0] == "Success"
    assert df["HC_CDN_MB"][0] > 0

    # Verify Excel export was created
    xlsx_files = list(output_dir.glob("*.xlsx"))
    assert len(xlsx_files) >= 1, f"No Excel export found in {output_dir}"
    assert df["Excel_Status"][0] == "Success"
    assert df["Excel_MB"][0] > 0


@pytest.mark.slow
def test_batch_healthcheck_full_embed(hc_layout, tmp_path):
    """Verify both CDN and full-embed reports are produced."""
    output_dir = tmp_path / "reports"
    result = _run_batch(hc_layout, output_dir)

    assert result.returncode == 0, f"batch_healthcheck.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    cdn_files = list(output_dir.glob("*_cdn.html"))
    full_files = list(output_dir.glob("*_full.html"))
    assert len(cdn_files) >= 1, f"No CDN report, files: {[f.name for f in output_dir.glob('*.html')]}"
    assert len(full_files) >= 1, f"No full-embed report, files: {[f.name for f in output_dir.glob('*.html')]}"

    # HealthCheck: both rendering modes should produce non-trivial reports.
    hc_cdn = [f for f in cdn_files if "model" not in f.name]
    hc_full = [f for f in full_files if "model" not in f.name]
    assert len(hc_cdn) >= 1, f"No HealthCheck CDN report, files: {[f.name for f in cdn_files]}"
    assert len(hc_full) >= 1, f"No HealthCheck full-embed report, files: {[f.name for f in full_files]}"
    for html_file in [hc_cdn[0], hc_full[0]]:
        size_kb = html_file.stat().st_size / 1024
        print(f"{html_file.name}: {size_kb:.1f} KB")
        assert size_kb > 100, f"{html_file.name} is suspiciously small: {size_kb:.1f} KB"

    # Verify summary has both HC modes
    df = pl.read_csv(output_dir / "summary.csv")
    assert df["HC_Embed_Status"][0] == "Success"
    assert df["HC_Embed_MB"][0] > 0

    # Verify model reports if the sample data had qualifying models
    n_models = df["ModelReport_Models"][0]
    print(f"Model reports generated: {n_models}")
    if n_models > 0:
        # Individual model reports use singular "model" in the generated file names.
        all_outputs = list(output_dir.glob("*model*"))
        print(f"Model report files: {[f.name for f in all_outputs]}")
        assert len(all_outputs) >= 2, (
            f"Expected CDN + full-embed model report outputs, found: {[f.name for f in all_outputs]}"
        )
        assert df["ModelReport_CDN_MB"][0] > 0
        assert df["ModelReport_Embed_MB"][0] > 0
