"""Integration test for the batch report generator (scripts/batch_healthcheck.py).

Runs the actual script as a subprocess with sample data, verifying it
produces valid HTML healthcheck reports, model reports, and Excel exports.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
SCRIPT = Path(__file__).parent.parent.parent.parent / "scripts" / "batch_healthcheck.py"
SPEC = importlib.util.spec_from_file_location("batch_healthcheck", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
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
    )
    assert (tmp_path / "summary.csv").exists()


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
    assert errors is not None
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
    """Verify both CDN and full-embed reports are produced and compare sizes."""
    output_dir = tmp_path / "reports"
    result = _run_batch(hc_layout, output_dir)

    assert result.returncode == 0, f"batch_healthcheck.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    cdn_files = list(output_dir.glob("*_cdn.html"))
    full_files = list(output_dir.glob("*_full.html"))
    assert len(cdn_files) >= 1, f"No CDN report, files: {[f.name for f in output_dir.glob('*.html')]}"
    assert len(full_files) >= 1, f"No full-embed report, files: {[f.name for f in output_dir.glob('*.html')]}"

    # HealthCheck: full-embed should be larger than CDN
    hc_cdn = [f for f in cdn_files if "models" not in f.name]
    hc_full = [f for f in full_files if "models" not in f.name]
    if hc_cdn and hc_full:
        cdn_size = hc_cdn[0].stat().st_size
        full_size = hc_full[0].stat().st_size
        print(f"HC CDN:        {cdn_size / (1024 * 1024):.1f} MB")
        print(f"HC full-embed: {full_size / (1024 * 1024):.1f} MB")
        print(f"Ratio:         {full_size / cdn_size:.1f}x")
        assert full_size > cdn_size, "Full-embed HC should be larger than CDN"

    # Verify summary has both HC modes
    df = pl.read_csv(output_dir / "summary.csv")
    assert df["HC_Embed_Status"][0] == "Success"
    assert df["HC_Embed_MB"][0] > 0

    # Verify model reports if the sample data had qualifying models
    n_models = df["ModelReport_Models"][0]
    print(f"Model reports generated: {n_models}")
    if n_models > 0:
        # Multi-model reports are zipped; single model reports are HTML
        all_outputs = list(output_dir.glob("*models*"))
        print(f"Model report files: {[f.name for f in all_outputs]}")
        assert len(all_outputs) >= 2, (
            f"Expected CDN + full-embed model report outputs, found: {[f.name for f in all_outputs]}"
        )
        assert df["ModelReport_CDN_MB"][0] > 0
        assert df["ModelReport_Embed_MB"][0] > 0
