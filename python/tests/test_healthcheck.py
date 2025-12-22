import pathlib

import pytest
from openpyxl import load_workbook
from pdstools import ADMDatamart, datasets, read_ds_export, Prediction

basePath = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture
def sample() -> ADMDatamart:
    return datasets.cdh_sample()


@pytest.fixture
def sample_without_predictor_binning() -> ADMDatamart:
    """Fixture to serve as class to call functions from."""
    # Using from_ds_export automaticaly detects predictor_snapshot.
    model_df = read_ds_export(
        filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
        path=f"{basePath}/data",
    )
    return ADMDatamart(model_df=model_df)


@pytest.fixture
def sample_prediction_data() -> Prediction:
    return Prediction.from_mock_data(days=30)


def test_GenerateHealthCheck(sample: ADMDatamart):
    hc = sample.generate.health_check()
    assert hc == pathlib.Path("./HealthCheck.html").resolve()
    assert pathlib.Path(hc).exists()
    pathlib.Path(hc).unlink()
    assert not pathlib.Path(hc).exists()


@pytest.mark.slow
def test_HealthCheck_size_reduction_methods(sample: ADMDatamart, tmp_path):
    """Test health check file sizes for all size_reduction_method options."""
    sizes = {}

    default = sample.generate.health_check(output_dir=tmp_path, name="default")
    sizes["default"] = default.stat().st_size

    no_reduction = sample.generate.health_check(
        output_dir=tmp_path, name="no_reduction", size_reduction_method=None
    )
    sizes["no_reduction"] = no_reduction.stat().st_size

    stripped = sample.generate.health_check(
        output_dir=tmp_path, name="stripped", size_reduction_method="strip"
    )
    sizes["stripped"] = stripped.stat().st_size

    cdn = sample.generate.health_check(
        output_dir=tmp_path, name="cdn", size_reduction_method="cdn"
    )
    sizes["cdn"] = cdn.stat().st_size

    # TODO: temporary default for DJS use cases
    size_diff = abs(sizes["default"] - sizes["cdn"]) / sizes["cdn"]
    assert (
        size_diff <= 0.01
    ), f"Default should be within 1% of cdn, got {size_diff:.1%} difference"

    no_reduction_mb = sizes["no_reduction"] / (1024 * 1024)
    assert (
        80 <= no_reduction_mb <= 150
    ), f"Embedded size should be can 100 MB, got {no_reduction_mb:.1f} MB"

    strip_reduction = 1 - (sizes["stripped"] / sizes["no_reduction"])
    assert (
        strip_reduction >= 0.50
    ), f"Strip should reduce by 50%+, got {strip_reduction:.0%}"

    cdn_reduction = 1 - (sizes["cdn"] / sizes["no_reduction"])
    assert cdn_reduction >= 0.80, f"CDN should reduce by 80%+, got {cdn_reduction:.0%}"


def test_ExportTables(sample: ADMDatamart):
    excel, warning_messages = sample.generate.excel_report(predictor_binning=True)
    assert excel == pathlib.Path("./Tables.xlsx")
    assert excel.exists()
    spreadsheet = load_workbook(excel)
    assert spreadsheet.sheetnames == [
        "adm_models",
        "predictors_detail",
        "predictors_overview",
        "predictor_binning",
    ]
    # TODO we could go further and check the size of the sheets
    # spreadsheet = read_excel(excel, sheet_name=None)
    pathlib.Path(excel).unlink()
    assert not pathlib.Path(excel).exists()


def test_ExportTables_NoBinning(sample: ADMDatamart):
    excel, warining_messages = sample.generate.excel_report(predictor_binning=False)
    assert excel == pathlib.Path("./Tables.xlsx")
    assert pathlib.Path(excel).exists()
    spreadsheet = load_workbook(excel)
    assert spreadsheet.sheetnames == [
        "adm_models",
        "predictors_detail",
        "predictors_overview",
    ]
    # TODO we could go further and check the size of the sheets
    # spreadsheet = read_excel(excel, sheet_name=None)
    pathlib.Path(excel).unlink()
    assert not pathlib.Path(excel).exists()


def test_GenerateHealthCheck_ModelDataOnly(
    sample_without_predictor_binning: ADMDatamart,
):
    hc = sample_without_predictor_binning.generate.health_check(name="MyOrg")
    assert hc == pathlib.Path("./HealthCheck_MyOrg.html").resolve()
    assert pathlib.Path(hc).exists()
    pathlib.Path(hc).unlink()
    assert not pathlib.Path(hc).exists()


def test_GenerateHealthCheck_PredictionData(
    sample: ADMDatamart,
    sample_prediction_data: Prediction,
):
    hc = sample.generate.health_check(
        prediction=sample_prediction_data, name="WithPredictions"
    )
    assert hc == pathlib.Path("./HealthCheck_WithPredictions.html").resolve()
    assert pathlib.Path(hc).exists()
    pathlib.Path(hc).unlink()
    assert not pathlib.Path(hc).exists()


def test_ExportTables_ModelDataOnly(sample_without_predictor_binning: ADMDatamart):
    excel, warning_messages = sample_without_predictor_binning.generate.excel_report(
        name="ModelTables.xlsx", predictor_binning=True
    )
    assert excel == pathlib.Path("ModelTables.xlsx")
    assert pathlib.Path(excel).exists()
    spreadsheet = load_workbook(excel)
    assert spreadsheet.sheetnames == [
        "adm_models",
    ]
    # TODO we could go further and check the size of the sheets
    # spreadsheet = read_excel(excel, sheet_name=None)
    pathlib.Path(excel).unlink()


def test_GenerateModelReport(sample: ADMDatamart):
    report = sample.generate.model_reports(
        model_ids=["bd70a915-697a-5d43-ab2c-53b0557c85a0"],
        name="MyOrg",
        only_active_predictors=True,
    )
    expected_path = pathlib.Path(
        "ModelReport_MyOrg_bd70a915-697a-5d43-ab2c-53b0557c85a0.html"
    ).resolve()
    assert report == expected_path
    assert pathlib.Path(report).exists()
    pathlib.Path(report).unlink()
    assert not pathlib.Path(report).exists()


def test_GenerateHealthCheck_CustomQmdFile(sample: ADMDatamart, tmp_path):
    """Test health_check with custom qmd_file argument"""
    # Create a custom qmd file
    custom_qmd = tmp_path / "custom_health.qmd"
    custom_qmd.write_text("""
---
title: "Custom Health Check Report"
format: html
params:
  title: "Custom Title"
---

# Custom Health Check

This is a custom health check template.

Test parameters: {{< meta params.title >}}
""")

    hc = sample.generate.health_check(name="CustomTemplate", qmd_file=custom_qmd)
    assert hc == pathlib.Path("./HealthCheck_CustomTemplate.html").resolve()
    assert pathlib.Path(hc).exists()
    pathlib.Path(hc).unlink()
    assert not pathlib.Path(hc).exists()


def test_GenerateModelReport_CustomQmdFile(sample: ADMDatamart, tmp_path):
    """Test model_reports with custom qmd_file argument"""
    # Create a custom qmd file
    custom_qmd = tmp_path / "custom_model.qmd"
    custom_qmd.write_text("""
---
title: "Custom Model Report"
format: html
params:
  title: "Custom Model Title"
  model_id: "test"
---

# Custom Model Report

This is a custom model report template.

Model ID: {{< meta params.model_id >}}
""")

    report = sample.generate.model_reports(
        model_ids=["bd70a915-697a-5d43-ab2c-53b0557c85a0"],
        name="CustomModel",
        qmd_file=custom_qmd,
    )
    expected_path = pathlib.Path(
        "ModelReport_CustomModel_bd70a915-697a-5d43-ab2c-53b0557c85a0.html"
    ).resolve()
    assert report == expected_path
    assert pathlib.Path(report).exists()
    pathlib.Path(report).unlink()
    assert not pathlib.Path(report).exists()
