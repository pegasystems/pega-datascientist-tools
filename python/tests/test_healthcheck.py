import pathlib

import pytest
from pandas import ExcelFile
from pdstools import ADMDatamart, datasets, read_ds_export

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


def test_GenerateHealthCheck(sample: ADMDatamart):
    hc = sample.generate.health_check()
    assert hc == pathlib.Path("./HealthCheck.html").resolve()
    assert pathlib.Path(hc).exists()
    pathlib.Path(hc).unlink()
    assert not pathlib.Path(hc).exists()


def test_ExportTables(sample: ADMDatamart):
    excel, warning_messages = sample.generate.excel_report(predictor_binning=True)
    assert excel == pathlib.Path("./Tables.xlsx")
    assert excel.exists()
    spreadsheet = ExcelFile(excel)
    assert list(spreadsheet.sheet_names) == [
        "modeldata_last_snapshot",
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
    spreadsheet = ExcelFile(excel)
    assert list(spreadsheet.sheet_names) == [
        "modeldata_last_snapshot",
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


def test_ExportTables_ModelDataOnly(sample_without_predictor_binning: ADMDatamart):
    excel, warning_messages = sample_without_predictor_binning.generate.excel_report(
        name="ModelTables.xlsx", predictor_binning=True
    )
    assert excel == pathlib.Path("ModelTables.xlsx")
    assert pathlib.Path(excel).exists()
    spreadsheet = ExcelFile(excel)
    assert list(spreadsheet.sheet_names) == [
        "modeldata_last_snapshot",
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


def test_GenerateModelReport_Failing(sample_without_predictor_binning: ADMDatamart):
    with pytest.raises(Exception) as e_info:
        sample_without_predictor_binning.generate.model_reports(
            model_ids="bd70a915-697a-5d43-ab2c-53b0557c85a0", name="MyOrg"
        )
    assert (
        "model_list argument is None, not a list, or contains non-string elements for generate_model_reports. Please provide a list of model_id strings to generate reports."
        in str(e_info)
    )
