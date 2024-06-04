import sys
import os
import pathlib
from pandas import read_excel, ExcelFile

basePath = pathlib.Path(__file__).parent.parent.parent
sys.path.append(f"{str(basePath)}/python")
from pdstools import ADMDatamart, datasets
import pytest


@pytest.fixture
def sample():
    return datasets.CDHSample()


@pytest.fixture
def sample_without_predictorbinning():
    return ADMDatamart(
        path=f"{basePath}/data",
        model_filename="Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210526T131808_GMT.zip",
        predictor_filename=None,
    )


def test_GenerateHealthCheck(sample):
    hc = sample.generateReport(verbose=True)
    assert hc == "./HealthCheck.html"
    assert pathlib.Path(hc).exists()
    pathlib.Path(hc).unlink()
    assert not pathlib.Path(hc).exists()

def test_ExportTables(sample):
    excel = sample.exportTables(predictorBinning=True)
    assert excel == "Tables.xlsx"
    assert pathlib.Path(excel).exists()
    spreadsheet = ExcelFile(excel)
    assert list(spreadsheet.sheet_names) == [
        "modeldata_last_snapshot",
        "predictor_last_snapshot",
        "predictorbinning",
    ]
    # TODO we could go further and check the size of the sheets
    # spreadsheet = read_excel(excel, sheet_name=None)
    pathlib.Path(excel).unlink()
    assert not pathlib.Path(excel).exists()

def test_ExportTables_NoBinning(sample):
    excel = sample.exportTables(predictorBinning=False)
    assert excel == "Tables.xlsx"
    assert pathlib.Path(excel).exists()
    spreadsheet = ExcelFile(excel)
    assert list(spreadsheet.sheet_names) == [
        "modeldata_last_snapshot",
        "predictor_last_snapshot",
    ]
    # TODO we could go further and check the size of the sheets
    # spreadsheet = read_excel(excel, sheet_name=None)
    pathlib.Path(excel).unlink()
    assert not pathlib.Path(excel).exists()

def test_GenerateHealthCheck_ModelDataOnly(sample_without_predictorbinning):
    hc = sample_without_predictorbinning.generateReport(
        name="MyOrg", verbose=True, modelData_only=True
    )
    assert hc == "./HealthCheck_MyOrg.html"
    assert pathlib.Path(hc).exists()
    pathlib.Path(hc).unlink()
    assert not pathlib.Path(hc).exists()


def test_ExportTables_ModelDataOnly(sample_without_predictorbinning):
    excel = sample_without_predictorbinning.exportTables(
        file="ModelTables.xlsx", predictorBinning=True
    )
    assert excel == "ModelTables.xlsx"
    assert pathlib.Path(excel).exists()
    spreadsheet = ExcelFile(excel)
    assert list(spreadsheet.sheet_names) == [
        "modeldata_last_snapshot",
    ]
    # TODO we could go further and check the size of the sheets
    # spreadsheet = read_excel(excel, sheet_name=None)
    pathlib.Path(excel).unlink()


def test_GenerateModelReport(sample):
    report = sample.generateReport(
        name="MyOrg", modelid="bd70a915-697a-5d43-ab2c-53b0557c85a0"
    )
    assert report == "./ModelReport_MyOrg_bd70a915-697a-5d43-ab2c-53b0557c85a0.html"
    assert pathlib.Path(report).exists()
    pathlib.Path(report).unlink()
    assert not pathlib.Path(report).exists()

def test_GenerateModelReport_Failing(sample_without_predictorbinning):
    with pytest.raises(Exception) as e_info:
        sample_without_predictorbinning.generateReport(
            name="MyOrg", modelid="bd70a915-697a-5d43-ab2c-53b0557c85a0"
        )
    assert "Error when generating healthcheck" in str(e_info)
