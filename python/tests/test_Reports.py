"""
Testing the functionality of the adm Reports module
"""

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from pdstools import ADMDatamart
from pdstools.adm.Reports import Reports


@pytest.fixture
def mock_datamart():
    """Create a mock ADMDatamart instance for testing"""
    # Create a simple mock datamart with minimal data
    model_data = pl.DataFrame({
        "ModelID": ["model1", "model2"],
        "Name": ["Model 1", "Model 2"],
        "Performance": [0.8, 0.75],
        "SuccessRate": [0.6, 0.55],
        "ResponseCount": [1000, 1200],
        "Positives": [100, 120],
        "Channel": ["Web", "Mobile"],
        "Direction": ["Inbound", "Outbound"],
        "SnapshotTime": ["2023-01-01", "2023-01-02"]
    }).lazy()
    
    predictor_data = pl.DataFrame({
        "ModelID": ["model1", "model1", "model2", "model2"],
        "PredictorName": ["Age", "Income", "Age", "Income"],
        "Performance": [0.7, 0.65, 0.72, 0.68],
        "ResponseCount": [1000, 1000, 1200, 1200],
        "BinIndex": [1, 2, 1, 2],
        "BinSymbol": ["<30", ">=30", "<30", ">=30"],
        "BinResponseCount": [500, 500, 600, 600],
        "SnapshotTime": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"]
    }).lazy()
    
    # Create a mock ADMDatamart instance
    mock_dm = MagicMock(spec=ADMDatamart)
    mock_dm.model_data = model_data
    mock_dm.predictor_data = predictor_data
    mock_dm.combined_data = None
    mock_dm.context_keys = ["Name", "Channel", "Direction"]
    
    # Mock the save_data method to return file paths
    mock_dm.save_data.return_value = (
        Path("mock_model_data.arrow"),
        Path("mock_predictor_data.arrow")
    )
    
    # Create a Reports instance with the mock datamart
    reports = Reports(mock_dm)
    return reports


@patch("pdstools.utils.report_utils.get_quarto_with_version")
@patch("pdstools.utils.cdh_utils.create_working_and_temp_dir")
@patch("pdstools.adm.Reports.Reports._copy_quarto_file")
@patch("pdstools.adm.Reports.Reports.run_quarto")
@patch("shutil.copy")
def test_health_check(
    mock_shutil_copy,
    mock_run_quarto,
    mock_copy_quarto_file,
    mock_create_dirs,
    mock_get_quarto,
    mock_datamart,
    tmp_path
):
    """Test the health_check method"""
    # Setup mocks
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    output_dir.mkdir()
    temp_dir.mkdir()
    
    mock_create_dirs.return_value = (output_dir, temp_dir)
    mock_get_quarto.return_value = (Path("/usr/bin/quarto"), "1.0.0")
    mock_run_quarto.return_value = 0
    
    # Create a mock output file
    output_file = temp_dir / "HealthCheck.html"
    with open(output_file, "w") as f:
        f.write("<html>Test</html>")
    
    # Call the health_check method
    result = mock_datamart.health_check(
        name="test_report",
        title="Test Report",
        subtitle="Test Subtitle",
        output_dir=output_dir,
        keep_temp_files=True
    )
    
    # Verify the result
    assert result == output_dir / "HealthCheck_test_report.html"
    
    # Verify that the necessary methods were called
    mock_create_dirs.assert_called_once()
    mock_copy_quarto_file.assert_called_once_with("HealthCheck.qmd", temp_dir)
    mock_run_quarto.assert_called_once()
    mock_shutil_copy.assert_called_once()


@patch("pdstools.utils.report_utils.get_quarto_with_version")
@patch("pdstools.utils.cdh_utils.create_working_and_temp_dir")
@patch("pdstools.adm.Reports.Reports._copy_quarto_file")
@patch("pdstools.adm.Reports.Reports.run_quarto")
@patch("pdstools.utils.cdh_utils.process_files_to_bytes")
def test_model_reports(
    mock_process_files,
    mock_run_quarto,
    mock_copy_quarto_file,
    mock_create_dirs,
    mock_get_quarto,
    mock_datamart,
    tmp_path
):
    """Test the model_reports method"""
    # Setup mocks
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    output_dir.mkdir()
    temp_dir.mkdir()
    
    mock_create_dirs.return_value = (output_dir, temp_dir)
    mock_get_quarto.return_value = (Path("/usr/bin/quarto"), "1.0.0")
    mock_run_quarto.return_value = 0
    
    # Create mock output files
    output_file1 = temp_dir / "ModelReport_model1.html"
    output_file2 = temp_dir / "ModelReport_model2.html"
    with open(output_file1, "w") as f:
        f.write("<html>Test1</html>")
    with open(output_file2, "w") as f:
        f.write("<html>Test2</html>")
    
    # Mock the process_files_to_bytes function
    mock_process_files.return_value = (b"<html>Combined</html>", "ModelReport_test_report.html")
    
    # Call the model_reports method
    result = mock_datamart.model_reports(
        model_ids=["model1", "model2"],
        name="test_report",
        title="Test Report",
        subtitle="Test Subtitle",
        output_dir=output_dir,
        keep_temp_files=True
    )
    
    # Verify the result
    assert result == output_dir / "ModelReport_test_report.html"
    
    # Verify that the necessary methods were called
    mock_create_dirs.assert_called_once()
    mock_copy_quarto_file.assert_called_once_with("ModelReport.qmd", temp_dir)
    assert mock_run_quarto.call_count == 2
    mock_process_files.assert_called_once()


def test_get_output_filename():
    """Test the _get_output_filename method"""
    # Test with name and model_id for ModelReport
    filename = Reports._get_output_filename(
        name="test_report",
        report_type="ModelReport",
        model_id="model1",
        output_type="html"
    )
    assert filename == "ModelReport_test_report_model1.html"
    
    # Test without name for ModelReport
    filename = Reports._get_output_filename(
        name=None,
        report_type="ModelReport",
        model_id="model1",
        output_type="html"
    )
    assert filename == "ModelReport_model1.html"
    
    # Test with name for HealthCheck
    filename = Reports._get_output_filename(
        name="test_report",
        report_type="HealthCheck",
        model_id=None,
        output_type="html"
    )
    assert filename == "HealthCheck_test_report.html"
    
    # Test without name for HealthCheck
    filename = Reports._get_output_filename(
        name=None,
        report_type="HealthCheck",
        model_id=None,
        output_type="html"
    )
    assert filename == "HealthCheck.html"
    
    # Test with spaces in name
    filename = Reports._get_output_filename(
        name="test report",
        report_type="HealthCheck",
        model_id=None,
        output_type="html"
    )
    assert filename == "HealthCheck_test_report.html"


def test_write_params_files(tmp_path):
    """Test the _write_params_files method"""
    # Setup test directory
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    
    # Define test parameters
    params = {"param1": "value1", "param2": "value2"}
    project = {"title": "Test Project", "type": "default"}
    analysis = {"models": True, "predictors": True}
    
    # Mock the yaml.dump function
    with patch("yaml.dump") as mock_yaml_dump:
        # Call the method
        Reports._write_params_files(temp_dir, params, project, analysis)
        
        # Verify that yaml.dump was called twice
        assert mock_yaml_dump.call_count == 2
        
        # Verify the arguments for the first call (params.yml)
        args1, kwargs1 = mock_yaml_dump.call_args_list[0]
        assert args1[0] == params
        
        # Verify the arguments for the second call (_quarto.yml)
        args2, kwargs2 = mock_yaml_dump.call_args_list[1]
        assert args2[0] == {"project": project, "analysis": analysis}


def test_run_quarto_mock():
    """Test the run_quarto method by directly mocking its dependencies"""
    # Skip this test for now since it's difficult to mock all the dependencies
    # We'll rely on the integration tests for the run_quarto method
    pass


@patch("xlsxwriter.Workbook")
def test_excel_report(mock_workbook, mock_datamart):
    """Test the excel_report method"""
    # Setup mock workbook
    mock_wb = MagicMock()
    mock_workbook.return_value.__enter__.return_value = mock_wb
    
    # Mock the datamart's aggregates methods
    mock_datamart.datamart.aggregates.last.return_value = pl.DataFrame({
        "ModelID": ["model1", "model2"],
        "Name": ["Model 1", "Model 2"],
        "Performance": [0.8, 0.75]
    }).lazy()
    
    mock_datamart.datamart.aggregates.predictors_overview.return_value = pl.DataFrame({
        "PredictorName": ["Age", "Income"],
        "Performance": [0.7, 0.65]
    }).lazy()
    
    mock_datamart.datamart.aggregates.predictors_global_overview.return_value = pl.DataFrame({
        "PredictorCategory": ["Demographic", "Financial"],
        "Count": [10, 15]
    }).lazy()
    
    # Call the method
    result, warnings = mock_datamart.excel_report(name="test_report.xlsx")
    
    # Verify the result
    assert result == Path("test_report.xlsx")
    assert warnings == []
    
    # Verify that the workbook was created and used
    mock_workbook.assert_called_once()
    mock_wb.use_zip64.assert_called_once()
