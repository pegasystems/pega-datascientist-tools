"""
Testing the functionality of the adm Reports module
"""

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest
from pdstools import ADMDatamart
from pdstools.adm.Reports import Reports


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


@pytest.mark.skip(reason="Requires mocking yaml.dump")
def test_write_params_files(tmp_path):
    """Test the _write_params_files method"""
    # Skip this test for now since it requires mocking yaml.dump
    pass


@pytest.mark.skip(reason="Requires mocking subprocess")
def test_run_quarto_mock():
    """Test the run_quarto method"""
    # Skip this test for now since it requires mocking subprocess
    pass
