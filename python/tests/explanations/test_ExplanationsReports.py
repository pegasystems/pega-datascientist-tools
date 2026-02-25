"""Test cases for the Reports class that handles generating reports from aggregated data."""

import os
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from pdstools.explanations import Explanations

basePath = Path(__file__).parent.parent.parent.parent


def clean_up(root_dir):
    _root_dir = Path(f"{basePath}/{root_dir}")
    if _root_dir.exists():
        for file in _root_dir.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                # Remove subdirectories recursively
                shutil.rmtree(file)
        _root_dir.rmdir()


@pytest.fixture(scope="module")
def reports():
    """Fixture to serve as class to call functions from."""
    explanations = Explanations(
        data_folder=f"{basePath}/data/explanations",
        model_name="AdaptiveBoostCT",
        from_date=datetime(2025, 3, 28),
        to_date=datetime(2025, 3, 28),
    )
    yield explanations.report

    # cleanup .tmp folder
    clean_up(explanations.root_dir)


def test_validate_report_dir(reports):
    """Test the report directory validation."""
    reports._validate_report_dir()

    assert os.path.exists(reports.report_folderpath), "Report folder does not exist."
    assert os.path.isdir(reports.report_folderpath), "Report folder is not a directory."


def test_copy_report_resources(reports):
    """Test the copy_report_resources method."""
    reports._validate_report_dir()
    reports._copy_report_resources()

    assert os.path.exists(reports.report_folderpath), "Report folder does not exist."
    assert any(os.scandir(reports.report_folderpath)), "Report folder is empty."

    assets_folder = os.path.join(reports.report_folderpath, "assets")
    assert os.path.exists(assets_folder), "Assets folder not copied."
    assert any(os.scandir(assets_folder)), "Assets folder is empty."


def test_copy_report_resources_raises_on_error(reports):
    with patch(
        "pdstools.explanations.Reports.copy_report_resources",
        side_effect=OSError("fail"),
    ):
        with pytest.raises(OSError):
            reports._copy_report_resources()


def test_set_params(reports):
    """Test the _set_params method."""
    reports._validate_report_dir()
    reports._copy_report_resources()
    reports._set_params(top_n=5, top_k=3, verbose=True)

    with open(reports.params_file, encoding="Utf-8") as f:
        params = f.read()

    assert "top_n: 5" in params
    assert "top_k: 3" in params
    assert "verbose: true" in params
