"""Test cases for the Reports class that handles generating reports from aggregated data."""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pdstools.explanations import Explanations
from pdstools.explanations.ExplanationsUtils import _CONTRIBUTION_TYPE, defaults

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
    """Test _set_params writes all parameters including sort_by and display_by."""
    reports._validate_report_dir()
    reports._copy_report_resources()
    reports._set_params(top_n=5, top_k=3, from_date="2026-01-01", to_date="2026-01-31")

    with open(reports.params_file, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    assert params["top_n"] == 5
    assert params["top_k"] == 3
    assert params["from_date"] == "2026-01-01"
    assert params["to_date"] == "2026-01-31"
    assert params["sort_by"] == defaults.sort_by.value
    assert params["sort_by_text"] == defaults.sort_by.text
    assert params["display_by"] == defaults.display_by.value
    assert params["display_by_text"] == defaults.display_by.text
    assert params["data_folder"] == reports.aggregate_folder.name


def test_set_params_custom_contribution_types(reports):
    """Test _set_params writes custom sort_by and display_by values."""
    reports._validate_report_dir()
    reports._copy_report_resources()

    sort_by = _CONTRIBUTION_TYPE.CONTRIBUTION_ABS
    display_by = _CONTRIBUTION_TYPE.CONTRIBUTION_ABS

    reports._set_params(
        top_n=10,
        top_k=5,
        from_date="2026-03-01",
        to_date="2026-03-31",
        sort_by=sort_by,
        display_by=display_by,
    )

    with open(reports.params_file, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    assert params["top_n"] == 10
    assert params["top_k"] == 5
    assert params["sort_by"] == sort_by.value
    assert params["sort_by_text"] == sort_by.text
    assert params["display_by"] == display_by.value
    assert params["display_by_text"] == display_by.text


def test_reports_logging(reports, caplog):
    """Test that report operations produce debug logs when logging enabled."""
    reports._validate_report_dir()

    with caplog.at_level(logging.DEBUG):
        reports._copy_report_resources()
        reports._set_params(top_n=5, top_k=3)

    # Should have debug messages from the operations
    debug_messages = [r.message for r in caplog.records if r.levelname == "DEBUG"]
    assert len(debug_messages) > 0, "Expected debug log messages"
