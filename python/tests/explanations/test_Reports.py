import os
import pytest
from unittest.mock import MagicMock, patch
from pdstools.explanations.Reports import Reports


@pytest.fixture
def reports():
    """Fixture to create a mock Reports object"""
    mock_explanations = MagicMock()
    mock_explanations.root_dir = "/test/root"
    mock_explanations.report_folder = "reports"
    mock_explanations.aggregates_folder = "/test/aggregates"
    return Reports(mock_explanations)

def test_copy_report_resources_calls_with_correct_args(reports):
    with patch("pdstools.explanations.Reports.copy_report_resources") as mock_copy:
        reports._copy_report_resources()
        expected_resource_dict = [
            ("GlobalExplanations", reports.report_dir),
            ("assets", os.path.join(reports.report_dir, "assets")),
        ]
        mock_copy.assert_called_once_with(resource_dict=expected_resource_dict)

def test_copy_report_resources_raises_on_error(reports):
    with patch("pdstools.explanations.Reports.copy_report_resources", side_effect=OSError("fail")):
        with pytest.raises(OSError):
            reports._copy_report_resources()



    