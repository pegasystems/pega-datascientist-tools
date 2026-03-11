import polars as pl
import pytest
from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer


@pytest.fixture
def sample_data():
    """Create sample decision data with multiple categorical dimensions."""
    return pl.LazyFrame(
        {
            "pyIssue": ["Sales", "Retention", "Service"] * 10,
            "pyGroup": ["Cards", "Loans", "Insurance"] * 10,
            "pyName": ["Action1", "Action2", "Action3"] * 10,
            "pyTreatment": ["Email", "SMS", "Web"] * 10,
            "pyChannel": ["Web", "Email", "Mobile"] * 10,
            "pyDirection": ["Inbound", "Outbound", "Inbound"] * 10,
            "Stage Group": ["Filtering", "Arbitration", "Filtering"] * 10,
            "Stage": ["Stage1", "Stage2", "Stage3"] * 10,
            "Priority": [0.5, 0.6, 0.7] * 10,
            "Propensity": [0.1, 0.2, 0.3] * 10,
            "Interaction ID": list(range(30)),
            "Decision Time": ["2024-01-01"] * 30,
        }
    )


def test_color_mappings_property_exists(sample_data):
    """Test that color_mappings property exists and returns a dict."""
    da = DecisionAnalyzer(sample_data)
    assert hasattr(da, "color_mappings")
    mappings = da.color_mappings
    assert isinstance(mappings, dict)


def test_color_mappings_includes_all_categorical_columns(sample_data):
    """Test that color_mappings includes all categorical dimensions."""
    da = DecisionAnalyzer(sample_data)
    mappings = da.color_mappings

    # Check that key dimensions present in the processed data are included
    # Stage Group is always present, Stage only in v2 data
    expected_columns = ["Issue", "Group", "Action", "Channel", "Direction", "Stage Group"]
    for col in expected_columns:
        assert col in mappings, f"Missing {col} in color_mappings"
        assert isinstance(mappings[col], dict), f"{col} mapping should be a dict"

    # Optional columns - only check if present in the data
    optional_columns = ["Treatment", "Stage"]
    for col in optional_columns:
        if col in da.decision_data.collect_schema().names():
            assert col in mappings, f"{col} is in data but missing from color_mappings"


def test_color_mappings_assigns_consistent_colors(sample_data):
    """Test that colors are assigned consistently based on sorted unique values."""
    da = DecisionAnalyzer(sample_data)
    mappings = da.color_mappings

    # Issue should map sorted values to colors
    issue_mapping = mappings["Issue"]
    assert "Sales" in issue_mapping
    assert "Retention" in issue_mapping
    assert "Service" in issue_mapping

    # All values should have color strings (hex codes)
    for color in issue_mapping.values():
        assert isinstance(color, str)
        assert color.startswith("#")


def test_color_mappings_is_cached(sample_data):
    """Test that color_mappings is computed only once (cached)."""
    da = DecisionAnalyzer(sample_data)

    # Access twice, should return same object (same id)
    mappings1 = da.color_mappings
    mappings2 = da.color_mappings
    assert mappings1 is mappings2
