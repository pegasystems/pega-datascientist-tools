"""Test cases for Aggregate class that handles loading and processing of aggregate data."""

import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from pdstools.explanations import Explanations

basePath = Path(__file__).parent.parent.parent.parent

def clean_up(root_dir):
    _root_dir = Path(f'{basePath}/{root_dir}')
    if _root_dir.exists():
        for file in _root_dir.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                # Remove subdirectories recursively
                shutil.rmtree(file)
        _root_dir.rmdir()

@pytest.fixture(scope="class")
def aggregate():
    """Fixture to serve as class to call functions from."""

    explanations = Explanations(
        data_folder=f'{basePath}/data/explanations',
        model_name="AdaptiveBoostCT",
        from_date=datetime(2025, 3, 28),
        to_date=datetime(2025, 3, 28),
    )
    yield explanations.aggregate

    # cleanup .tmp folder
    clean_up(explanations.root_dir)

@pytest.fixture
def selected_context():
    """Fixture to provide a selected context for testing."""
    return {
        "pyChannel": "PegaBatch",
        "pyDirection": "E2E Test",
        "pyGroup": "E2E Test",
        "pyIssue": "Batch",
        "pyName": "P1",
    }

@pytest.fixture
def predictors():
    """Fixture to provide a list of predictors for testing."""
    return ["Age", "EyeColor"]


class TestAggregateLoadData:
    """Test cases for Aggregate.load_data method."""

    def test_initial_state(self, aggregate):
        """Test the initial state of Aggregate before load_data is called."""
        assert aggregate.initialized is False
        assert aggregate.df_contextual is None
        assert aggregate.df_overall is None

    def test_load_data_success(self, aggregate):
        """Test successful data loading."""

        aggregate.initialized
        aggregate._load_data()

        assert aggregate.initialized is True
        assert aggregate.df_contextual is not None
        assert aggregate.df_overall is not None
    
    def test_load_data_folder_does_not_exist(self, aggregate):
        """Test handling of non-existent aggregates folder."""

        aggregate.initialized = False
        aggregate.data_folderpath = "/non/existent/path"
        with pytest.raises(FileNotFoundError):
            aggregate._load_data()
            assert aggregate.initialized is False

    @patch("polars.scan_parquet")
    def test_load_data_file_not_found_error(self, mock_scan_parquet, aggregate):
        """Test handling of file not found errors."""
        aggregate.data_folderpath = "/non/existent/path"
        mock_scan_parquet.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            aggregate._load_data()
            assert aggregate.initialized is False

class TestAggregatePredictorContributions:
    """Test cases for Aggregate contribution methods."""
    
    def test_get_predictor_contributions_overall_default_params(self, aggregate):
        """Test get_predictor_contributions with default parameters."""

        df = aggregate.get_predictor_contributions()
        assert isinstance(df, pl.DataFrame)

    def test_get_predictor_contributions_overall_custom_params(self, aggregate):
        """Test get_predictor_contributions with custom parameters."""

        df = aggregate.get_predictor_contributions(
            top_n=3)

        assert isinstance(df, pl.DataFrame)
        assert_df_has_top_n(df, 3)

    def test_get_predictor_contributions_overall_invalid_contribution_type(self, aggregate):
        """Test contribution type validation."""
        with pytest.raises(ValueError, match="Invalid contribution type"):
            aggregate.get_predictor_contributions(contribution_calculation="invalid_type")

    def test_get_predictor_contributions_overall_invalid_top_n(self, aggregate):
        """Test with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid top_n value"):
            aggregate.get_predictor_contributions(top_n=-1)

    def test_get_predictor_contributions_for_context_default_params(self, aggregate, selected_context):
        """Test get_predictor_contributions with default parameters."""

        df = aggregate.get_predictor_contributions(context=selected_context)
        assert isinstance(df, pl.DataFrame)

    def test_get_predictor_contributions_for_context_custom_params(self, aggregate, selected_context):
        """Test get_predictor_contributions with custom parameters."""

        df = aggregate.get_predictor_contributions(
            context=selected_context,
            top_n=3)

        assert isinstance(df, pl.DataFrame)
        assert_df_has_top_n(df, 3)

    def test_get_predictor_contributions_for_context_invalid_contribution_type(self, aggregate, selected_context):
        """Test contribution type validation."""
        with pytest.raises(ValueError, match="Invalid contribution type"):
            aggregate.get_predictor_contributions(
                context=selected_context,
                contribution_calculation="invalid_type")

    def test_get_predictor_contributions_for_context_invalid_top_n(self, aggregate, selected_context):
        """Test with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid top_n value"):
            aggregate.get_predictor_contributions(
                context=selected_context,
                top_n=-1)

class TestAggregatePredictorValueContributions:
    """Test cases for Aggregate predictor value contributions."""

    def test_get_predictor_value_contributions_overall_default_params(self, aggregate, predictors):
        """Test get_predictor_value_contributions with default parameters."""

        df = aggregate.get_predictor_value_contributions(predictors=predictors)
        assert isinstance(df, pl.DataFrame)

    def test_get_predictor_value_contributions_overall_custom_params(self, aggregate, predictors):
        """Test get_predictor_value_contributions with custom parameters."""

        df = aggregate.get_predictor_value_contributions(
            predictors=predictors,
            top_k=3)

        assert isinstance(df, pl.DataFrame)
        assert_df_has_top_k(df, 3)

    def test_get_predictor_value_contributions_overall_invalid_contribution_type(self, aggregate, predictors):
        """Test contribution type validation."""
        with pytest.raises(ValueError, match="Invalid contribution type"):
            aggregate.get_predictor_value_contributions(
                predictors=predictors,
                contribution_calculation="invalid_type")

    def test_get_predictor_value_contributions_overall_invalid_top_k(self, aggregate, predictors):
        """Test with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid top_k value"):
            aggregate.get_predictor_value_contributions(
                predictors=predictors,
                top_k=-1)
    
    def test_get_predictor_value_contributions_for_context_default_params(self, aggregate, predictors, selected_context):
        """Test get_predictor_value_contributions with default parameters."""

        df = aggregate.get_predictor_value_contributions(
            predictors=predictors,
            context=selected_context
            )
        assert isinstance(df, pl.DataFrame)

    def test_get_predictor_value_contributions_for_context_custom_params(self, aggregate, predictors, selected_context):
        """Test get_predictor_value_contributions with custom parameters."""

        df = aggregate.get_predictor_value_contributions(
            predictors=predictors,
            context=selected_context,
            top_k=3)

        assert isinstance(df, pl.DataFrame)
        assert_df_has_top_k(df, 3)

    def test_get_predictor_value_contributions_for_context_invalid_contribution_type(self, aggregate, predictors, selected_context):
        """Test contribution type validation."""
        with pytest.raises(ValueError, match="Invalid contribution type"):
            aggregate.get_predictor_value_contributions(
                predictors=predictors,
                context=selected_context,
                contribution_calculation="invalid_type")

    def test_get_predictor_value_contributions_for_context_invalid_top_k(self, aggregate, predictors, selected_context):
        """Test with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid top_k value"):
            aggregate.get_predictor_value_contributions(
                predictors=predictors,
                context=selected_context,
                top_k=-1)

def assert_df_has_top_n(df, top_n):
    """Assert that the DataFrame has at least top_n rows."""
    assert df.shape[0] >= top_n, f"DataFrame should have at least {top_n} rows, but has {df.shape[0]} rows."

def assert_df_has_top_k(df, top_k):
    """Assert that the DataFrame has at least top_k rows."""
    predictors_list = df.group_by(["predictor_name", "predictor_type"]).agg(pl.len()).to_dicts()
    for predictor in predictors_list:
        if predictor["predictor_type"] == "SYMBOLIC":
            assert predictor["len"] >= top_k, f"Symbolic predictor {predictor['predictor_name']} should have at least {top_k} rows"
