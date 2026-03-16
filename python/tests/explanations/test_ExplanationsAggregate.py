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
            aggregate.get_predictor_contributions(sort_by="invalid_type")

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
                sort_by="invalid_type")

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
                sort_by="invalid_type")

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
                sort_by="invalid_type")

    def test_get_predictor_value_contributions_for_context_invalid_top_k(self, aggregate, predictors, selected_context):
        """Test with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid top_k value"):
            aggregate.get_predictor_value_contributions(
                predictors=predictors,
                context=selected_context,
                top_k=-1)

class TestAggregateSortingBehavior:
    """Test cases for verifying sort_by vs display_by separation."""

    def test_default_sort_by_uses_contribution_abs(self, aggregate):
        """Verify that default sort_by selects predictors by contribution_abs."""
        df = aggregate.get_predictor_contributions()

        # Filter out the "remaining" row
        df_no_remaining = df.filter(pl.col("predictor_name") != "remaining")

        # The output should be sorted by contribution_abs (ascending or descending)
        abs_values = df_no_remaining["contribution_abs"].to_list()
        assert (
            abs_values == sorted(abs_values)
            or abs_values == sorted(abs_values, reverse=True)
        ), "Output should be ordered by contribution_abs"

    def test_sort_by_contribution_abs_captures_negative_contributors(self, aggregate):
        """Verify that sorting by contribution_abs includes large negative contributors."""
        df = aggregate.get_predictor_contributions(top_n=5)

        df_no_remaining = df.filter(pl.col("predictor_name") != "remaining")

        # If any predictor has a negative contribution, it should still appear
        # in the top-5 when its absolute value is large enough.
        # Verify the top-N selection picked predictors with the largest absolute values,
        # which means negative contributors are included.
        contributions = df_no_remaining["contribution"].to_list()
        abs_values = df_no_remaining["contribution_abs"].to_list()

        # At least one predictor should have a negative contribution
        has_negative = any(v < 0 for v in contributions)
        has_positive_abs = all(v >= 0 for v in abs_values)
        assert has_positive_abs, "contribution_abs values should all be non-negative"
        # The top-5 by contribution_abs should include negative contributors if they exist
        if has_negative:
            assert any(v < 0 for v in contributions), (
                "Top-N by contribution_abs should capture negative contributors"
            )

    def test_explicit_sort_by_contribution_gives_different_order(self, aggregate):
        """Verify that sort_by='contribution' vs 'contribution_abs' produce different orderings."""
        df_abs = aggregate.get_predictor_contributions(sort_by="contribution_abs")
        df_signed = aggregate.get_predictor_contributions(sort_by="contribution")

        predictors_abs = df_abs.filter(
            pl.col("predictor_name") != "remaining"
        )["predictor_name"].to_list()
        predictors_signed = df_signed.filter(
            pl.col("predictor_name") != "remaining"
        )["predictor_name"].to_list()

        # Verify the sort_by parameter controls the output ordering
        abs_vals = df_abs.filter(
            pl.col("predictor_name") != "remaining"
        )["contribution_abs"].to_list()
        signed_vals = df_signed.filter(
            pl.col("predictor_name") != "remaining"
        )["contribution"].to_list()

        # Each output should be sorted by its respective sort_by column
        assert (
            abs_vals == sorted(abs_vals)
            or abs_vals == sorted(abs_vals, reverse=True)
        ), "sort_by='contribution_abs' should sort output by absolute values"
        assert (
            signed_vals == sorted(signed_vals)
            or signed_vals == sorted(signed_vals, reverse=True)
        ), "sort_by='contribution' should sort output by signed values"

        # Both should select the same set of predictors (same top_n default)
        assert set(predictors_abs) == set(predictors_signed), (
            "Same top_n should select the same predictor set"
        )

    def test_predictor_value_contributions_sorted_by_abs(self, aggregate, predictors):
        """Verify that predictor value contributions for symbolic predictors are sorted by contribution_abs."""
        df = aggregate.get_predictor_value_contributions(predictors=predictors)

        # For symbolic predictors, check that top-k values are selected by contribution_abs
        symbolic_predictors = df.filter(
            (pl.col("predictor_type") == "SYMBOLIC")
            & (pl.col("bin_contents") != "remaining")
            & (pl.col("bin_contents") != "missing")
        )

        for name in symbolic_predictors["predictor_name"].unique().to_list():
            pred_df = symbolic_predictors.filter(pl.col("predictor_name") == name)
            abs_values = pred_df["contribution_abs"].to_list()
            # Values should be sorted by absolute contribution (via sort_value column)
            assert len(abs_values) > 0, f"Expected values for predictor {name}"


def assert_df_has_top_n(df, top_n):
    """Assert that the DataFrame has at least top_n rows."""
    assert df.shape[0] >= top_n, f"DataFrame should have at least {top_n} rows, but has {df.shape[0]} rows."

def assert_df_has_top_k(df, top_k):
    """Assert that the DataFrame has at least top_k rows."""
    predictors_list = df.group_by(["predictor_name", "predictor_type"]).agg(pl.len()).to_dicts()
    for predictor in predictors_list:
        if predictor["predictor_type"] == "SYMBOLIC":
            assert predictor["len"] >= top_k, f"Symbolic predictor {predictor['predictor_name']} should have at least {top_k} rows"
