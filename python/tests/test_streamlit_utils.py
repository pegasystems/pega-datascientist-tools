"""Tests for streamlit utility functions.
Focuses on testable utilities, not UI-heavy functions.
"""

import datetime
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from pdstools.utils import streamlit_utils


@pytest.fixture
def mock_spinner():
    """Fixture to provide a properly configured spinner mock."""
    mock = MagicMock()
    mock.return_value.__enter__ = MagicMock()
    mock.return_value.__exit__ = MagicMock()
    return mock


class TestDataProcessingUtilities:
    """Test pure data processing functions - no Streamlit mocking needed."""

    def test_model_and_row_counts_dataframe(self):
        """Test counting with regular DataFrame."""
        df = pl.DataFrame(
            {"ModelID": ["M1", "M1", "M2", "M2", "M3"], "Value": [1, 2, 3, 4, 5]},
        )

        unique_count, row_count = streamlit_utils.model_and_row_counts(df)

        assert unique_count == 3
        assert row_count == 5

    def test_model_and_row_counts_lazyframe(self):
        """Test counting with LazyFrame."""
        df = pl.LazyFrame(
            {"ModelID": ["M1", "M2", "M3", "M1"], "Data": [10, 20, 30, 40]},
        )

        unique_count, row_count = streamlit_utils.model_and_row_counts(df)

        assert unique_count == 3
        assert row_count == 4

    def test_model_selection_df(self):
        """Test model selection dataframe generation."""
        df = pl.LazyFrame(
            {
                "ModelID": ["M1", "M2", "M3"],
                "Configuration": ["C1", "C1", "C2"],
                "Name": ["Action3", "Action1", "Action2"],
                "Channel": ["Web", "Mobile", "Web"],
            },
        )
        context_keys = ["Name", "Channel"]

        result = streamlit_utils.model_selection_df(df, context_keys)

        # Should be a DataFrame (collected)
        assert isinstance(result, pl.DataFrame)
        # Should have Generate Report column
        assert "Generate Report" in result.columns
        # Should have all original columns
        assert all(col in result.columns for col in ["ModelID", "Configuration", "Name", "Channel"])
        # Generate Report should be False by default
        assert result["Generate Report"].to_list() == [False, False, False]
        # Should be sorted by Name
        assert result["Name"].to_list() == ["Action1", "Action2", "Action3"]


class TestCachingFunctions:
    """Test cached functions with minimal mocking."""

    @patch("streamlit.cache_resource", lambda f: f)
    def test_cached_sample(self):
        """Test loading sample data."""
        result = streamlit_utils.cached_sample()

        assert result is not None
        assert hasattr(result, "model_data")
        assert hasattr(result, "predictor_data")

    @patch("streamlit.cache_resource", lambda f: f)
    def test_cached_sample_prediction(self):
        """Test loading sample prediction data."""
        result = streamlit_utils.cached_sample_prediction()

        assert result is not None
        assert hasattr(result, "predictions")
        assert result.is_available

    @patch("streamlit.cache_resource", lambda f: f)
    @patch("streamlit.spinner")
    @patch("pdstools.utils.streamlit_utils.ADMDatamart.from_ds_export")
    def test_cached_datamart_success(self, mock_load, mock_spinner):
        """Test successful datamart loading."""
        mock_spinner.return_value.__enter__ = MagicMock()
        mock_spinner.return_value.__exit__ = MagicMock()

        # Mock a successful load
        mock_dm = MagicMock()
        mock_dm.model_data = pl.LazyFrame({"ModelID": ["M1"]})
        mock_load.return_value = mock_dm

        result = streamlit_utils.cached_datamart(
            model_filename="test.zip",
            predictor_filename="test.zip",
        )

        assert result is not None
        assert result == mock_dm
        mock_load.assert_called_once()

    @patch("streamlit.cache_resource", lambda f: f)
    @patch("streamlit.spinner")
    @patch("streamlit.warning")
    @patch("pdstools.utils.streamlit_utils.ADMDatamart.from_ds_export")
    def test_cached_datamart_returns_none(self, mock_load, mock_warning, mock_spinner):
        """Test when datamart loading returns None."""
        mock_spinner.return_value.__enter__ = MagicMock()
        mock_spinner.return_value.__exit__ = MagicMock()

        # Mock returning None
        mock_load.return_value = None

        result = streamlit_utils.cached_datamart(model_filename="test.zip")

        assert result is None
        mock_warning.assert_called_once_with("Unable to load datamart.")

    @patch("streamlit.cache_resource", lambda f: f)
    @patch("streamlit.spinner")
    @patch("streamlit.error")
    @patch("pdstools.utils.streamlit_utils.ADMDatamart.from_ds_export")
    def test_cached_datamart_error_handling(self, mock_load, mock_error, mock_spinner):
        """Test error handling in datamart loading."""
        mock_spinner.return_value.__enter__ = MagicMock()
        mock_spinner.return_value.__exit__ = MagicMock()

        # Mock an exception
        mock_load.side_effect = Exception("Load failed")

        result = streamlit_utils.cached_datamart(model_filename="bad.zip")

        assert result is None
        assert mock_error.called
        error_message = mock_error.call_args[0][0]
        assert "Load failed" in error_message

    @patch("streamlit.cache_resource", lambda f: f)
    @patch("streamlit.spinner")
    @patch("pdstools.utils.streamlit_utils.Prediction.from_ds_export")
    def test_cached_prediction_table_success(self, mock_load, mock_spinner):
        """Test successful prediction table loading."""
        mock_spinner.return_value.__enter__ = MagicMock()
        mock_spinner.return_value.__exit__ = MagicMock()

        # Mock a successful load
        mock_pred = MagicMock()
        mock_pred.predictions = pl.LazyFrame({"pyModelId": ["M1"]})
        mock_load.return_value = mock_pred

        result = streamlit_utils.cached_prediction_table(
            predictions_filename="test.zip",
        )

        assert result is not None
        assert result == mock_pred
        mock_load.assert_called_once()

    @patch("streamlit.cache_resource", lambda f: f)
    @patch("streamlit.spinner")
    @patch("streamlit.error")
    @patch("pdstools.utils.streamlit_utils.Prediction.from_ds_export")
    def test_cached_prediction_table_error_handling(
        self,
        mock_load,
        mock_error,
        mock_spinner,
    ):
        """Test error handling in prediction table loading."""
        mock_spinner.return_value.__enter__ = MagicMock()
        mock_spinner.return_value.__exit__ = MagicMock()

        # Mock an exception
        mock_load.side_effect = Exception("Load failed")

        result = streamlit_utils.cached_prediction_table(predictions_filename="bad.zip")

        assert result is None
        assert mock_error.called
        error_message = mock_error.call_args[0][0]
        assert "Load failed" in error_message

    @patch("streamlit.cache_data", lambda f: f)
    def test_convert_df(self):
        """Test dataframe to CSV conversion."""
        df = pl.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})

        result = streamlit_utils.convert_df(df)

        assert isinstance(result, bytes)
        assert len(result) > 0
        # Verify it's actually CSV data
        assert b"A" in result  # Column header
        assert b"1" in result  # Data

    @patch("streamlit.cache_data", lambda f: f)
    @patch("pdstools.utils.cdh_utils.get_latest_pdstools_version")
    def test_st_get_latest_pdstools_version(self, mock_version):
        """Test getting latest pdstools version."""
        mock_version.return_value = "4.0.0"

        result = streamlit_utils.st_get_latest_pdstools_version()

        assert result == "4.0.0"
        mock_version.assert_called_once()


class TestFilterDataframe:
    """Test the filter_dataframe function with various column types."""

    @pytest.fixture(autouse=True)
    def setup_session_state(self):
        """Reset session state before each test."""
        with patch("streamlit.session_state", {}) as mock_state:
            yield mock_state

    @patch("streamlit.multiselect")
    @patch("streamlit.columns")
    def test_filter_categorical_column(self, mock_cols, mock_multi):
        """Test filtering on categorical column."""
        # Setup mocks
        mock_multi.return_value = ["Channel"]
        left_col = MagicMock()
        right_col = MagicMock()
        right_col.multiselect.return_value = ["Web"]
        mock_cols.return_value = (left_col, right_col)

        # Create test data
        df = pl.LazyFrame({"Channel": ["Web", "Mobile", "Email"], "Value": [1, 2, 3]})

        # Need to properly mock session_state to simulate the filtering logic
        session_state = {
            "categories_Channel": ["Web", "Mobile", "Email"],
            "selected_Channel": ["Web"],  # This is what determines the filter
        }
        with patch("streamlit.session_state", session_state):
            queries = streamlit_utils.filter_dataframe(df, queries=[])

            # Should create a filter query
            assert len(queries) > 0
            # Verify the filter actually works
            filtered_df = df.filter(queries[0]).collect()
            assert filtered_df["Channel"].to_list() == ["Web"]

    @patch("streamlit.multiselect")
    @patch("streamlit.columns")
    def test_filter_numeric_column_with_slider(self, mock_cols, mock_multi):
        """Test filtering on numeric column with slider (small range)."""
        mock_multi.return_value = ["Score"]
        left_col = MagicMock()
        right_col = MagicMock()
        # Need to mock columns() call that happens in the numeric branch
        right_col.columns.return_value = (MagicMock(), MagicMock())
        right_col.slider.return_value = (20, 80)
        mock_cols.return_value = (left_col, right_col)

        df = pl.LazyFrame({"Score": [10, 50, 90, 100], "Name": ["A", "B", "C", "D"]})

        with patch("streamlit.session_state", {}):
            queries = streamlit_utils.filter_dataframe(df)

            assert len(queries) == 1
            assert isinstance(queries[0], pl.Expr)
            filtered_df = df.filter(queries[0]).collect()
            # Should filter to values between 20 and 80
            assert filtered_df["Score"].to_list() == [50]

    @patch("streamlit.multiselect")
    @patch("streamlit.columns")
    def test_filter_large_range_numeric(self, mock_cols, mock_multi):
        """Test filtering on numeric column with large range (uses number_input)."""
        mock_multi.return_value = ["BigValue"]
        left_col = MagicMock()
        right_col = MagicMock()
        min_col = MagicMock()
        max_col = MagicMock()
        min_col.number_input.return_value = 1000
        max_col.number_input.return_value = 5000
        right_col.columns.return_value = (min_col, max_col)
        mock_cols.return_value = (left_col, right_col)

        df = pl.LazyFrame({"BigValue": [1000, 3000, 5000, 10000], "ID": [1, 2, 3, 4]})

        with patch("streamlit.session_state", {}):
            queries = streamlit_utils.filter_dataframe(df, queries=[])

            assert len(queries) == 1
            filtered_df = df.filter(queries[0]).collect()
            assert all(1000 <= val <= 5000 for val in filtered_df["BigValue"].to_list())
            assert filtered_df["BigValue"].to_list() == [1000, 3000, 5000]

    @patch("streamlit.multiselect")
    @patch("streamlit.columns")
    def test_filter_date_column(self, mock_cols, mock_multi):
        """Test filtering on date column."""
        mock_multi.return_value = ["Date"]
        left_col = MagicMock()
        right_col = MagicMock()
        right_col.date_input.return_value = (
            datetime.date(2023, 3, 1),
            datetime.date(2023, 9, 30),
        )
        mock_cols.return_value = (left_col, right_col)

        df = pl.LazyFrame(
            {
                "Date": [
                    datetime.date(2023, 1, 1),
                    datetime.date(2023, 6, 1),
                    datetime.date(2023, 12, 31),
                ],
                "Event": ["A", "B", "C"],
            },
        )

        with patch("streamlit.session_state", {}):
            queries = streamlit_utils.filter_dataframe(df, queries=[])

            assert len(queries) > 0
            filtered_df = df.filter(queries[0]).collect()
            # Only June 1 should be in range
            assert filtered_df["Event"].to_list() == ["B"]

    @patch("streamlit.multiselect")
    @patch("streamlit.columns")
    def test_filter_large_category_list_uses_text_input(self, mock_cols, mock_multi):
        """Test that large category lists use text input instead of multiselect."""
        mock_multi.return_value = ["LargeText"]
        left_col = MagicMock()
        right_col = MagicMock()
        right_col.text_input.return_value = "X1"
        mock_cols.return_value = (left_col, right_col)

        # Create data with many unique categories
        df = pl.LazyFrame(
            {
                "LargeText": ["A", "B", "C", "X10", "X20", "X100"] + [f"X{i}" for i in range(200)],
                "Value": list(range(206)),
            },
        )

        with patch(
            "streamlit.session_state",
            {
                "categories_LargeText": ["A", "B", "C", "X10", "X20", "X100"] + [f"X{i}" for i in range(200)],
                "selected_LargeText": ["A", "B"],
            },
        ):
            queries = streamlit_utils.filter_dataframe(df, queries=[])

            # Should create a query using text input
            assert len(queries) == 1
            # Verify the filter works - should match items containing "X1"
            filtered_df = df.filter(queries[0]).collect()
            filtered_values = filtered_df["LargeText"].to_list()
            # Should include X10, X100, X1, X11-X19, etc.
            assert "X10" in filtered_values
            assert "X100" in filtered_values


class TestConfigurePredictorCategorization:
    """Test predictor categorization (UI function but can test data flow)."""

    @patch("streamlit.session_state", {"filters": [], "dm": None})
    @patch("streamlit.plotly_chart")
    @patch("pdstools.utils.cdh_utils.weighted_average_polars")
    def test_configure_predictor_categorization_with_data(
        self,
        mock_weighted_avg,
        mock_chart,
    ):
        """Test predictor categorization with mocked data."""
        # Setup weighted average to return a reasonable aggregation
        mock_weighted_avg.return_value = pl.col("PredictorPerformance").mean()

        # Create mock datamart
        mock_dm = MagicMock()
        mock_dm.combinedData = pl.LazyFrame(
            {
                "PredictorName": ["Pred1", "Pred2", "Pred3", "Classifier"],
                "PredictorCategory": ["Cat1", "Cat2", "Cat1", "System"],
                "PredictorPerformance": [0.6, 0.7, 0.55, 0.5],
                "BinResponseCount": [100, 200, 150, 50],
            },
        )

        with patch("streamlit.session_state", {"filters": [], "dm": mock_dm}):
            # Execute the function
            streamlit_utils.configure_predictor_categorization()

            # Verify plotly_chart was called
            assert mock_chart.called, "plotly_chart should have been called"

            # Verify weighted_average_polars was called
            assert mock_weighted_avg.called, "weighted_average_polars should have been called"

            # Get the figure that was passed to plotly_chart
            chart_call_args = mock_chart.call_args[0][0]
            assert chart_call_args is not None, "Chart should have been created"
