"""Test cases for the Preprocess class that handles preprocessing (generates aggregations) of explanations data."""

import os
import shutil
from datetime import datetime
from pathlib import Path
from unittest import mock

import duckdb
import pytest
from pdstools.explanations import Explanations
from pdstools.explanations.ExplanationsUtils import _PREDICTOR_TYPE, _TABLE_NAME
from pdstools.explanations.Preprocess import Preprocess

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


@pytest.fixture
def preprocess_instance():
    """Fixture to serve as class to call functions from."""
    explanations = Explanations(
        data_folder=f"{basePath}/data/explanations",
        model_name="AdaptiveBoostCT",
        from_date=datetime(2025, 3, 28),
        to_date=datetime(2025, 3, 28),
    )
    yield explanations.preprocess

    # cleanup .tmp folder
    clean_up(explanations.root_dir)


@pytest.fixture
def empty_folder():
    """Fixture to create an empty data folder."""
    os.makedirs("empty_folder", exist_ok=True)
    yield "empty_folder"

    # Cleanup after test
    if os.path.exists("empty_folder"):
        shutil.rmtree("empty_folder")


@pytest.fixture
def dummy_explanations_data():
    """Fixture to create a dummy explanations data folder."""
    os.makedirs("explanations_data", exist_ok=True)
    file_path = "explanations_data/dummy_explanations.parquet"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("dummy data")
    yield file_path

    # Cleanup after test
    if os.path.exists("explanations_data"):
        shutil.rmtree("explanations_data")


@pytest.fixture
def dummy_aggregate_data():
    """Fixture to create a dummy aggregate data folder."""
    os.makedirs(".tmp/aggregated_data", exist_ok=True)
    file_path = ".tmp/aggregated_data/dummy_aggregate.parquet"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("dummy data")
    yield file_path

    # Cleanup after test
    if os.path.exists(".tmp"):
        shutil.rmtree(".tmp")


@pytest.fixture
def mock_explanations(tmp_path):
    class MockExplanations:
        data_folder = tmp_path / "data"
        root_dir = tmp_path
        from_date = datetime(2023, 1, 1)
        to_date = datetime(2023, 1, 31)
        model_name = "test_model"

    return MockExplanations()


class TestAggregatesGenerate:
    """Test the generate method and related functionality"""

    def test_generate_invalid_explanations_folder(self):
        with pytest.raises(
            ValueError,
            match="Explanations folder invalid_path does not exist or is empty.",
        ):
            Explanations(data_folder="invalid_path")

    def test_generate_empty_explanations_folder(self, empty_folder):
        """Test that an empty explanations folder raises an error."""
        data_folder = "empty_folder"
        with pytest.raises(
            ValueError,
            match=f"Explanations folder {data_folder} does not exist or is empty.",
        ):
            Explanations(data_folder=data_folder)

    def test_generate_uses_cache(self, dummy_explanations_data, dummy_aggregate_data):
        """Test that the generate method uses cached data if available."""
        with mock.patch("pdstools.explanations.Preprocess.logger") as logger_mock:
            Explanations(data_folder="explanations_data")
            logger_mock.debug.assert_called_with("Using cached data for preprocessing.")

    def test_generate_no_files(self, dummy_explanations_data):
        """Test that generate raises an error when no files are found to aggregate."""
        with pytest.raises(ValueError, match="No files found to aggregate!"):
            Explanations(data_folder="explanations_data")


class TestAggregatesQueryOperations:
    """Test query execution and database operations"""

    def test_execute_query_no_connection(self, preprocess_instance):
        """Test execute_query when connection is not initialized"""
        preprocess_instance._conn = None
        with pytest.raises(ValueError, match="DuckDB connection is not initialized"):
            preprocess_instance._execute_query("SELECT 1")

    def test_execute_query_success(self, preprocess_instance):
        """Test successful query execution"""
        preprocess_instance._conn = duckdb.connect(database=":memory:")
        result = preprocess_instance._execute_query("SELECT 1").df()
        assert result.shape == (1, 1)

    def test_clean_query(self):
        """Test query cleaning functionality"""
        dirty_query = "SELECT  *\nFROM   table\n  WHERE  condition"
        clean_query = Preprocess._clean_query(dirty_query)

        expected = "SELECT * FROM table WHERE condition"
        assert clean_query == expected

    def test_get_table_name_numeric(self):
        """Test getting table name for numeric predictor type"""
        expected_table_name = _TABLE_NAME.NUMERIC
        result = Preprocess._get_table_name(_PREDICTOR_TYPE.NUMERIC)
        assert result is not None
        assert result == expected_table_name

    def test_get_table_name_symbolic(self):
        """Test getting table name for symbolic predictor type"""
        expected_table_name = _TABLE_NAME.SYMBOLIC
        result = Preprocess._get_table_name(_PREDICTOR_TYPE.SYMBOLIC)
        assert result is not None
        assert result == expected_table_name


class TestDataFileSupport:
    """Test data input as a single "data_file" parameter"""

    def test_populate_selected_files_local_path(self, tmp_path):
        """Test _populate_selected_files with local file path"""
        local_file = tmp_path / "test.parquet"
        local_file.write_text("dummy")

        preprocess = Preprocess.__new__(Preprocess)
        preprocess._dependencies_checked = True
        preprocess.data_file = str(local_file)
        preprocess.selected_files = []

        with mock.patch("pdstools.explanations.Preprocess.logger"):
            preprocess._populate_selected_files()
            assert str(local_file) in preprocess.selected_files

    def test_populate_selected_files_https_url(self, tmp_path):
        """Test _populate_selected_files with https URL"""
        preprocess = Preprocess.__new__(Preprocess)
        preprocess._dependencies_checked = True
        preprocess.data_file = "https://example.com/data.parquet"
        preprocess.selected_files = []
        preprocess.explanations = mock.MagicMock()
        preprocess.explanations.root_dir = str(tmp_path)
        preprocess.explanations_folder = Path("explanations")

        with mock.patch.object(
            preprocess,
            "_populate_selected_files_from_url",
        ) as mock_url:
            preprocess._populate_selected_files()
            mock_url.assert_called_once_with("https://example.com/data.parquet")

    def test_populate_selected_files_fallback_to_local(self):
        """Test _populate_selected_files falls back to local when data_file is None"""
        preprocess = Preprocess.__new__(Preprocess)
        preprocess._dependencies_checked = True
        preprocess.data_file = None
        preprocess.selected_files = []

        with mock.patch.object(
            preprocess,
            "_populate_selected_files_from_local",
        ) as mock_local:
            preprocess._populate_selected_files()
            mock_local.assert_called_once()

    def test_populate_selected_files_from_url_success(self, tmp_path):
        """Test _populate_selected_files_from_url downloads and saves file"""
        import polars as pl

        preprocess = Preprocess.__new__(Preprocess)
        preprocess._dependencies_checked = True
        preprocess.explanations = mock.MagicMock()
        preprocess.explanations.root_dir = str(tmp_path)
        preprocess.explanations_folder = Path("explanations")
        preprocess.selected_files = []

        mock_df = pl.DataFrame({"col": [1]})

        with mock.patch(
            "pdstools.pega_io.File.read_ds_export",
            return_value=mock_df.lazy(),
        ):
            with mock.patch("pdstools.explanations.Preprocess.logger"):
                preprocess._populate_selected_files_from_url(
                    "https://example.com/file.parquet",
                )
                assert len(preprocess.selected_files) == 1

    def test_populate_selected_files_from_url_failure(self, tmp_path):
        """Test _populate_selected_files_from_url handles download errors"""
        preprocess = Preprocess.__new__(Preprocess)
        preprocess._dependencies_checked = True
        preprocess.explanations = mock.MagicMock()
        preprocess.explanations.root_dir = str(tmp_path)
        preprocess.explanations_folder = Path("explanations")

        with mock.patch(
            "pdstools.pega_io.File.read_ds_export",
            side_effect=Exception("Network error"),
        ):
            with pytest.raises(ValueError, match="Failed to download file from"):
                preprocess._populate_selected_files_from_url(
                    "https://example.com/file.parquet",
                )
