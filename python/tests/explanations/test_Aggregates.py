import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from pdstools.explanations.Aggregates import Aggregates

@pytest.fixture
def mock_aggregates():
    mock_explanations = Mock()
    mock_explanations.root_dir = ".tmp"
    mock_explanations.data_folder = "explanations_data"
    mock_explanations.aggregates_folder = "aggregated_data"
    mock_explanations.report_folder = "reports"
    mock_explanations.model_name = ""
    mock_explanations.from_date = datetime(2023, 1, 1)
    mock_explanations.to_date = datetime(2023, 1, 8)
    return Aggregates(explanations=mock_explanations)

class TestAggregatesFileOps:

    @patch('pdstools.explanations.Aggregates.shutil')
    def test_clean_aggregates_folder_exists(self, mock_shutil, mock_aggregates):
        """Test cleaning existing aggregates folder"""
        mock_folder = Mock()
        mock_folder.exists.return_value = True
        mock_folder.is_dir.return_value = True
        mock_aggregates.aggregates_folder = mock_folder

        mock_aggregates._clean_aggregates_folder()

        mock_shutil.rmtree.assert_called_once_with(mock_folder)
        mock_folder.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('pdstools.explanations.Aggregates.shutil')
    def test_clean_aggregates_folder_not_exists(self, mock_shutil, mock_aggregates):
        """Test cleaning non-existing aggregates folder"""
        mock_folder = Mock()
        mock_folder.exists.return_value = False
        mock_aggregates.aggregates_folder = mock_folder

        mock_aggregates._clean_aggregates_folder()

        mock_shutil.rmtree.assert_not_called()
        mock_folder.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('pdstools.explanations.Aggregates.glob')
    @patch('pdstools.explanations.Aggregates.pathlib.Path')
    @patch('pdstools.explanations.Aggregates.logger')
    def test_populate_selected_files_success(self, mock_logger, mock_path, mock_glob, mock_aggregates):
        """Test successful population of selected files"""
        mock_aggregates.from_date = datetime(2023, 1, 1)
        mock_aggregates.to_date = datetime(2023, 1, 3)
        mock_aggregates.model_name = "test_model"
        mock_aggregates.explanations_folder = "/test/data"

        mock_glob.side_effect = [
            ["/test/data/test_model_20230101.parquet"],
            ["/test/data/test_model_20230102.parquet"],
            ["/test/data/test_model_20230103.parquet"]
        ]

        mock_path.exists.return_value = True
        mock_aggregates._populate_selected_files()

        expected_files = [
            "/test/data/test_model_20230101.parquet",
            "/test/data/test_model_20230102.parquet",
            "/test/data/test_model_20230103.parquet"
        ]
        assert mock_aggregates.selected_files == expected_files
        mock_logger.info.assert_called_once()

    def test_populate_selected_files_no_dates(self, mock_aggregates):
        """Test populate_selected_files with missing dates"""
        mock_aggregates.from_date = None
        mock_aggregates.to_date = None

        with pytest.raises(ValueError, match="Either from_date or to_date must be passed"):
            mock_aggregates._populate_selected_files()

class TestAggregatesGenerate:
    """Test the generate method and related functionality"""

    @patch.object(Aggregates, '_clean_aggregates_folder')
    @patch.object(Aggregates, '_populate_selected_files')
    def test_generate_no_files(self, mock_populate, mock_clean, mock_aggregates):
        """Test generation when no files are found"""
        mock_aggregates.selected_files = []
        with patch("pdstools.explanations.Aggregates.logger") as mock_logger:
            mock_aggregates.generate()
            mock_clean.assert_called_once()
            mock_populate.assert_called_once()
            mock_logger.error.assert_called_once_with("No files found to aggregate!")


    @patch.object(Aggregates, '_run_agg')
    @patch.object(Aggregates, '_populate_selected_files')
    def test_generate_success(self, mock_populate, mock_run_agg, mock_aggregates):
        """Test successful generation of aggregates"""
        mock_aggregates.selected_files = ["file1.parquet", "file2.parquet"]
        mock_populate.side_effect = lambda: None
        mock_run_agg.side_effect = [None, None]  # Simulate successful runs
        mock_aggregates.generate()
        assert mock_run_agg.call_count == 2  # Called for NUMERIC and SYMBOLIC

    @patch('pdstools.explanations.Aggregates.logger')
    @patch.object(Aggregates, '_run_agg')
    @patch.object(Aggregates, '_populate_selected_files')
    def test_generate_numeric_failure(self, mock_populate, mock_run_agg, mock_logger, mock_aggregates):
        """Test generation when numeric aggregation fails"""
        mock_aggregates.selected_files = ["file1.parquet"]
        mock_populate.side_effect = lambda: None
        mock_run_agg.side_effect = [Exception("Numeric error"), None]
        mock_aggregates.generate()
        mock_logger.error.assert_called_with("Failed to aggregate numeric data, err=Numeric error")

class TestAggregatesQueryOperations:
    """Test query execution and database operations"""

    def test_execute_query_no_connection(self, mock_aggregates):
        """Test execute_query when connection is not initialized"""
        mock_aggregates._conn = None

        with pytest.raises(ValueError, match="DuckDB connection is not initialized"):
            mock_aggregates._execute_query("SELECT 1")

    @patch('pdstools.explanations.Aggregates.duckdb')
    def test_execute_query_success(self, mock_duckdb, mock_aggregates):
        """Test successful query execution"""
        mock_conn = Mock()
        mock_result = Mock()
        mock_conn.execute.return_value = mock_result
        mock_aggregates._conn = mock_conn

        result = mock_aggregates._execute_query("SELECT 1")

        mock_conn.execute.assert_called_once_with("SELECT 1")
        assert result == mock_result

    def test_clean_query(self):
        """Test query cleaning functionality"""
        dirty_query = "SELECT  *\nFROM   table\n  WHERE  condition"
        clean_query = Aggregates._clean_query(dirty_query)
        
        expected = "SELECT * FROM table WHERE condition"
        assert clean_query == expected

class TestAggregatesTableOperations:
    """Test table name and SQL operations"""

    def test_get_table_name_numeric(self, mock_aggregates):
        """Test getting table name for numeric predictor type"""
        from pdstools.explanations.ExplanationsUtils import _PREDICTOR_TYPE
        from pdstools.explanations.ExplanationsUtils import _TABLE_NAME
        predictor_type = _PREDICTOR_TYPE.NUMERIC
        expected_table_name = _TABLE_NAME.NUMERIC
        
        result = mock_aggregates._get_table_name(predictor_type)
        # Assuming the method returns a _TABLE_NAME enum value
        assert result is not None
        assert result == expected_table_name

    def test_get_table_name_symbolic(self, mock_aggregates):
        """Test getting table name for symbolic predictor type"""
        from pdstools.explanations.ExplanationsUtils import _PREDICTOR_TYPE
        from pdstools.explanations.ExplanationsUtils import _TABLE_NAME
        predictor_type = _PREDICTOR_TYPE.SYMBOLIC
        expected_table_name = _TABLE_NAME.SYMBOLIC
        
        result = mock_aggregates._get_table_name(predictor_type)
        # Assuming the method returns a _TABLE_NAME enum value
        assert result is not None
        assert result == expected_table_name

class TestAggregatesContextOperations:
    """Test context-related operations"""

    def test_get_contexts_cached(self, mock_aggregates):
        """Test _get_contexts when contexts are already cached"""
        mock_aggregates.contexts = ["context1", "context2"]
        result = mock_aggregates._get_contexts(Mock())
        assert result == ["context1", "context2"]

    @patch.object(Aggregates, '_execute_query')
    @patch.object(Aggregates, '_get_table_name')
    def test_get_contexts_from_db(self, mock_get_table_name, mock_execute_query, mock_aggregates):
        """Test _get_contexts when fetching from database"""
        from pdstools.explanations.ExplanationsUtils import _COL
        mock_aggregates.contexts = None

        mock_table_name = Mock()
        mock_table_name.value = "test_table"
        mock_get_table_name.return_value = mock_table_name

        mock_result = Mock()
        mock_pl_result = Mock()
        mock_pl_result.to_list.return_value = ["context1", "context2"]
        mock_result.pl.return_value = {_COL.PARTITON.value: mock_pl_result}
        mock_execute_query.return_value = mock_result

        result = mock_aggregates._get_contexts(Mock())
        assert result == ["context1", "context2"]
        assert mock_aggregates.contexts == ["context1", "context2"]


class TestAggregatesParametrized:
    """Parametrized tests for various scenarios"""

    @pytest.mark.parametrize("memory_limit,thread_count,batch_limit", [
        (2, 4, 10),
        (4, 8, 20),
        (1, 2, 5),
    ])
    def test_aggregates_init_various_configs(self, memory_limit, thread_count, batch_limit):
        """Test Aggregates initialization with various configurations"""
        mock_explanations = Mock()
        mock_explanations.memory_limit = memory_limit
        mock_explanations.thread_count = thread_count
        mock_explanations.batch_limit = batch_limit
        mock_explanations.root_dir = "/test"
        mock_explanations.aggregates_folder = "agg"
        mock_explanations.data_folder = "data"
        mock_explanations.from_date = datetime.now()
        mock_explanations.to_date = datetime.now()
        mock_explanations.model_name = "test"
        mock_explanations.progress_bar = False

        aggregates = Aggregates(explanations=mock_explanations)
        
        assert aggregates.memory_limit == memory_limit
        assert aggregates.thread_count == thread_count
        assert aggregates.batch_limit == batch_limit