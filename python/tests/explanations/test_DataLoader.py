from pathlib import Path
import pytest
from unittest.mock import Mock, patch
import polars as pl
from pdstools.explanations import Explanations
from pdstools.explanations.DataLoader import DataLoader
from pdstools.explanations.ExplanationsUtils import _CONTRIBUTION_TYPE

basePath = Path(__file__).parent.parent / "resources" / "explanations"

@pytest.fixture
def data_loader():
    explanations = Explanations(aggregates_folder=basePath)
    return DataLoader(explanations)

@pytest.fixture
def mock_explanations():
    """Create a mock explanations object with aggregates folder."""
    explanations = Mock()
    explanations.aggregates.aggregates_folder = "/test/aggregates/folder"
    return explanations


@pytest.fixture
def mock_data_loader(mock_explanations):
    """Create a DataLoader instance with mocked explanations."""
    return DataLoader(mock_explanations)


@pytest.fixture
def mock_lazy_frame():
    """Create a mock LazyFrame with select and sort methods."""
    mock_frame = Mock(spec=pl.LazyFrame)
    mock_frame.select.return_value = mock_frame
    mock_frame.sort.return_value = mock_frame
    return mock_frame


class TestDataLoaderLoadData:
    """Test cases for DataLoader.load_data method."""

    def test_load_data_success(self, data_loader):
        """Test successful data loading."""
        
        data_loader.load_data()
        
        assert data_loader.initialized is True
        assert data_loader.df_contextual is not None
        assert data_loader.df_overall is not None

    @patch('polars.scan_parquet')
    def test_load_data_file_not_found_error(self, mock_scan_parquet, data_loader):
        """Test handling of file not found errors."""
        data_loader.explanations.aggregates.aggregates_folder = "/non/existent/path"
        mock_scan_parquet.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            data_loader.load_data()
        
        assert data_loader.initialized is False

    def test_initial_state(self, mock_data_loader):
        """Test the initial state of DataLoader before load_data is called."""
        assert mock_data_loader.initialized is False
        assert mock_data_loader.df_contextual is None
        assert mock_data_loader.df_overall is None


class TestGetTopNPredictorContributionOverall:
    """Test cases for DataLoader.get_top_n_predictor_contribution_overall method."""

    @patch.object(DataLoader, '_get_predictor_contributions')
    @patch.object(DataLoader, 'load_data')
    def test_get_top_n_predictor_contribution_overall_default_params(
        self, mock_load_data, mock_get_predictor_contributions, mock_data_loader
    ):
        """Test get_top_n_predictor_contribution_overall with default parameters."""
        # Setup
        mock_df = Mock(spec=pl.DataFrame)
        mock_get_predictor_contributions.return_value = mock_df
        mock_data_loader.initialized = True
        
        # Execute
        result = mock_data_loader.get_top_n_predictor_contribution_overall()
        
        # Verify
        assert result is mock_df
        mock_load_data.assert_not_called()  # Should not call load_data when already initialized
        mock_get_predictor_contributions.assert_called_once_with(
            limit=10,
            descending=True,
            missing=True,
            remaining=True,
            contribution_type=_CONTRIBUTION_TYPE.CONTRIBUTION.value,
        )

    @patch.object(DataLoader, '_get_predictor_contributions')
    @patch.object(DataLoader, 'load_data')
    def test_get_top_n_predictor_contribution_overall_custom_params(
        self, mock_load_data, mock_get_predictor_contributions, mock_data_loader
    ):
        """Test get_top_n_predictor_contribution_overall with custom parameters."""
        # Setup
        mock_df = Mock(spec=pl.DataFrame)
        mock_get_predictor_contributions.return_value = mock_df
        mock_data_loader.initialized = True
        
        # Execute
        result = mock_data_loader.get_top_n_predictor_contribution_overall(
            top_n=5,
            descending=False,
            missing=False,
            remaining=False,
            contribution_calculation=_CONTRIBUTION_TYPE.CONTRIBUTION_ABS.value,
        )
        
        # Verify
        assert result is mock_df
        mock_load_data.assert_not_called()
        mock_get_predictor_contributions.assert_called_once_with(
            limit=5,
            descending=False,
            missing=False,
            remaining=False,
            contribution_type=_CONTRIBUTION_TYPE.CONTRIBUTION_ABS.value,
        )

    @patch.object(DataLoader, '_get_predictor_contributions')
    @patch.object(DataLoader, 'load_data')
    def test_get_top_n_predictor_contribution_overall_loads_data_when_not_initialized(
        self, mock_load_data, mock_get_predictor_contributions, mock_data_loader
    ):
        """Test that load_data is called when not initialized."""
        # Setup
        mock_df = Mock(spec=pl.DataFrame)
        mock_get_predictor_contributions.return_value = mock_df
        mock_data_loader.initialized = False
        
        # Execute
        result = mock_data_loader.get_top_n_predictor_contribution_overall()
        
        # Verify
        assert result is mock_df
        mock_load_data.assert_called_once()
        mock_get_predictor_contributions.assert_called_once()

    @patch.object(DataLoader, '_get_predictor_contributions')
    def test_get_top_n_predictor_contribution_overall_contribution_type_validation(
        self, mock_get_predictor_contributions, mock_data_loader
    ):
        """Test contribution type validation."""
        # Setup
        mock_df = Mock(spec=pl.DataFrame)
        mock_get_predictor_contributions.return_value = mock_df
        mock_data_loader.initialized = True
        
        # Test with valid contribution types
        valid_types = [
            _CONTRIBUTION_TYPE.CONTRIBUTION.value,
            _CONTRIBUTION_TYPE.CONTRIBUTION_ABS.value,
            _CONTRIBUTION_TYPE.CONTRIBUTION_WEIGHTED.value,
            _CONTRIBUTION_TYPE.CONTRIBUTION_WEIGHTED_ABS.value,
        ]
        
        for contrib_type in valid_types:
            # Execute
            result = mock_data_loader.get_top_n_predictor_contribution_overall(
                contribution_calculation=contrib_type
            )
            
            # Verify
            assert result is mock_df
            # Verify the contribution type was passed correctly
            call_args = mock_get_predictor_contributions.call_args
            assert call_args[1]['contribution_type'] == contrib_type

    @patch.object(DataLoader, '_get_predictor_contributions')
    def test_get_top_n_predictor_contribution_overall_invalid_contribution_type(
        self, mock_get_predictor_contributions, mock_data_loader
    ):
        """Test handling of invalid contribution type."""
        # Setup
        mock_data_loader.initialized = True
        
        # Execute & Verify - should raise an exception for invalid contribution type
        with pytest.raises(Exception):  # The actual exception type depends on _CONTRIBUTION_TYPE.validate_and_get_type implementation
            mock_data_loader.get_top_n_predictor_contribution_overall(
                contribution_calculation="invalid_contribution_type"
            )

    @patch.object(DataLoader, '_get_predictor_contributions')
    def test_get_top_n_predictor_contribution_overall_returns_dataframe(
        self, mock_get_predictor_contributions, mock_data_loader
    ):
        """Test that the method returns a polars DataFrame."""
        # Setup
        mock_df = Mock(spec=pl.DataFrame)
        mock_get_predictor_contributions.return_value = mock_df
        mock_data_loader.initialized = True
        
        # Execute
        result = mock_data_loader.get_top_n_predictor_contribution_overall()
        
        # Verify
        assert isinstance(result, type(mock_df))
        assert result is mock_df

    @patch.object(DataLoader, '_get_predictor_contributions')
    def test_get_top_n_predictor_contribution_overall_edge_cases(
        self, mock_get_predictor_contributions, mock_data_loader
    ):
        """Test edge cases for parameters."""
        # Setup
        mock_df = Mock(spec=pl.DataFrame)
        mock_get_predictor_contributions.return_value = mock_df
        mock_data_loader.initialized = True
        
        # Test with top_n = 0
        result = mock_data_loader.get_top_n_predictor_contribution_overall(top_n=0)
        assert result is mock_df
        call_args = mock_get_predictor_contributions.call_args
        assert call_args[1]['limit'] == 0
        
        # Test with very large top_n
        result = mock_data_loader.get_top_n_predictor_contribution_overall(top_n=1000)
        assert result is mock_df
        call_args = mock_get_predictor_contributions.call_args
        assert call_args[1]['limit'] == 1000

    @patch.object(DataLoader, '_get_predictor_contributions')
    def test_get_top_n_predictor_contribution_overall_all_boolean_combinations(
        self, mock_get_predictor_contributions, mock_data_loader
    ):
        """Test all combinations of boolean parameters."""
        # Setup
        mock_df = Mock(spec=pl.DataFrame)
        mock_get_predictor_contributions.return_value = mock_df
        mock_data_loader.initialized = True
        
        # Test all combinations of boolean parameters
        boolean_combinations = [
            (True, True, True),
            (True, True, False),
            (True, False, True),
            (True, False, False),
            (False, True, True),
            (False, True, False),
            (False, False, True),
            (False, False, False),
        ]
        
        for descending, missing, remaining in boolean_combinations:
            result = mock_data_loader.get_top_n_predictor_contribution_overall(
                descending=descending,
                missing=missing,
                remaining=remaining
            )
            assert result is mock_df
            call_args = mock_get_predictor_contributions.call_args
            assert call_args[1]['descending'] == descending
            assert call_args[1]['missing'] == missing
            assert call_args[1]['remaining'] == remaining
            assert call_args[1]['contribution_type'] == _CONTRIBUTION_TYPE.CONTRIBUTION.value   
            