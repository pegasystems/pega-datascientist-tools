"""Tests for DecisionAnalyzer.filtered_sample property."""

import polars as pl
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_data_v2():
    """Create sample v2 data with Channel and Direction."""
    return pl.DataFrame(
        {
            "Interaction ID": ["I1", "I1", "I2", "I2", "I3", "I3"],
            "Channel": ["Web", "Web", "Email", "Email", "Mobile", "Mobile"],
            "Direction": ["Inbound", "Inbound", "Outbound", "Outbound", "Inbound", "Inbound"],
            "Action": ["A1", "A2", "A3", "A4", "A5", "A6"],
            "Decision Time": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "Priority": [100.0, 90.0, 85.0, 80.0, 75.0, 70.0],
        }
    ).with_columns(pl.col("Decision Time").str.strptime(pl.Datetime))


@pytest.fixture
def mock_decision_analyzer(sample_data_v2):
    """Create a mock DecisionAnalyzer with sample property."""
    from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer

    # Create analyzer with minimal init
    analyzer = DecisionAnalyzer.__new__(DecisionAnalyzer)

    # Mock the sample property to return our test data
    type(analyzer).sample = property(lambda self: sample_data_v2.lazy())

    return analyzer


def test_filtered_sample_without_filters(mock_decision_analyzer, sample_data_v2):
    """When no page filters set, filtered_sample should return sample unchanged."""
    with patch("streamlit.session_state", new_callable=MagicMock) as mock_st:
        mock_st.get.return_value = None  # No page_channel_expr

        result = mock_decision_analyzer.filtered_sample
        expected = sample_data_v2.lazy()

        # Compare collected DataFrames
        assert result.collect().equals(expected.collect())


def test_filtered_sample_with_channel_filter(mock_decision_analyzer, sample_data_v2):
    """When page_channel_expr is set, filtered_sample should apply the filter."""
    with patch("streamlit.session_state", new_callable=MagicMock) as mock_st:
        # Set up a channel filter for Web/Inbound
        channel_expr = (pl.col("Channel") == "Web") & (pl.col("Direction") == "Inbound")
        mock_st.get.return_value = channel_expr

        result = mock_decision_analyzer.filtered_sample

        # Should only have Web/Inbound rows (first 2 rows)
        result_collected = result.collect()
        assert len(result_collected) == 2
        assert (result_collected["Channel"] == "Web").all()
        assert (result_collected["Direction"] == "Inbound").all()


def test_filtered_sample_outside_streamlit(mock_decision_analyzer, sample_data_v2):
    """When not in Streamlit context, filtered_sample should return sample."""
    # Don't mock streamlit - let ImportError occur naturally
    result = mock_decision_analyzer.filtered_sample
    expected = sample_data_v2.lazy()

    assert result.collect().equals(expected.collect())
