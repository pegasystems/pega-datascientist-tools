"""Tests for DecisionAnalyzer.filtered and the Streamlit filter glue."""

import polars as pl
import pytest


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

    analyzer = DecisionAnalyzer.__new__(DecisionAnalyzer)
    type(analyzer).sample = property(lambda self: sample_data_v2.lazy())
    return analyzer


def test_filtered_without_filters(mock_decision_analyzer, sample_data_v2):
    """``filtered(None)`` and ``filtered([])`` should return sample unchanged."""
    expected = sample_data_v2.lazy().collect()
    assert mock_decision_analyzer.filtered(None).collect().equals(expected)
    assert mock_decision_analyzer.filtered([]).collect().equals(expected)


def test_filtered_with_single_expression(mock_decision_analyzer):
    """A single Polars expression should be applied."""
    expr = (pl.col("Channel") == "Web") & (pl.col("Direction") == "Inbound")
    result = mock_decision_analyzer.filtered(expr).collect()

    assert len(result) == 2
    assert (result["Channel"] == "Web").all()
    assert (result["Direction"] == "Inbound").all()


def test_filtered_with_list_of_expressions(mock_decision_analyzer):
    """Multiple expressions should AND together."""
    filters = [pl.col("Channel") == "Web", pl.col("Priority") >= 95.0]
    result = mock_decision_analyzer.filtered(filters).collect()

    assert len(result) == 1
    assert result["Action"][0] == "A1"


def test_filtered_does_not_import_streamlit(mock_decision_analyzer, monkeypatch):
    """``filtered`` must not look at Streamlit state even when it's set."""
    from pdstools.app.decision_analyzer import da_streamlit_utils

    monkeypatch.setattr(
        da_streamlit_utils.st,
        "session_state",
        {"page_channel_expr": pl.col("Channel") == "Web"},
        raising=False,
    )

    # No filter passed → full sample regardless of session_state contents.
    assert len(mock_decision_analyzer.filtered().collect()) == 6


def test_collect_page_filters_returns_expressions(monkeypatch):
    """The Streamlit helper should pull only set keys from session_state."""
    from pdstools.app.decision_analyzer import da_streamlit_utils
    from pdstools.app.decision_analyzer.da_streamlit_utils import collect_page_filters

    expr = pl.col("Channel") == "Web"
    monkeypatch.setattr(da_streamlit_utils.st, "session_state", {"page_channel_expr": expr}, raising=False)

    result = collect_page_filters()
    assert len(result) == 1
    # Expressions don't compare with ==, but should be the same object we stored.
    assert result[0] is expr


def test_collect_page_filters_empty_when_none_set(monkeypatch):
    """No keys set → empty list."""
    from pdstools.app.decision_analyzer import da_streamlit_utils
    from pdstools.app.decision_analyzer.da_streamlit_utils import collect_page_filters

    monkeypatch.setattr(da_streamlit_utils.st, "session_state", {}, raising=False)
    assert collect_page_filters() == []


@pytest.fixture
def sample_data_v1():
    """Create sample v1 data with Channel only (no Direction)."""
    return pl.DataFrame(
        {
            "Interaction ID": ["I1", "I1", "I2", "I2"],
            "Channel": ["Web", "Web", "Email", "Email"],
            "Action": ["A1", "A2", "A3", "A4"],
            "Decision Time": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "Priority": [100.0, 90.0, 85.0, 80.0],
        }
    ).with_columns(pl.col("Decision Time").str.strptime(pl.Datetime))


@pytest.fixture
def sample_data_no_channel():
    """Create sample data without Channel or Direction columns."""
    return pl.DataFrame(
        {
            "Interaction ID": ["I1", "I2"],
            "Action": ["A1", "A2"],
            "Decision Time": ["2024-01-01", "2024-01-02"],
        }
    ).with_columns(pl.col("Decision Time").str.strptime(pl.Datetime))


def test_get_available_channel_directions_v2(sample_data_v2):
    """Should return sorted Channel/Direction combinations for v2 data."""
    from pdstools.app.decision_analyzer.da_streamlit_utils import (
        get_available_channel_directions,
    )

    result = get_available_channel_directions(sample_data_v2.lazy())

    expected = [
        "Email/Outbound",
        "Mobile/Inbound",
        "Web/Inbound",
    ]
    assert result == expected


def test_get_available_channel_directions_v1(sample_data_v1):
    """Should return sorted Channel values for v1 data without Direction."""
    from pdstools.app.decision_analyzer.da_streamlit_utils import (
        get_available_channel_directions,
    )

    result = get_available_channel_directions(sample_data_v1.lazy())

    expected = ["Email", "Web"]
    assert result == expected


def test_get_available_channel_directions_no_channel(sample_data_no_channel):
    """Should return empty list when no Channel column exists."""
    from pdstools.app.decision_analyzer.da_streamlit_utils import (
        get_available_channel_directions,
    )

    result = get_available_channel_directions(sample_data_no_channel.lazy())

    assert result == []
