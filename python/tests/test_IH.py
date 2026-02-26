"""Testing the functionality of the IH class"""

import math
from datetime import timedelta

import polars as pl
import pytest
from pdstools import IH
from plotly.graph_objs import Figure


@pytest.fixture
def ih():
    return IH.from_mock_data()


def test_mockdata(ih):
    assert ih.data.collect().height > 100000  # interactions
    assert ih.data.collect().width == 13  # nr of IH properties in the sample data

    summary = ih.aggregates.summarize_by_interaction().collect()
    assert summary.height == 100000


def test_summarize_by_interaction_basic(ih):
    """Test basic functionality of summarize_by_interaction"""
    # Basic call without parameters
    result = ih.aggregates.summarize_by_interaction().collect()
    assert result.height == 100000
    assert "InteractionID" in result.columns
    assert "Interaction_Outcome_Engagement" in result.columns
    assert "Interaction_Outcome_Conversion" in result.columns
    assert "Interaction_Outcome_OpenRate" in result.columns


def test_summarize_by_interaction_with_by(ih):
    """Test summarize_by_interaction with 'by' parameter"""
    # Test with string parameter
    result_channel = ih.aggregates.summarize_by_interaction(by="Channel").collect()
    assert "Channel" in result_channel.columns
    assert result_channel.height == 100000

    # Test with list parameter
    result_multi = ih.aggregates.summarize_by_interaction(
        by=["Channel", "Direction"],
    ).collect()
    assert "Channel" in result_multi.columns
    assert "Direction" in result_multi.columns

    # Test with Polars expression
    result_expr = ih.aggregates.summarize_by_interaction(
        by=pl.col("Channel").alias("ChannelRenamed"),
    ).collect()
    assert "ChannelRenamed" in result_expr.columns


def test_summarize_by_interaction_with_every(ih):
    """Test summarize_by_interaction with 'every' parameter"""
    # Test with string parameter
    result_daily = ih.aggregates.summarize_by_interaction(every="1d").collect()
    assert "OutcomeTime" in result_daily.columns

    # Test with timedelta
    result_td = ih.aggregates.summarize_by_interaction(
        every=timedelta(days=1),
    ).collect()
    assert "OutcomeTime" in result_td.columns


def test_summarize_by_interaction_with_query(ih):
    """Test summarize_by_interaction with 'query' parameter"""
    # Test with simple query
    result_web = ih.aggregates.summarize_by_interaction(
        query=pl.col("Channel") == "Web",
    ).collect()
    assert result_web.height < 100000  # Should be filtered

    # Test with complex query
    result_complex = ih.aggregates.summarize_by_interaction(
        query=(pl.col("Channel") == "Web") & (pl.col("Direction") == "Inbound"),
    ).collect()
    # In the mock data, all Web channel interactions have Direction as "Inbound"
    # So we just verify that the complex query still returns results
    assert result_complex.height > 0
    assert result_complex.height <= result_web.height


def test_summarize_by_interaction_complex(ih):
    """Test summarize_by_interaction with multiple parameters"""
    result = ih.aggregates.summarize_by_interaction(
        by=["Channel", "Direction"],
        every="1w",
        query=pl.col("Channel").is_not_null(),
        debug=True,
    ).collect()

    assert "Channel" in result.columns
    assert "Direction" in result.columns
    assert "OutcomeTime" in result.columns
    assert "Outcomes" in result.columns
    assert result.height > 0


def test_summary_success_rates_basic(ih):
    """Test basic functionality of summary_success_rates"""
    result = ih.aggregates.summary_success_rates().collect()

    # Check that the result contains the expected columns
    for metric in ih.positive_outcome_labels.keys():
        assert f"Positives_{metric}" in result.columns
        assert f"Negatives_{metric}" in result.columns
        assert f"SuccessRate_{metric}" in result.columns
        assert f"StdErr_{metric}" in result.columns

    assert "Interactions" in result.columns
    assert result.height > 0


def test_summary_success_rates_with_by(ih):
    """Test summary_success_rates with 'by' parameter"""
    # Test with string parameter
    result_channel = ih.aggregates.summary_success_rates(by="Channel").collect()
    assert "Channel" in result_channel.columns
    assert result_channel.height > 0

    # Test with list parameter
    result_multi = ih.aggregates.summary_success_rates(
        by=["Channel", "Direction"],
    ).collect()
    assert "Channel" in result_multi.columns
    assert "Direction" in result_multi.columns
    assert result_multi.height > 0

    # Test with Polars expression
    result_expr = ih.aggregates.summary_success_rates(
        by=pl.col("Channel").alias("ChannelRenamed"),
    ).collect()
    assert "ChannelRenamed" in result_expr.columns
    assert result_expr.height > 0


def test_summary_success_rates_with_every(ih):
    """Test summary_success_rates with 'every' parameter"""
    # Test with string parameter
    result_daily = ih.aggregates.summary_success_rates(every="1d").collect()
    assert "OutcomeTime" in result_daily.columns
    assert result_daily.height > 0

    # Test with timedelta
    result_td = ih.aggregates.summary_success_rates(every=timedelta(days=1)).collect()
    assert "OutcomeTime" in result_td.columns
    assert result_td.height > 0


def test_summary_success_rates_with_query(ih):
    """Test summary_success_rates with 'query' parameter"""
    # Test with simple query
    result_web = ih.aggregates.summary_success_rates(
        query=pl.col("Channel") == "Web",
    ).collect()
    assert result_web.height > 0

    # Test with complex query
    result_complex = ih.aggregates.summary_success_rates(
        query=(pl.col("Channel") == "Web") & (pl.col("Direction") == "Inbound"),
    ).collect()
    assert result_complex.height > 0
    assert result_complex.height <= result_web.height

    # Verify that the query affects the data
    all_results = ih.aggregates.summary_success_rates().collect()
    # In the mock data, the summary is aggregated to a single row regardless of query
    # So we check that the values are different instead of the row count
    assert (
        result_web["Positives_Engagement"].item() != all_results["Positives_Engagement"].item()
        or result_web["Negatives_Engagement"].item() != all_results["Negatives_Engagement"].item()
    )


def test_summary_success_rates_complex(ih):
    """Test summary_success_rates with multiple parameters"""
    result = ih.aggregates.summary_success_rates(
        by=[
            pl.col("Propensity").qcut(10).alias("PropensityBin"),
            "Channel",
            "Direction",
        ],
        every="1w",
        query=pl.col("Channel").is_not_null(),
        debug=True,
    ).collect()

    assert "PropensityBin" in result.columns
    assert "Channel" in result.columns
    assert "Direction" in result.columns
    assert "OutcomeTime" in result.columns
    assert "Outcomes" in result.columns
    assert result.height > 0

    # Verify calculations
    for metric in ih.positive_outcome_labels.keys():
        # Check that success rates are between 0 and 1
        success_rates = result[f"SuccessRate_{metric}"].to_list()
        for rate in success_rates:
            # Skip null and NaN values (NaN can occur when both positives and negatives are 0)
            if rate is not None and not math.isnan(rate):
                assert 0 <= rate <= 1

        # Check that standard errors are positive
        std_errs = result[f"StdErr_{metric}"].to_list()
        for err in std_errs:
            # Skip null and NaN values
            if err is not None and not math.isnan(err):
                assert err >= 0


def test_summary_outcomes_basic(ih):
    """Test basic functionality of summary_outcomes"""
    result = ih.aggregates.summary_outcomes().collect()

    # Check that the result contains the expected columns
    assert "Outcome" in result.columns
    assert "Count" in result.columns
    assert result.height > 0


def test_summary_outcomes_with_by(ih):
    """Test summary_outcomes with 'by' parameter"""
    # Test with string parameter
    result_channel = ih.aggregates.summary_outcomes(by="Channel").collect()
    assert "Channel" in result_channel.columns
    assert "Outcome" in result_channel.columns
    assert result_channel.height > 0

    # Test with list parameter
    result_multi = ih.aggregates.summary_outcomes(by=["Channel", "Direction"]).collect()
    assert "Channel" in result_multi.columns
    assert "Direction" in result_multi.columns
    assert "Outcome" in result_multi.columns
    assert result_multi.height > 0


def test_summary_outcomes_with_every(ih):
    """Test summary_outcomes with 'every' parameter"""
    # Test with string parameter
    result_daily = ih.aggregates.summary_outcomes(every="1d").collect()
    assert "OutcomeTime" in result_daily.columns
    assert "Outcome" in result_daily.columns
    assert result_daily.height > 0

    # Test with timedelta
    result_td = ih.aggregates.summary_outcomes(every=timedelta(days=1)).collect()
    assert "OutcomeTime" in result_td.columns
    assert "Outcome" in result_td.columns
    assert result_td.height > 0


def test_summary_outcomes_with_query(ih):
    """Test summary_outcomes with 'query' parameter"""
    # Test with simple query
    result_web = ih.aggregates.summary_outcomes(
        query=pl.col("Channel") == "Web",
    ).collect()
    assert result_web.height > 0

    # Test with complex query
    result_complex = ih.aggregates.summary_outcomes(
        query=(pl.col("Channel") == "Web") & (pl.col("Direction") == "Inbound"),
    ).collect()
    assert result_complex.height > 0

    # Verify that the complex query returns fewer or equal rows
    assert result_complex.height <= result_web.height

    # Verify that the query actually filtered the data
    all_results = ih.aggregates.summary_outcomes().collect()
    assert result_web.height < all_results.height


def test_summary_outcomes_complex(ih):
    """Test summary_outcomes with multiple parameters"""
    result = ih.aggregates.summary_outcomes(
        by=["Channel", "Direction"],
        every="1w",
        query=pl.col("Channel").is_not_null(),
    ).collect()

    assert "Channel" in result.columns
    assert "Direction" in result.columns
    assert "OutcomeTime" in result.columns
    assert "Outcome" in result.columns
    assert "Count" in result.columns
    assert result.height > 0

    # Verify sorting
    # The result should be sorted by Count (descending) and then by the group_by columns
    counts = result["Count"].to_list()
    for i in range(1, len(counts)):
        if counts[i - 1] < counts[i]:
            # If the previous count is less than the current count,
            # then the group_by columns must have changed
            assert (
                (result[i - 1, "Channel"] != result[i, "Channel"])
                or (result[i - 1, "Direction"] != result[i, "Direction"])
                or (result[i - 1, "OutcomeTime"] != result[i, "OutcomeTime"])
                or (result[i - 1, "Outcome"] != result[i, "Outcome"])
            )


def test_plots(ih):
    assert isinstance(ih.plot.overall_gauges(condition="ExperimentGroup"), Figure)
    assert isinstance(ih.plot.response_count_tree_map(), Figure)
    assert isinstance(ih.plot.success_rate_tree_map(), Figure)
    assert isinstance(ih.plot.action_distribution(), Figure)
    assert isinstance(ih.plot.success_rate(), Figure)
    assert isinstance(ih.plot.response_count(), Figure)
    assert isinstance(ih.plot.model_performance_trend(), Figure)
    assert isinstance(ih.plot.model_performance_trend(by="ModelTechnique"), Figure)
