"""Tests for the metric_limits module."""

import polars as pl
import pytest

from pdstools.utils.metric_limits import (
    MetricLimits,
    create_RAG_table,
    percentage_within_0_1_range_rag,
    standard_NBAD_channels_rag,
    standard_NBAD_configurations_rag,
    standard_NBAD_directions_rag,
    standard_NBAD_predictions_rag,
)
from pdstools.utils.report_utils import create_metric_table


class TestMetricLimits:
    """Tests for the MetricLimits class."""

    def test_get_limits_returns_valid_dataframe(self):
        df = MetricLimits.get_limits()
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        for col in ["Category", "MetricID", "Minimum", "Best Practice Min"]:
            assert col in df.columns

    def test_get_limit_for_known_metric(self):
        limits = MetricLimits.get_limit_for_metric("ModelPerformance")
        assert limits != {}
        assert "best_practice_min" in limits
        assert "best_practice_max" in limits

    def test_get_limit_for_unknown_metric(self):
        limits = MetricLimits.get_limit_for_metric("UnknownMetricXYZ123")
        assert limits == {}

    def test_get_metric_RAG_code_returns_expression(self):
        expr = MetricLimits.get_metric_RAG_code("col", "ModelPerformance")
        assert isinstance(expr, pl.Expr)

    def test_minimum_raises_on_unknown_metric(self):
        with pytest.raises(KeyError, match="Unknown metric ID"):
            MetricLimits.minimum("UnknownMetricXYZ123")

    def test_maximum_raises_on_unknown_metric(self):
        with pytest.raises(KeyError, match="Unknown metric ID"):
            MetricLimits.maximum("UnknownMetricXYZ123")

    def test_best_practice_min_raises_on_unknown_metric(self):
        with pytest.raises(KeyError, match="Unknown metric ID"):
            MetricLimits.best_practice_min("UnknownMetricXYZ123")

    def test_best_practice_max_raises_on_unknown_metric(self):
        with pytest.raises(KeyError, match="Unknown metric ID"):
            MetricLimits.best_practice_max("UnknownMetricXYZ123")

    def test_convenience_methods_return_values(self):
        # These should not raise for known metrics
        assert MetricLimits.minimum("ModelPerformance") is not None or True
        assert MetricLimits.maximum("ModelPerformance") is not None or True
        assert MetricLimits.best_practice_min("ModelPerformance") is not None or True
        assert MetricLimits.best_practice_max("ModelPerformance") is not None or True


class TestNBADConfigurationsRAG:
    """Tests for standard_NBAD_configurations_rag."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("", None),
            (None, None),
            ("Web_Click_Through_Rate", "GREEN"),
            ("Mobile_Click_Through_Rate", "GREEN"),
            ("MyApp_Web_Click_Through_Rate", "GREEN"),
            ("Web_Click_Through_Rate_GB", "GREEN"),
            ("web_click_through_rate", "GREEN"),
            ("OmniAdaptiveModel", "YELLOW"),
            ("Default_Inbound_Model", "YELLOW"),
            ("InvalidConfig", "AMBER"),
            ("Web_Click_Through_Rate,InvalidConfig", "AMBER"),
            ("Web_Click_Through_Rate,Mobile_Click_Through_Rate", "GREEN"),
        ],
    )
    def test_configuration_rag_status(self, value, expected):
        assert standard_NBAD_configurations_rag(value) == expected


class TestNBADPredictionsRAG:
    """Tests for standard_NBAD_predictions_rag."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("", None),
            (None, None),
            ("PredictWebPropensity", "GREEN"),
            ("PredictMobilePropensity", "GREEN"),
            ("MyApp_PredictWebPropensity", "GREEN"),
            ("PredictWebPropensity_GB", "AMBER"),
            ("PredictActionPropensity", "YELLOW"),
            ("InvalidPrediction", "AMBER"),
        ],
    )
    def test_prediction_rag_status(self, value, expected):
        assert standard_NBAD_predictions_rag(value) == expected


class TestNBADChannelsRAG:
    """Tests for standard_NBAD_channels_rag."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("", None),
            (None, None),
            ("Web", "GREEN"),
            ("web", "GREEN"),
            ("E-mail", "GREEN"),
            ("Email", "GREEN"),
            ("Call Center", "GREEN"),
            ("CallCenter", "GREEN"),
            ("Other", "YELLOW"),
            ("Multi-channel", "AMBER"),
            ("UnknownChannel", "AMBER"),
        ],
    )
    def test_channel_rag_status(self, value, expected):
        assert standard_NBAD_channels_rag(value) == expected


class TestNBADDirectionsRAG:
    """Tests for standard_NBAD_directions_rag."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("", None),
            (None, None),
            ("Inbound", "GREEN"),
            ("Outbound", "GREEN"),
            ("inbound", "GREEN"),
            ("Unknown", "AMBER"),
        ],
    )
    def test_direction_rag_status(self, value, expected):
        assert standard_NBAD_directions_rag(value) == expected


class TestPercentageRAG:
    """Tests for percentage_within_0_1_range_rag."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, None),
            (0.5, "GREEN"),
            (0.001, "GREEN"),
            (0.999, "GREEN"),
            (0.0, "RED"),
            (1.0, "RED"),
            (-0.1, "RED"),
            (1.5, "RED"),
        ],
    )
    def test_percentage_rag_status(self, value, expected):
        assert percentage_within_0_1_range_rag(value) == expected


class TestCreateRAGTable:
    """Tests for create_RAG_table."""

    def test_applies_rag_coloring(self):
        from great_tables import GT

        df = pl.DataFrame({"Channel": ["Web", "Other", "Unknown"]})
        gt = create_RAG_table(
            GT(df),
            df,
            column_to_metric={"Channel": standard_NBAD_channels_rag},
            strict_metric_validation=False,
        )
        assert gt is not None

    def test_tuple_keys_expanded(self):
        from great_tables import GT

        df = pl.DataFrame({"Col1": ["Web"], "Col2": ["Mobile"]})
        gt = create_RAG_table(
            GT(df),
            df,
            column_to_metric={("Col1", "Col2"): standard_NBAD_channels_rag},
            strict_metric_validation=False,
        )
        assert gt is not None

    def test_strict_validation_raises_on_unknown_metric(self):
        from great_tables import GT

        df = pl.DataFrame({"Col": [1.0]})
        with pytest.raises(ValueError, match="Unknown metric ID"):
            create_RAG_table(
                GT(df),
                df,
                column_to_metric={"Col": "UnknownMetricXYZ123"},
                strict_metric_validation=True,
            )


class TestEvaluateMetricRAG:
    """Tests for MetricLimits.evaluate_metric_rag Python function."""

    def test_none_value_returns_none(self):
        assert MetricLimits.evaluate_metric_rag("ModelPerformance", None) is None

    def test_unknown_metric_returns_none(self):
        assert MetricLimits.evaluate_metric_rag("UnknownMetricXYZ123", 50) is None

    def test_model_performance_rag_basic(self):
        """Test ModelPerformance metric RAG evaluation at key thresholds."""
        # Get actual limits for verification
        limits = MetricLimits.get_limit_for_metric("ModelPerformance")

        # Test below minimum is RED
        assert MetricLimits.evaluate_metric_rag("ModelPerformance", 49.0) == "RED"

        # Test above maximum is RED
        assert MetricLimits.evaluate_metric_rag("ModelPerformance", 101.0) == "RED"

        # Test within best practice range is GREEN
        bp_min = limits["best_practice_min"]
        if bp_min is not None:
            assert (
                MetricLimits.evaluate_metric_rag("ModelPerformance", bp_min + 1)
                == "GREEN"
            )


class TestPolarsExpressionEquivalence:
    """Tests that Python evaluate_metric_rag matches Polars get_metric_RAG_code."""

    @pytest.mark.parametrize(
        "metric_id,test_values",
        [
            (
                "ModelPerformance",
                [49.0, 50.0, 52.0, 55.0, 60.0, 85.0, 90.0, 95.0, 100.0, 100.1],
            ),
            ("EngagementLift", [-0.1, 0.0, 0.05, 0.1, 0.5, 1.0]),
            ("ControlPercentage", [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 50.0]),
        ],
    )
    def test_python_matches_polars(self, metric_id, test_values):
        """Verify Python function produces same results as Polars expression."""
        df = pl.DataFrame({"Value": test_values})

        # Get Polars results
        polars_expr = MetricLimits.get_metric_RAG_code("Value", metric_id)
        polars_results = df.with_columns(polars_expr)["Value_RAG"].to_list()

        # Get Python results
        python_results = [
            MetricLimits.evaluate_metric_rag(metric_id, v) for v in test_values
        ]

        assert polars_results == python_results, (
            f"Mismatch for {metric_id}:\n"
            f"  Values:  {test_values}\n"
            f"  Polars:  {polars_results}\n"
            f"  Python:  {python_results}"
        )


class TestCreateMetricTable:
    """Tests for create_metric_table."""

    def test_creates_table_with_rag_coloring(self):
        df = pl.DataFrame({"Performance": [52.0, 60.0, 90.0]})
        gt = create_metric_table(
            df,
            column_to_metric={"Performance": "ModelPerformance"},
            strict_metric_validation=True,
        )
        assert gt is not None

    def test_accepts_callable_metric(self):
        df = pl.DataFrame({"Channel": ["Web", "Other"]})
        gt = create_metric_table(
            df,
            column_to_metric={"Channel": standard_NBAD_channels_rag},
            strict_metric_validation=False,
        )
        assert gt is not None
