"""Tests for the NumberFormat class and MetricFormats registry."""

import polars as pl

from pdstools.utils.number_format import NumberFormat
from pdstools.utils.metric_limits import MetricFormats


class TestNumberFormat:
    """Tests for the NumberFormat class."""

    def test_basic_decimal_formatting(self):
        """Test basic decimal formatting."""
        fmt = NumberFormat(decimals=2)
        assert fmt.format_value(1234.567) == "1,234.57"
        assert fmt.format_value(89.1) == "89.10"
        assert fmt.format_value(0) == "0.00"

    def test_percentage_formatting(self):
        """Test percentage formatting with scale_by and suffix."""
        fmt = NumberFormat(decimals=1, scale_by=100, suffix="%")
        assert fmt.format_value(0.1234) == "12.3%"
        assert fmt.format_value(0.567) == "56.7%"
        assert fmt.format_value(1.0) == "100.0%"

    def test_compact_formatting(self):
        """Test compact notation (K, M, B, T)."""
        fmt = NumberFormat(compact=True)
        assert fmt.format_value(1234) == "1K"
        assert fmt.format_value(1234567) == "1M"
        assert fmt.format_value(1234567890) == "1B"
        assert fmt.format_value(1234567890000) == "1T"
        assert fmt.format_value(500) == "500"
        assert fmt.format_value(0) == "0"

    def test_compact_with_decimals(self):
        """Test compact notation with decimals."""
        fmt = NumberFormat(decimals=1, compact=True)
        assert fmt.format_value(1500) == "1.5K"
        assert fmt.format_value(2500000) == "2.5M"

    def test_german_locale(self):
        """Test German locale formatting (dot for thousands, comma for decimal)."""
        fmt = NumberFormat(decimals=2, locale="de_DE")
        assert fmt.format_value(1234.56) == "1.234,56"

    def test_none_and_nan_handling(self):
        """Test handling of None and NaN values."""
        fmt = NumberFormat(decimals=2)
        assert fmt.format_value(None) == ""
        assert fmt.format_value(float("nan")) == ""

    def test_to_polars_expr(self):
        """Test Polars expression generation."""
        fmt = NumberFormat(decimals=2)
        df = pl.DataFrame({"value": [1234.567, 89.1]})
        result = df.with_columns(fmt.to_polars_expr("value").alias("formatted"))

        assert result["formatted"].to_list() == ["1,234.57", "89.10"]

    def test_format_polars_column(self):
        """Test the convenience method for formatting Polars columns."""
        fmt = NumberFormat(decimals=1, scale_by=100, suffix="%")
        df = pl.DataFrame({"rate": [0.1234, 0.567]})
        result = fmt.format_polars_column(df, "rate")

        assert "rate_formatted" in result.columns
        assert result["rate_formatted"].to_list() == ["12.3%", "56.7%"]

    def test_format_polars_column_custom_output(self):
        """Test custom output column name."""
        fmt = NumberFormat(decimals=0)
        df = pl.DataFrame({"count": [1000, 2500]})
        result = fmt.format_polars_column(df, "count", output_column="count_str")

        assert "count_str" in result.columns


class TestMetricFormats:
    """Tests for the MetricFormats registry."""

    def test_get_existing_metric(self):
        """Test retrieving an existing metric format."""
        fmt = MetricFormats.get("ModelPerformance")
        assert fmt is not None
        assert fmt.decimals == 2

    def test_get_nonexistent_metric(self):
        """Test retrieving a non-existent metric returns None."""
        assert MetricFormats.get("NonExistentMetric") is None

    def test_get_or_default(self):
        """Test get_or_default returns default for unknown metrics."""
        fmt = MetricFormats.get_or_default("UnknownMetric")
        assert fmt == MetricFormats.DEFAULT_FORMAT
        assert fmt.compact is True

    def test_has_format(self):
        """Test checking if a metric has a format."""
        assert MetricFormats.has_format("CTR") is True
        assert MetricFormats.has_format("NonExistent") is False

    def test_list_metrics(self):
        """Test listing all defined metrics."""
        metrics = MetricFormats.list_metrics()
        assert "ModelPerformance" in metrics
        assert "CTR" in metrics
        assert len(metrics) >= 6  # At least the 6 predefined metrics

    def test_register_custom_metric(self):
        """Test registering a custom metric format."""
        custom_fmt = NumberFormat(decimals=4)
        MetricFormats.register("CustomTestMetric", custom_fmt)

        retrieved = MetricFormats.get("CustomTestMetric")
        assert retrieved is not None
        assert retrieved.decimals == 4

        # Clean up
        del MetricFormats._FORMATS["CustomTestMetric"]

    def test_predefined_metrics_format_correctly(self):
        """Test that predefined metrics format values correctly."""
        # ModelPerformance: 2 decimals
        perf_fmt = MetricFormats.get("ModelPerformance")
        assert perf_fmt.format_value(0.875) == "0.88"

        # CTR: 3 decimals, percentage
        ctr_fmt = MetricFormats.get("CTR")
        assert ctr_fmt.format_value(0.00123) == "0.123%"

        # EngagementLift: 0 decimals, percentage
        lift_fmt = MetricFormats.get("EngagementLift")
        assert lift_fmt.format_value(0.15) == "15%"
