"""Tests for NumberFormat and MetricFormats."""

import polars as pl
import pytest

from pdstools.utils.metric_limits import MetricFormats
from pdstools.utils.number_format import NumberFormat


class TestNumberFormat:
    """Tests for the NumberFormat class."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (1234.567, "1,234.57"),
            (89.1, "89.10"),
            (0, "0.00"),
        ],
    )
    def test_decimal_formatting(self, value, expected):
        assert NumberFormat(decimals=2).format_value(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.1234, "12.3%"),
            (0.567, "56.7%"),
            (1.0, "100.0%"),
        ],
    )
    def test_percentage_formatting(self, value, expected):
        fmt = NumberFormat(decimals=1, scale_by=100, suffix="%")
        assert fmt.format_value(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (500, "500"),
            (1234, "1K"),
            (1234567, "1M"),
            (1234567890, "1B"),
            (1234567890000, "1T"),
            (0, "0"),
        ],
    )
    def test_compact_formatting(self, value, expected):
        assert NumberFormat(compact=True).format_value(value) == expected

    def test_compact_with_decimals(self):
        fmt = NumberFormat(decimals=1, compact=True)
        assert fmt.format_value(1500) == "1.5K"
        assert fmt.format_value(2500000) == "2.5M"

    def test_german_locale(self):
        fmt = NumberFormat(decimals=2, locale="de_DE")
        assert fmt.format_value(1234.56) == "1.234,56"

    @pytest.mark.parametrize("value", [None, float("nan")])
    def test_none_and_nan_return_empty_string(self, value):
        assert NumberFormat(decimals=2).format_value(value) == ""

    def test_polars_expression(self):
        fmt = NumberFormat(decimals=2)
        df = pl.DataFrame({"value": [1234.567, 89.1]})
        result = df.with_columns(fmt.to_polars_expr("value").alias("formatted"))
        assert result["formatted"].to_list() == ["1,234.57", "89.10"]

    def test_format_polars_column(self):
        fmt = NumberFormat(decimals=1, scale_by=100, suffix="%")
        df = pl.DataFrame({"rate": [0.1234, 0.567]})
        result = fmt.format_polars_column(df, "rate")
        assert result["rate_formatted"].to_list() == ["12.3%", "56.7%"]

    def test_format_polars_column_custom_name(self):
        fmt = NumberFormat(decimals=0)
        df = pl.DataFrame({"count": [1000]})
        result = fmt.format_polars_column(df, "count", output_column="count_str")
        assert "count_str" in result.columns


class TestMetricFormats:
    """Tests for the MetricFormats registry."""

    def test_get_returns_format_for_known_metric(self):
        fmt = MetricFormats.get("ModelPerformance")
        assert fmt is not None
        assert fmt.decimals == 2

    def test_get_returns_none_for_unknown_metric(self):
        assert MetricFormats.get("NonExistentMetric") is None

    def test_get_or_default_returns_default_format(self):
        fmt = MetricFormats.get_or_default("UnknownMetric")
        assert fmt == MetricFormats.DEFAULT_FORMAT
        assert fmt.compact is True

    def test_has_format(self):
        assert MetricFormats.has_format("CTR") is True
        assert MetricFormats.has_format("NonExistent") is False

    def test_list_metrics_contains_predefined(self):
        metrics = MetricFormats.list_metrics()
        assert "ModelPerformance" in metrics
        assert "CTR" in metrics
        assert len(metrics) >= 6

    def test_register_and_retrieve_custom_format(self):
        custom_fmt = NumberFormat(decimals=4)
        MetricFormats.register("_TestMetric", custom_fmt)
        try:
            retrieved = MetricFormats.get("_TestMetric")
            assert retrieved is not None
            assert retrieved.decimals == 4
        finally:
            del MetricFormats._FORMATS["_TestMetric"]

    @pytest.mark.parametrize(
        "metric_id,value,expected",
        [
            ("ModelPerformance", 0.875, "87.50"),
            ("CTR", 0.00123, "0.123%"),
            ("EngagementLift", 0.15, "15%"),
        ],
    )
    def test_predefined_metric_formatting(self, metric_id, value, expected):
        fmt = MetricFormats.get(metric_id)
        assert fmt.format_value(value) == expected
