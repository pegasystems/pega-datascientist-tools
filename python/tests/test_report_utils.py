"""Tests for the report_utils module."""

import polars as pl
import pytest
from pdstools.utils import report_utils


def test_get_version_only():
    """Test extracting version numbers from version strings."""
    assert report_utils._get_version_only("quarto 1.3.450") == "1.3.450"
    assert report_utils._get_version_only("pandoc 2.19.2") == "2.19.2"
    assert report_utils._get_version_only("v3.2.1-beta") == "3.2.1"
    assert report_utils._get_version_only("1.0.0-alpha+001") == "1.0.0"


def test_polars_col_exists():
    """Test checking if column exists in dataframe."""
    df = pl.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]}).lazy()

    assert report_utils.polars_col_exists(df, "A")
    assert report_utils.polars_col_exists(df, "B")
    assert not report_utils.polars_col_exists(df, "C")


def test_polars_subset_to_existing_cols():
    """Test filtering list of columns to those that exist."""
    all_columns = ["A", "B", "C"]

    assert report_utils.polars_subset_to_existing_cols(all_columns, ["A", "B"]) == [
        "A",
        "B",
    ]
    assert report_utils.polars_subset_to_existing_cols(all_columns, ["A", "D"]) == ["A"]
    assert report_utils.polars_subset_to_existing_cols(all_columns, ["D", "E"]) == []
    assert report_utils.polars_subset_to_existing_cols(all_columns, []) == []


def test_serialize_query():
    """Test query serialization."""
    assert report_utils.serialize_query(None) is None

    # Single expression
    expr = pl.col("A") > 5
    result = report_utils.serialize_query(expr)
    assert result["type"] == "expr_list"
    assert "0" in result["expressions"]

    # List of expressions
    expr_list = [pl.col("A") > 5, pl.col("B").is_null()]
    result = report_utils.serialize_query(expr_list)
    assert result["type"] == "expr_list"
    assert "0" in result["expressions"]
    assert "1" in result["expressions"]

    # Dict
    query_dict = {"column": ["value1", "value2"]}
    result = report_utils.serialize_query(query_dict)
    assert result["type"] == "dict"
    assert result["data"] == query_dict

    # Invalid type
    with pytest.raises(ValueError):
        report_utils.serialize_query("invalid")


def test_get_output_filename():
    """Test the get_output_filename function."""
    assert (
        report_utils.get_output_filename("test_report", "ModelReport", "model1", "html")
        == "ModelReport_test_report_model1.html"
    )
    assert (
        report_utils.get_output_filename(None, "ModelReport", "model1", "html")
        == "ModelReport_model1.html"
    )
    assert (
        report_utils.get_output_filename("test_report", "HealthCheck", None, "html")
        == "HealthCheck_test_report.html"
    )
    assert (
        report_utils.get_output_filename(None, "HealthCheck", None, "html")
        == "HealthCheck.html"
    )
    assert (
        report_utils.get_output_filename("test report", "HealthCheck", None, "html")
        == "HealthCheck_test_report.html"
    )


def test_set_command_options():
    """Test the _set_command_options function."""
    options = report_utils._set_command_options(
        output_type="html",
        output_filename="test.html",
        execute_params=True,
    )
    assert "--to" in options
    assert "html" in options
    assert "--output" in options
    assert "test.html" in options
    assert "--execute-params" in options
    assert "params.yml" in options

    assert report_utils._set_command_options() == []
    assert report_utils._set_command_options(output_type="pdf") == ["--to", "pdf"]
    assert report_utils._set_command_options(output_filename="report.html") == [
        "--output",
        "report.html",
    ]
    assert report_utils._set_command_options(execute_params=False) == []


def test_create_metric_gttable_column_descriptions():
    """Test that column_descriptions adds tooltips to GT table headers."""
    df = pl.DataFrame(
        {
            "Model": ["A", "B", "C"],
            "Performance": [0.72, 0.58, 0.85],
            "Channel": ["Web", "Email", "Mobile"],
        },
    )

    column_descriptions = {
        "Model": "The predictive model name",
        "Performance": "Model AUC score (0.5-1.0)",
    }

    gt = report_utils.create_metric_gttable(
        df,
        column_descriptions=column_descriptions,
        strict_metric_validation=False,
    )

    # Convert to HTML and check for title attributes
    html = gt.as_raw_html()
    assert 'title="The predictive model name"' in html
    assert 'title="Model AUC score (0.5-1.0)"' in html
    # Channel was not in column_descriptions, so should not have tooltip
    assert html.count('title="') == 2


def test_create_metric_itable_column_descriptions(monkeypatch):
    """Test that column_descriptions renames columns with HTML tooltips for itables."""
    df = pl.DataFrame(
        {
            "Model": ["A", "B", "C"],
            "Performance": [0.72, 0.58, 0.85],
            "Channel": ["Web", "Email", "Mobile"],
        },
    )

    column_descriptions = {
        "Model": "The predictive model name",
        "Performance": "Model AUC score (0.5-1.0)",
    }

    # Mock itables.show to capture what it receives
    captured_df = None

    def mock_show(styled_df, **kwargs):
        nonlocal captured_df
        # The styled_df is a pandas Styler, get the underlying DataFrame
        captured_df = styled_df.data

    import itables

    monkeypatch.setattr(itables, "show", mock_show)

    report_utils.create_metric_itable(
        df,
        column_descriptions=column_descriptions,
        strict_metric_validation=False,
    )

    assert captured_df is not None
    # Check that Model column was renamed to include tooltip
    model_col = [c for c in captured_df.columns if "Model" in c][0]
    assert 'title="The predictive model name"' in model_col
    assert "<span" in model_col

    # Check that Performance column was renamed to include tooltip
    perf_col = [c for c in captured_df.columns if "Performance" in c][0]
    assert 'title="Model AUC score (0.5-1.0)"' in perf_col

    # Channel was not in column_descriptions, so should not have tooltip
    channel_col = [c for c in captured_df.columns if "Channel" in c][0]
    assert channel_col == "Channel"  # unchanged


def test_create_metric_gttable_without_column_descriptions():
    """Test that GT table works normally without column_descriptions."""
    df = pl.DataFrame(
        {
            "Model": ["A", "B"],
            "Value": [1.0, 2.0],
        },
    )

    gt = report_utils.create_metric_gttable(
        df,
        strict_metric_validation=False,
    )

    html = gt.as_raw_html()
    # No tooltips should be added
    assert 'title="' not in html


def test_write_params_files_size_reduction_method(tmp_path):
    """Test _write_params_files sets plotly-connected correctly based on size_reduction_method."""
    import yaml

    # Default (None) - embed plotly
    report_utils._write_params_files(
        tmp_path,
        params={"test": "value"},
        size_reduction_method=None,
    )
    with open(tmp_path / "_quarto.yml") as f:
        config = yaml.safe_load(f)
    assert config["format"]["html"]["embed-resources"] is True
    assert config["format"]["html"]["plotly-connected"] is True

    # "strip" - embed plotly
    report_utils._write_params_files(
        tmp_path,
        params={"test": "value"},
        size_reduction_method="strip",
    )
    with open(tmp_path / "_quarto.yml") as f:
        config = yaml.safe_load(f)
    assert config["format"]["html"]["plotly-connected"] is True

    # "cdn" - load plotly from CDN
    report_utils._write_params_files(
        tmp_path,
        params={"test": "value"},
        size_reduction_method="cdn",
    )
    with open(tmp_path / "_quarto.yml") as f:
        config = yaml.safe_load(f)
    assert config["format"]["html"]["plotly-connected"] is False


def test_create_metric_gttable_callable_rag_with_metric_format():
    """Test that columns with callable RAG functions still get formatting from MetricFormats.

    This is critical for the Channel Overview table in HealthCheck.qmd where CTR
    (Base Rate) uses exclusive_0_1_range_rag for RAG coloring but should still be
    formatted as a percentage with 3 decimals from the MetricFormats registry.

    Fixes: https://github.com/pegasystems/pega-datascientist-tools/issues/527
    """
    from pdstools.utils.metric_limits import exclusive_0_1_range_rag

    df = pl.DataFrame(
        {
            "Channel": ["Web", "Mobile", "Email"],
            "CTR": [0.023456, 0.00123, 0.1],  # Small decimal values like real CTR
        },
    )

    gt = report_utils.create_metric_gttable(
        df,
        column_to_metric={
            "CTR": exclusive_0_1_range_rag,  # Callable for RAG
        },
        strict_metric_validation=False,
    )

    html = gt.as_raw_html()

    # CTR should be formatted as percentage with 3 decimals (from MetricFormats)
    # 0.023456 -> "2.346%" (3 decimals)
    # 0.00123 -> "0.123%"
    # 0.1 -> "10.000%"
    assert "2.346%" in html
    assert "0.123%" in html
    assert "10.000%" in html

    # Should NOT show raw decimal values
    assert "0.023456" not in html
    assert "0.00123" not in html


def test_create_metric_gttable_callable_rag_without_format():
    """Test that columns with callable RAG but no MetricFormat use default formatting."""
    df = pl.DataFrame(
        {
            "Channel": ["Web", "Mobile"],
            "CustomMetric": [12345.678, 99999.123],  # No format defined for this
        },
    )

    def custom_rag(value):
        return "GREEN" if value > 50000 else "AMBER"

    gt = report_utils.create_metric_gttable(
        df,
        column_to_metric={
            "CustomMetric": custom_rag,  # Callable for RAG, no format defined
        },
        strict_metric_validation=False,
    )

    html = gt.as_raw_html()

    # Should use default compact formatting (from MetricFormats.DEFAULT_FORMAT)
    # Large numbers get formatted with K/M suffixes or similar
    # At minimum, shouldn't show raw floating point values
    assert "CustomMetric" in html  # Column exists
    # Default format is NumberFormat(decimals=0, compact=True)
    # 12345.678 -> "12K" or similar compact format
    # 99999.123 -> "100K" or similar compact format
    assert "12345.678" not in html  # Raw value should not appear
