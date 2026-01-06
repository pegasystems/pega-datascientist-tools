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
        output_type="html", output_filename="test.html", execute_params=True
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


def test_write_params_files_size_reduction_method(tmp_path):
    """Test _write_params_files sets plotly-connected correctly based on size_reduction_method."""
    import yaml

    # Default (None) - embed plotly
    report_utils._write_params_files(
        tmp_path, params={"test": "value"}, size_reduction_method=None
    )
    with open(tmp_path / "_quarto.yml") as f:
        config = yaml.safe_load(f)
    assert config["format"]["html"]["embed-resources"] is True
    assert config["format"]["html"]["plotly-connected"] is True

    # "strip" - embed plotly
    report_utils._write_params_files(
        tmp_path, params={"test": "value"}, size_reduction_method="strip"
    )
    with open(tmp_path / "_quarto.yml") as f:
        config = yaml.safe_load(f)
    assert config["format"]["html"]["plotly-connected"] is True

    # "cdn" - load plotly from CDN
    report_utils._write_params_files(
        tmp_path, params={"test": "value"}, size_reduction_method="cdn"
    )
    with open(tmp_path / "_quarto.yml") as f:
        config = yaml.safe_load(f)
    assert config["format"]["html"]["plotly-connected"] is False
