"""
Testing the functionality of the report_utils module
"""

import polars as pl
import pytest
from pdstools.utils import report_utils


def test_get_version_only():
    """Test extracting version numbers from version strings"""
    assert report_utils._get_version_only("quarto 1.3.450") == "1.3.450"
    assert report_utils._get_version_only("pandoc 2.19.2") == "2.19.2"
    assert report_utils._get_version_only("v3.2.1-beta") == "3.2.1"
    # Now the implementation correctly handles pre-release and build metadata
    assert report_utils._get_version_only("1.0.0-alpha+001") == "1.0.0"


def test_polars_col_exists():
    """Test checking if column exists in dataframe"""
    df = pl.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]}).lazy()

    assert report_utils.polars_col_exists(df, "A")
    assert report_utils.polars_col_exists(df, "B")
    assert not report_utils.polars_col_exists(df, "C")


def test_polars_subset_to_existing_cols():
    """Test filtering list of columns to those that exist in dataframe"""
    all_columns = ["A", "B", "C"]

    # All columns exist
    assert report_utils.polars_subset_to_existing_cols(all_columns, ["A", "B"]) == [
        "A",
        "B",
    ]

    # Some columns don't exist
    assert report_utils.polars_subset_to_existing_cols(all_columns, ["A", "D"]) == ["A"]

    # No columns exist
    assert report_utils.polars_subset_to_existing_cols(all_columns, ["D", "E"]) == []

    # Empty list
    assert report_utils.polars_subset_to_existing_cols(all_columns, []) == []


def test_rag_background_styler():
    """Test RAG background styling"""

    # Test valid RAG values
    assert report_utils.rag_background_styler("R").color == "orangered"
    assert report_utils.rag_background_styler("A").color == "orange"
    assert report_utils.rag_background_styler("Y").color == "yellow"
    assert report_utils.rag_background_styler("G") is None  # Green has no background

    # Test lowercase
    assert report_utils.rag_background_styler("r").color == "orangered"

    # Test invalid RAG value
    with pytest.raises(ValueError):
        report_utils.rag_background_styler("X")

    # Test None
    with pytest.raises(ValueError):
        report_utils.rag_background_styler(None)


def test_rag_background_styler_dense():
    """Test dense RAG background styling"""

    # Test valid RAG values
    assert report_utils.rag_background_styler_dense("R").color == "orangered"
    assert report_utils.rag_background_styler_dense("A").color == "orange"
    assert report_utils.rag_background_styler_dense("Y").color == "yellow"
    assert report_utils.rag_background_styler_dense("G").color == "green"

    # Test lowercase
    assert report_utils.rag_background_styler_dense("g").color == "green"

    # Test invalid RAG value
    with pytest.raises(ValueError):
        report_utils.rag_background_styler_dense("X")


def test_rag_textcolor_styler():
    """Test RAG text color styling"""

    # Test valid RAG values
    assert report_utils.rag_textcolor_styler("R").color == "orangered"
    assert report_utils.rag_textcolor_styler("A").color == "orange"
    assert report_utils.rag_textcolor_styler("Y").color == "yellow"
    assert report_utils.rag_textcolor_styler("G").color == "green"

    # Test lowercase
    assert report_utils.rag_textcolor_styler("g").color == "green"

    # Test invalid RAG value
    with pytest.raises(ValueError):
        report_utils.rag_textcolor_styler("X")


def test_serialize_query():
    """Test query serialization"""
    import polars as pl

    # Test with None
    assert report_utils.serialize_query(None) is None

    # Test with single expression
    expr = pl.col("A") > 5
    result = report_utils.serialize_query(expr)
    assert result["type"] == "expr_list"
    assert "0" in result["expressions"]

    # Test with list of expressions
    expr_list = [pl.col("A") > 5, pl.col("B").is_null()]
    result = report_utils.serialize_query(expr_list)
    assert result["type"] == "expr_list"
    assert "0" in result["expressions"]
    assert "1" in result["expressions"]

    # Test with dict
    query_dict = {"column": ["value1", "value2"]}
    result = report_utils.serialize_query(query_dict)
    assert result["type"] == "dict"
    assert result["data"] == query_dict

    # Test with invalid type
    with pytest.raises(ValueError):
        report_utils.serialize_query("invalid")


def test_get_output_filename():
    """Test the get_output_filename function"""
    # Test with name and model_id for ModelReport
    filename = report_utils.get_output_filename(
        name="test_report",
        report_type="ModelReport",
        model_id="model1",
        output_type="html",
    )
    assert filename == "ModelReport_test_report_model1.html"

    # Test without name for ModelReport
    filename = report_utils.get_output_filename(
        name=None, report_type="ModelReport", model_id="model1", output_type="html"
    )
    assert filename == "ModelReport_model1.html"

    # Test with name for HealthCheck
    filename = report_utils.get_output_filename(
        name="test_report", report_type="HealthCheck", model_id=None, output_type="html"
    )
    assert filename == "HealthCheck_test_report.html"

    # Test without name for HealthCheck
    filename = report_utils.get_output_filename(
        name=None, report_type="HealthCheck", model_id=None, output_type="html"
    )
    assert filename == "HealthCheck.html"

    # Test with spaces in name
    filename = report_utils.get_output_filename(
        name="test report", report_type="HealthCheck", model_id=None, output_type="html"
    )
    assert filename == "HealthCheck_test_report.html"


def test_set_command_options():
    """Test the _set_command_options function"""
    # Test basic options
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

    # Test with no options
    options = report_utils._set_command_options()
    assert options == []

    # Test with only output_type
    options = report_utils._set_command_options(output_type="pdf")
    assert options == ["--to", "pdf"]

    # Test with only output_filename
    options = report_utils._set_command_options(output_filename="report.html")
    assert options == ["--output", "report.html"]

    # Test with execute_params=False
    options = report_utils._set_command_options(execute_params=False)
    assert options == []


def test_write_params_files_size_reduction_method(tmp_path):
    """Test that _write_params_files sets embed-resources and plotly-connected correctly based on size_reduction_method"""
    import yaml

    # Test with size_reduction_method=None (default) - should embed resources and plotly
    report_utils._write_params_files(
        temp_dir=tmp_path,
        params={"test": "value"},
        size_reduction_method=None,
    )
    with open(tmp_path / "_quarto.yml") as f:
        config = yaml.safe_load(f)
    assert config["format"]["html"]["embed-resources"] is True
    assert config["format"]["html"]["plotly-connected"] is True

    # Test with size_reduction_method="strip" - should embed resources and plotly
    report_utils._write_params_files(
        temp_dir=tmp_path,
        params={"test": "value"},
        size_reduction_method="strip",
    )
    with open(tmp_path / "_quarto.yml") as f:
        config = yaml.safe_load(f)
    assert config["format"]["html"]["embed-resources"] is True
    assert config["format"]["html"]["plotly-connected"] is True

    # Test with size_reduction_method="cdn" - should embed resources but NOT plotly (load from CDN)
    report_utils._write_params_files(
        temp_dir=tmp_path,
        params={"test": "value"},
        size_reduction_method="cdn",
    )
    with open(tmp_path / "_quarto.yml") as f:
        config = yaml.safe_load(f)
    assert config["format"]["html"]["embed-resources"] is True
    assert config["format"]["html"]["plotly-connected"] is False


@pytest.mark.skip(reason="Requires mocking __reports__ import")
def test_copy_quarto_file(tmp_path):
    """Test the copy_quarto_file function"""
    # This test would require mocking the __reports__ import
    pass


@pytest.mark.skip(reason="Requires mocking subprocess and get_quarto_with_version")
def test_run_quarto():
    """Test the run_quarto function"""
    # This test would require mocking subprocess.Popen and get_quarto_with_version
    pass
