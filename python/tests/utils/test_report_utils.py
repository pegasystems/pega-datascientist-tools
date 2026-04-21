"""Tests for the report_utils module."""

import itables
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
    assert report_utils.get_output_filename(None, "ModelReport", "model1", "html") == "ModelReport_model1.html"
    assert (
        report_utils.get_output_filename("test_report", "HealthCheck", None, "html") == "HealthCheck_test_report.html"
    )
    assert report_utils.get_output_filename(None, "HealthCheck", None, "html") == "HealthCheck.html"
    assert (
        report_utils.get_output_filename("test report", "HealthCheck", None, "html") == "HealthCheck_test_report.html"
    )

    # When `name` already ends with `.<output_type>`, return it verbatim
    # (after underscore-substitution) instead of double-suffixing.
    assert report_utils.get_output_filename("my_report.html", "HealthCheck", None, "html") == "my_report.html"
    assert report_utils.get_output_filename("My Report.HTML", "HealthCheck", None, "html") == "My_Report.HTML"
    assert report_utils.get_output_filename("my_report.html", "ModelReport", "model1", "html") == "my_report.html"
    # Mismatched extension still gets the canonical extension appended.
    assert (
        report_utils.get_output_filename("my_report.html", "HealthCheck", None, "pdf")
        == "HealthCheck_my_report.html.pdf"
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


@pytest.fixture()
def capture_itable(monkeypatch):
    """Capture the styled DataFrame and kwargs passed to itables.show."""
    captured = {}

    def mock_show(styled_df, **kwargs):
        captured["df"] = styled_df.data
        captured["kwargs"] = kwargs

    monkeypatch.setattr(itables, "show", mock_show)
    return captured


def _assert_numeric_sort_wired(captured, col: str, raw_values: list):
    """Assert that a hidden sort column and matching columnDefs are present."""
    sort_col = f"__sort__{col}"
    df = captured["df"]
    col_defs = captured["kwargs"].get("columnDefs", [])
    final_cols = list(df.columns)

    assert sort_col in final_cols, f"Hidden sort column {sort_col!r} not found"
    assert list(df[sort_col]) == raw_values

    display_idx = final_cols.index(col)
    sort_idx = final_cols.index(sort_col)

    order_def = next((d for d in col_defs if d.get("targets") == display_idx), None)
    assert order_def is not None, f"No columnDef found for display column {col!r}"
    assert order_def["orderData"] == [sort_idx]

    hidden_def = next((d for d in col_defs if d.get("targets") == sort_idx), None)
    assert hidden_def is not None, f"No columnDef found for sort column {sort_col!r}"
    assert hidden_def["visible"] is False
    assert hidden_def["searchable"] is False


@pytest.mark.parametrize(
    "source_table, rag_source, extra_kwargs",
    [
        # source_table is already numeric, metric resolved via string ID
        (
            pl.DataFrame({"Name": ["A", "B", "C"], "Count": [500, 4000, 200]}),
            None,
            {},
        ),
        # source_table has pre-formatted strings (e.g. after apply_display_formatting),
        # rag_source holds the raw numerics
        (
            pl.DataFrame({"Name": ["A", "B", "C"], "Count": ["500", "4K", "200"]}),
            pl.DataFrame({"Name": ["A", "B", "C"], "Count": [500, 4000, 200]}),
            {},
        ),
        # metric_id is a callable (else branch): numeric column with custom RAG function
        (
            pl.DataFrame({"Name": ["A", "B", "C"], "Count": [500, 4000, 200]}),
            None,
            {"column_to_metric": {"Count": lambda v: "GREEN" if v > 300 else "RED"}},
        ),
    ],
    ids=["numeric_source", "preformatted_with_rag_source", "callable_metric"],
)
def test_create_metric_itable_numeric_sort(capture_itable, source_table, rag_source, extra_kwargs):
    """All formatted numeric columns sort numerically, not lexicographically.

    Pandas Styler converts numbers to display strings, which DataTables would
    otherwise sort as text (e.g. "4K" < "500", "100%" < "2%"). A hidden
    raw-value column is added and DataTables is configured to sort by it.
    Covers a plain numeric source, a pre-formatted source with rag_source, and a
    numeric column mapped to a callable RAG function (the else branch).
    """
    report_utils.create_metric_itable(
        source_table,
        rag_source=rag_source,
        strict_metric_validation=False,
        **extra_kwargs,
    )
    _assert_numeric_sort_wired(capture_itable, "Count", [500, 4000, 200])


def test_create_metric_itable_rag_source_no_format_error(monkeypatch):
    """No ValueError when source_table has pre-formatted strings.

    format_dict must only be populated for columns that are still numeric in
    source_table. Format strings (e.g. ModelPerformance -> '{:,.2f}') must not
    be applied to string values or pandas raises a ValueError.
    """
    numeric_df = pl.DataFrame({"Model": ["A", "B"], "Performance": [0.72, 0.58]})
    display_df = pl.DataFrame({"Model": ["A", "B"], "Performance": ["72.00", "58.00"]})

    def mock_show(styled_df, **kwargs):
        # Force rendering so any format errors surface here
        styled_df.to_html()

    monkeypatch.setattr(itables, "show", mock_show)

    report_utils.create_metric_itable(
        display_df,
        column_to_metric={"Performance": "ModelPerformance"},
        rag_source=numeric_df,
        strict_metric_validation=True,
    )


def test_create_metric_itable_user_columndefs_preserved(capture_itable):
    """User-provided columnDefs are merged with, not replaced by, sort defs."""
    df = pl.DataFrame({"Group": ["A", "A", "B"], "Count": [500, 4000, 200]})
    user_col_defs = [{"targets": 0, "visible": False}]

    report_utils.create_metric_itable(
        df,
        strict_metric_validation=False,
        columnDefs=user_col_defs,
    )

    col_defs = capture_itable["kwargs"].get("columnDefs", [])
    # User's def must still be present
    assert {"targets": 0, "visible": False} in col_defs
    # Sort defs must also be present
    sort_col = "__sort__Count"
    final_cols = list(capture_itable["df"].columns)
    sort_idx = final_cols.index(sort_col)
    assert any(d.get("targets") == sort_idx for d in col_defs)


def test_create_metric_itable_percentage_sort(capture_itable):
    """Percentage-formatted columns must sort numerically, not lexicographically.

    Without the fix, "100.000%" sorts before "2.000%" because "1" < "2" as strings.
    Uses OmniChannelPercentage (scale_by=100, suffix="%") as a representative
    non-compact percentage metric.
    """
    df = pl.DataFrame(
        {
            "Name": ["A", "B", "C"],
            # OmniChannelPercentage: scale_by=100, suffix="%", decimals=1
            # formats to "2.0%", "100.0%", "50.0%" — "100.0%" < "2.0%" lexicographically
            "OmniChannelPercentage": [0.02, 1.0, 0.5],
        }
    )
    report_utils.create_metric_itable(
        df,
        column_to_metric={"OmniChannelPercentage": "OmniChannelPercentage"},
        strict_metric_validation=True,
    )
    _assert_numeric_sort_wired(capture_itable, "OmniChannelPercentage", [0.02, 1.0, 0.5])


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


def test_write_params_files_full_embed(tmp_path):
    """Test _write_params_files sets plotly-connected correctly based on full_embed."""
    import yaml

    # full_embed=True - embed plotly
    report_utils._write_params_files(
        tmp_path,
        params={"test": "value"},
        full_embed=True,
    )
    with open(tmp_path / "_quarto.yml") as f:
        config = yaml.safe_load(f)
    assert config["format"]["html"]["embed-resources"] is True
    assert config["format"]["html"]["plotly-connected"] is True

    # full_embed=False (default) - load plotly from CDN, disable embed-resources to avoid esbuild (#620)
    report_utils._write_params_files(
        tmp_path,
        params={"test": "value"},
        full_embed=False,
    )
    with open(tmp_path / "_quarto.yml") as f:
        config = yaml.safe_load(f)
    assert config["format"]["html"]["plotly-connected"] is False
    assert config["format"]["html"]["embed-resources"] is False


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


class TestGetVersionOnlyEdgeCases:
    """Edge cases for _get_version_only not covered by test_get_version_only."""

    def test_empty_string(self):
        assert report_utils._get_version_only("") == ""

    def test_no_digits(self):
        assert report_utils._get_version_only("no version here") == ""

    def test_single_number(self):
        assert report_utils._get_version_only("version 42") == "42"

    def test_leading_v(self):
        assert report_utils._get_version_only("v1.2.3") == "1.2.3"

    def test_version_with_trailing_text(self):
        assert report_utils._get_version_only("1.3.450 (some build info)") == "1.3.450"

    def test_four_part_version(self):
        assert report_utils._get_version_only("tool 10.20.30.40") == "10.20.30.40"


class TestGenerateZippedReport:
    """Tests for generate_zipped_report."""

    def test_creates_zip_from_directory(self, tmp_path):
        """Test that a valid directory is zipped successfully."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "file1.txt").write_text("hello")
        (source_dir / "file2.txt").write_text("world")

        output_name = str(tmp_path / "my_archive.html")
        report_utils.generate_zipped_report(output_name, str(source_dir))

        expected_zip = tmp_path / "my_archive.zip"
        assert expected_zip.exists()
        assert expected_zip.stat().st_size > 0

    def test_creates_zip_strips_extension(self, tmp_path):
        """Test that the output filename extension is replaced with .zip."""
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "data.csv").write_text("a,b,c")

        output_name = str(tmp_path / "report.pdf")
        report_utils.generate_zipped_report(output_name, str(source_dir))

        assert (tmp_path / "report.zip").exists()
        assert not (tmp_path / "report.pdf").exists()

    def test_nonexistent_directory_returns_none(self, tmp_path):
        """Test that a non-existent directory logs an error and returns."""
        fake_dir = str(tmp_path / "does_not_exist")
        # Should not raise — just logs and returns
        result = report_utils.generate_zipped_report("output.html", fake_dir)
        assert result is None

    def test_file_instead_of_directory(self, tmp_path):
        """Test that passing a file path (not a directory) logs an error and returns."""
        a_file = tmp_path / "not_a_dir.txt"
        a_file.write_text("I am a file")
        result = report_utils.generate_zipped_report("output.html", str(a_file))
        assert result is None


class TestBundleQuartoResources:
    """Tests for bundle_quarto_resources."""

    def test_bundles_html_with_resources_folder(self, tmp_path):
        """When a `<stem>_files` folder exists, both are zipped into `<stem>.zip`."""
        import zipfile

        html_path = tmp_path / "report.html"
        html_path.write_text("<html>hi</html>")
        resources = tmp_path / "report_files"
        (resources / "libs" / "plotly").mkdir(parents=True)
        (resources / "libs" / "plotly" / "plotly.js").write_text("/* js */")
        (resources / "figure-html" / "fig.png").parent.mkdir(parents=True)
        (resources / "figure-html" / "fig.png").write_text("png")

        result = report_utils.bundle_quarto_resources(html_path)

        assert result == tmp_path / "report.zip"
        assert result.exists()
        assert not html_path.exists()
        assert not resources.exists()
        with zipfile.ZipFile(result) as zf:
            names = set(zf.namelist())
        assert "report.html" in names
        assert "report_files/libs/plotly/plotly.js" in names
        assert "report_files/figure-html/fig.png" in names

    def test_no_resources_folder_is_noop(self, tmp_path):
        """Without a companion folder, the original file path is returned unchanged."""
        html_path = tmp_path / "solo.html"
        html_path.write_text("<html>embedded</html>")

        result = report_utils.bundle_quarto_resources(html_path)

        assert result == html_path
        assert html_path.exists()
        assert not (tmp_path / "solo.zip").exists()

    def test_missing_file_returns_same_path(self, tmp_path):
        """A non-existent input is returned unchanged without error."""
        missing = tmp_path / "ghost.html"
        result = report_utils.bundle_quarto_resources(missing)
        assert result == missing


class TestDeserializeQuery:
    """Tests for deserialize_query and serialize/deserialize roundtrip."""

    def test_none_roundtrip(self):
        assert report_utils.deserialize_query(None) is None

    def test_expr_list_roundtrip(self):
        """Expressions survive a serialize → deserialize roundtrip."""
        original = [pl.col("A") > 5, pl.col("B").is_null()]
        serialized = report_utils.serialize_query(original)
        restored = report_utils.deserialize_query(serialized)
        assert isinstance(restored, list)
        assert len(restored) == 2
        # Verify expressions work on real data
        df = pl.DataFrame({"A": [3, 6, 10], "B": [None, "x", None]})
        filtered = df.filter(restored)
        assert len(filtered) == 1  # only row (10, None) matches both

    def test_dict_roundtrip(self):
        original = {"col": ["v1", "v2"]}
        serialized = report_utils.serialize_query(original)
        restored = report_utils.deserialize_query(serialized)
        assert restored == original

    def test_single_expr_roundtrip(self):
        """A single expression (not a list) survives roundtrip."""
        original = pl.col("score") >= 0.5
        serialized = report_utils.serialize_query(original)
        restored = report_utils.deserialize_query(serialized)
        assert isinstance(restored, list)
        assert len(restored) == 1
        # Verify the expression works
        df = pl.DataFrame({"score": [0.3, 0.5, 0.9]})
        assert len(df.filter(restored)) == 2

    def test_unknown_type_raises(self):
        """Deserializing an unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown query type"):
            report_utils.deserialize_query({"type": "unknown"})


class TestShowCredits:
    """Tests for show_credits output formatting."""

    @pytest.fixture(autouse=True)
    def _mock_dependencies(self, monkeypatch):
        """Mock external dependencies for deterministic output."""
        monkeypatch.setattr(
            report_utils._quarto,
            "get_quarto_with_version",
            lambda: (None, "1.4.0"),
        )
        monkeypatch.setattr(
            report_utils._quarto,
            "get_pandoc_with_version",
            lambda: (None, "3.1.0"),
        )
        self.printed_text = ""

        def capture_print(text):
            self.printed_text = text

        monkeypatch.setattr(report_utils._quarto, "quarto_print", capture_print)

    def test_with_source(self):
        """Test output includes notebook path when quarto_source is provided."""
        report_utils.show_credits("reports/HealthCheck.qmd")

        assert "Document created at:" in self.printed_text
        assert "This notebook: reports/HealthCheck.qmd" in self.printed_text
        assert "Quarto runtime: 1.4.0" in self.printed_text
        assert "Pandoc: 3.1.0" in self.printed_text

    def test_without_source(self):
        """Test output omits notebook line when quarto_source is None."""
        report_utils.show_credits()

        assert "Document created at:" in self.printed_text
        assert "This notebook" not in self.printed_text
        assert "Quarto runtime: 1.4.0" in self.printed_text
        assert "Pandoc: 3.1.0" in self.printed_text

    def test_with_empty_string_source(self):
        """Test empty string source is treated as no source."""
        report_utils.show_credits("")

        assert "This notebook" not in self.printed_text


class TestGainsTable:
    """Tests for gains_table utility function."""

    def test_basic_gains_calculation(self):
        """Test basic gains table with simple data."""
        df = pl.DataFrame(
            {
                "ModelID": ["A", "B", "C", "D"],
                "ResponseCount": [100, 200, 50, 150],
            }
        ).lazy()

        result = report_utils.gains_table(df, value="ResponseCount")

        assert isinstance(result, pl.DataFrame)
        assert "cum_x" in result.columns
        assert "cum_y" in result.columns

        # Cumulative y should end at 1.0 (100%)
        assert result["cum_y"][-1] == pytest.approx(1.0, abs=0.01)

        # Should include (0,0) starting point plus 4 data points
        assert result.height == 5

    def test_gains_with_index(self):
        """Test gains table with index parameter for normalization."""
        df = pl.DataFrame(
            {
                "Action": ["A", "B", "C"],
                "Positives": [100, 200, 50],
                "ResponseCount": [1000, 2000, 500],
            }
        ).lazy()

        result = report_utils.gains_table(df, value="Positives", index="ResponseCount")

        assert isinstance(result, pl.DataFrame)
        assert result.height == 4  # 3 data points + (0,0)
        # Should sort by Positives/ResponseCount ratio
        assert "cum_x" in result.columns
        assert "cum_y" in result.columns
        assert result["cum_y"][-1] == pytest.approx(1.0, abs=0.01)

    def test_gains_with_grouping(self):
        """Test gains table with by parameter for grouping."""
        df = pl.DataFrame(
            {
                "Channel": ["Web", "Web", "Email", "Email"],
                "Action": ["A", "B", "C", "D"],
                "ResponseCount": [100, 200, 150, 50],
            }
        ).lazy()

        result = report_utils.gains_table(df, value="ResponseCount", by="Channel")

        assert isinstance(result, pl.DataFrame)
        # Should have data for both channels
        assert "Channel" in result.columns
        channels = result["Channel"].unique().to_list()
        assert set(channels) == {"Web", "Email"}

        # Each channel should have cumulative y ending at 1.0
        for channel in channels:
            channel_data = result.filter(pl.col("Channel") == channel)
            last_y = channel_data["cum_y"][-1]
            assert last_y == pytest.approx(1.0, abs=0.01)

    def test_gains_single_row(self):
        """Test gains table with single row edge case."""
        df = pl.DataFrame({"Model": ["A"], "Value": [100]}).lazy()

        result = report_utils.gains_table(df, value="Value")

        assert isinstance(result, pl.DataFrame)
        assert result.height == 2  # (0,0) + 1 data point
        assert result["cum_y"][1] == pytest.approx(1.0, abs=0.01)

    def test_gains_zero_values(self):
        """Test gains table with zero values."""
        df = pl.DataFrame(
            {
                "Model": ["A", "B", "C"],
                "Value": [0, 100, 200],
            }
        ).lazy()

        result = report_utils.gains_table(df, value="Value")

        assert isinstance(result, pl.DataFrame)
        assert result.height == 4  # (0,0) + 3 data points
        # Should handle zeros gracefully
        assert "cum_x" in result.columns
        assert "cum_y" in result.columns


class _FakeDM:
    """Minimal stand-in for an ADMDatamart-like object exposing .model_data."""

    def __init__(self, df):
        self.model_data = df.lazy() if isinstance(df, pl.DataFrame) else df


class TestHierarchyHelpers:
    """Exact-value tests for n_unique_values, max_by_hierarchy, avg_by_hierarchy, sample_values."""

    def _dm(self):
        df = pl.DataFrame(
            {
                "Channel": ["Web", "Web", "Web", "Email", "Email", "SMS"],
                "Issue": ["Sales", "Sales", "Retention", "Sales", "Retention", "Sales"],
                "Action": ["A", "B", "C", "A", "D", "E"],
                "Name": ["a", "b", "c", "a", "d", "e"],
                "Nulls": [None, None, None, None, None, None],
            }
        )
        return _FakeDM(df), df.columns

    def test_n_unique_values_single_field(self):
        dm, cols = self._dm()
        # 5 distinct Action values ("A" appears twice)
        assert report_utils.n_unique_values(dm, cols, "Action") == 5

    def test_n_unique_values_field_as_list(self):
        dm, cols = self._dm()
        # Channel+Issue combinations: Web/Sales, Web/Retention, Email/Sales, Email/Retention, SMS/Sales
        assert report_utils.n_unique_values(dm, cols, ["Channel", "Issue"]) == 5

    def test_n_unique_values_missing_field_returns_zero(self):
        dm, cols = self._dm()
        assert report_utils.n_unique_values(dm, cols, "DoesNotExist") == 0
        assert report_utils.n_unique_values(dm, cols, ["DoesNotExist"]) == 0

    def test_max_by_hierarchy(self):
        dm, cols = self._dm()
        # Per Channel: Web has 3 unique actions, Email has 2, SMS has 1 -> max=3
        assert report_utils.max_by_hierarchy(dm, cols, "Action", ["Channel"]) == 3

    def test_max_by_hierarchy_missing_field(self):
        dm, cols = self._dm()
        assert report_utils.max_by_hierarchy(dm, cols, "Missing", ["Channel"]) == 0

    def test_max_by_hierarchy_missing_grouping(self):
        dm, cols = self._dm()
        assert report_utils.max_by_hierarchy(dm, cols, "Action", ["Missing"]) == 0

    def test_avg_by_hierarchy(self):
        dm, cols = self._dm()
        # Per Channel: Web=3, Email=2, SMS=1 -> mean=(3+2+1)/3=2.0
        assert report_utils.avg_by_hierarchy(dm, cols, "Action", ["Channel"]) == pytest.approx(2.0)

    def test_avg_by_hierarchy_missing_field(self):
        dm, cols = self._dm()
        assert report_utils.avg_by_hierarchy(dm, cols, "Missing", ["Channel"]) == 0

    def test_avg_by_hierarchy_missing_grouping(self):
        dm, cols = self._dm()
        assert report_utils.avg_by_hierarchy(dm, cols, "Action", ["Missing"]) == 0

    def test_sample_values_default(self):
        dm, cols = self._dm()
        result = report_utils.sample_values(dm, cols, "Action")
        # Sorted unique: A, B, C, D, E
        assert result == ["A", "B", "C", "D", "E"]

    def test_sample_values_truncated(self):
        dm, cols = self._dm()
        result = report_utils.sample_values(dm, cols, "Action", n=2)
        assert result == ["A", "B"]

    def test_sample_values_concat_fields(self):
        dm, cols = self._dm()
        result = report_utils.sample_values(dm, cols, ["Channel", "Action"])
        # Unique (Channel/Action) combos sorted lexicographically
        assert result == ["Email/A", "Email/D", "SMS/E", "Web/A", "Web/B", "Web/C"]

    def test_sample_values_missing_field_returns_dash(self):
        dm, cols = self._dm()
        assert report_utils.sample_values(dm, cols, "Missing") == "-"
        assert report_utils.sample_values(dm, cols, ["Missing"]) == "-"


class TestPolarsColExistsNullType:
    """polars_col_exists should treat all-Null columns as non-existent."""

    def test_null_dtype_column_is_false(self):
        df = pl.DataFrame({"A": [1, 2, 3], "B": [None, None, None]}).lazy()
        assert report_utils.polars_col_exists(df, "A")
        # "B" has dtype pl.Null because all values are None at construction time
        assert not report_utils.polars_col_exists(df, "B")


class TestCheckReportForErrors:
    """Tests for check_report_for_errors utility function."""

    def test_clean_report(self, tmp_path):
        """Test that a clean HTML report returns no errors."""
        html_file = tmp_path / "clean_report.html"
        html_file.write_text(
            """
            <html>
                <head><title>Test Report</title></head>
                <body>
                    <h1>Health Check Report</h1>
                    <p>Everything is working correctly.</p>
                </body>
            </html>
            """
        )

        errors = report_utils.check_report_for_errors(html_file)
        assert errors == []

    def test_report_with_rendering_error(self, tmp_path):
        """Test detection of plot rendering errors."""
        html_file = tmp_path / "error_report.html"
        html_file.write_text(
            """
            <html>
                <body>
                    <div class="callout-important">
                        <p>Error rendering Model Performance plot: ValueError</p>
                    </div>
                </body>
            </html>
            """
        )

        errors = report_utils.check_report_for_errors(html_file)
        assert len(errors) > 0
        assert any("Plot rendering error" in e for e in errors)

    def test_report_with_traceback(self, tmp_path):
        """Test detection of Python tracebacks."""
        html_file = tmp_path / "traceback_report.html"
        html_file.write_text(
            """
            <html>
                <body>
                    <pre>
                    Traceback (most recent call last):
                      File "test.py", line 10, in function
                        raise ValueError("Something went wrong")
                    ValueError: Something went wrong
                    </pre>
                </body>
            </html>
            """
        )

        errors = report_utils.check_report_for_errors(html_file)
        assert len(errors) >= 2  # Should catch both "Traceback" and "ValueError:"
        assert any("Python traceback" in e for e in errors)
        assert any("ValueError" in e for e in errors)

    def test_report_with_multiple_errors(self, tmp_path):
        """Test counting of multiple occurrences of the same error."""
        html_file = tmp_path / "multi_error_report.html"
        html_file.write_text(
            """
            <html>
                <body>
                    <p>Error rendering Chart 1: TypeError</p>
                    <p>Error rendering Chart 2: TypeError</p>
                    <p>Error rendering Chart 3: TypeError</p>
                </body>
            </html>
            """
        )

        errors = report_utils.check_report_for_errors(html_file)
        # Should report count of errors
        assert any("found 3 times" in e for e in errors)

    def test_nonexistent_file(self, tmp_path):
        """Test that nonexistent file raises FileNotFoundError."""
        html_file = tmp_path / "does_not_exist.html"

        with pytest.raises(FileNotFoundError):
            report_utils.check_report_for_errors(html_file)

    def test_empty_dataframe_error(self, tmp_path):
        """Test detection of empty dataframe error."""
        html_file = tmp_path / "empty_df_report.html"
        html_file.write_text(
            """
            <html>
                <body>
                    <p>The given query resulted in an empty dataframe</p>
                </body>
            </html>
            """
        )

        errors = report_utils.check_report_for_errors(html_file)
        assert len(errors) > 0
        assert any("Empty dataframe error" in e for e in errors)

    def test_zip_input_scans_inner_html(self, tmp_path):
        """A `.zip` input is unpacked and its inner HTML is scanned."""
        import zipfile

        html_content = "<html><body>ValueError: bad input</body></html>"
        zip_path = tmp_path / "report.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("report.html", html_content)
            zf.writestr("report_files/libs/plotly/plotly.js", "/* js */")

        errors = report_utils.check_report_for_errors(zip_path)
        assert any("ValueError exception" in e for e in errors)

    def test_zip_input_without_html_raises(self, tmp_path):
        """A zip with no HTML member raises IOError."""
        import zipfile

        zip_path = tmp_path / "no_html.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.txt", "hello")

        with pytest.raises(IOError, match="No HTML file"):
            report_utils.check_report_for_errors(zip_path)


class TestGetCmdOutput:
    """Tests for the _get_cmd_output helper."""

    def test_returns_lines_on_success(self):
        lines = report_utils._get_cmd_output(["echo", "hello world"])
        assert lines[0] == "hello world"

    def test_raises_on_missing_executable(self):
        with pytest.raises(FileNotFoundError, match="Command failed"):
            report_utils._get_cmd_output(["false"])


class TestGetQuartoAndPandoc:
    """Tests for get_quarto_with_version / get_pandoc_with_version."""

    def test_quarto_missing_raises(self, monkeypatch):
        monkeypatch.setattr(report_utils._quarto.shutil, "which", lambda _name: None)
        with pytest.raises(FileNotFoundError, match="Quarto executable not found"):
            report_utils.get_quarto_with_version()

    def test_pandoc_missing_raises(self, monkeypatch):
        monkeypatch.setattr(report_utils._quarto.shutil, "which", lambda _name: None)
        with pytest.raises(FileNotFoundError, match="Pandoc executable not found"):
            report_utils.get_pandoc_with_version()

    def test_quarto_success(self, monkeypatch, tmp_path):
        fake_exe = tmp_path / "quarto"
        fake_exe.write_text("")
        monkeypatch.setattr(report_utils._quarto.shutil, "which", lambda _name: str(fake_exe))
        monkeypatch.setattr(
            report_utils._quarto,
            "_get_cmd_output",
            lambda _args: ["quarto 1.4.550"],
        )
        path, version = report_utils.get_quarto_with_version()
        assert path == fake_exe
        assert version == "1.4.550"

    def test_pandoc_success(self, monkeypatch, tmp_path):
        fake_exe = tmp_path / "pandoc"
        fake_exe.write_text("")
        monkeypatch.setattr(report_utils._quarto.shutil, "which", lambda _name: str(fake_exe))
        monkeypatch.setattr(
            report_utils._quarto,
            "_get_cmd_output",
            lambda _args: ["pandoc 3.1.2"],
        )
        path, version = report_utils.get_pandoc_with_version()
        assert path == fake_exe
        assert version == "3.1.2"


class TestQuartoCallouts:
    """Tests for the quarto_print / callout helper functions.

    These wrappers render Markdown via IPython.display; we patch the display
    function and assert the rendered text contains the expected callout
    structure and content.
    """

    @pytest.fixture
    def captured(self, monkeypatch):
        captured: list[str] = []

        def _fake_display(md):
            captured.append(getattr(md, "data", str(md)))

        # Patch the symbol resolved inside the function via lazy import.
        import IPython.display as ipd

        monkeypatch.setattr(ipd, "display", _fake_display)
        return captured

    def test_quarto_print(self, captured):
        report_utils.quarto_print("hello **world**")
        assert captured == ["hello **world**"]

    def test_callout_info(self, captured):
        report_utils.quarto_callout_info("be aware")
        assert "{.callout-note}" in captured[0]
        assert "be aware" in captured[0]

    def test_callout_important(self, captured):
        report_utils.quarto_callout_important("read me")
        assert "{.callout-important}" in captured[0]
        assert "read me" in captured[0]

    def test_plot_exception(self, captured):
        try:
            raise ValueError("bad plot input")
        except ValueError as e:
            report_utils.quarto_plot_exception("Foo", e)
        rendered = captured[0]
        assert "Error rendering Foo plot" in rendered
        assert "bad plot input" in rendered
        assert 'collapse="true"' in rendered

    def test_no_prediction_data_warning(self, captured):
        report_utils.quarto_callout_no_prediction_data_warning("extra info")
        assert "Prediction Data is not available" in captured[0]
        assert "extra info" in captured[0]

    def test_no_predictor_data_warning(self, captured):
        report_utils.quarto_callout_no_predictor_data_warning()
        assert "Predictor Data is not available" in captured[0]


# ---------------------------------------------------------------------------
# _inline_css tests
# ---------------------------------------------------------------------------


class TestInlineCss:
    def test_relative_link_is_inlined(self, tmp_path):
        css = "body { color: red; }"
        css_file = tmp_path / "style.css"
        css_file.write_text(css, encoding="utf-8")

        html = '<html><head><link rel="stylesheet" href="style.css"></head><body></body></html>'
        html_file = tmp_path / "report.html"
        html_file.write_text(html, encoding="utf-8")

        n = report_utils._inline_css(html_file, tmp_path)

        result = html_file.read_text(encoding="utf-8")
        assert n == 1
        assert css in result
        assert "<style>" in result
        assert 'href="style.css"' not in result

    def test_href_before_rel_is_inlined(self, tmp_path):
        css = "h1 { font-size: 2em; }"
        (tmp_path / "theme.css").write_text(css, encoding="utf-8")

        html = '<html><head><link href="theme.css" rel="stylesheet" type="text/css"></head></html>'
        html_file = tmp_path / "report.html"
        html_file.write_text(html, encoding="utf-8")

        n = report_utils._inline_css(html_file, tmp_path)

        result = html_file.read_text(encoding="utf-8")
        assert n == 1
        assert css in result
        assert "<style>" in result

    def test_absolute_url_is_left_alone(self, tmp_path):
        html = '<html><head><link rel="stylesheet" href="https://cdn.example.com/bootstrap.css"></head></html>'
        html_file = tmp_path / "report.html"
        html_file.write_text(html, encoding="utf-8")

        n = report_utils._inline_css(html_file, tmp_path)

        result = html_file.read_text(encoding="utf-8")
        assert n == 0
        assert 'href="https://cdn.example.com/bootstrap.css"' in result

    def test_protocol_relative_url_is_left_alone(self, tmp_path):
        html = '<html><head><link rel="stylesheet" href="//cdn.example.com/bootstrap.css"></head></html>'
        html_file = tmp_path / "report.html"
        html_file.write_text(html, encoding="utf-8")

        n = report_utils._inline_css(html_file, tmp_path)
        assert n == 0

    def test_missing_file_logs_warning_and_leaves_tag(self, tmp_path, caplog):
        import logging

        html = '<html><head><link rel="stylesheet" href="missing.css"></head></html>'
        html_file = tmp_path / "report.html"
        html_file.write_text(html, encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            n = report_utils._inline_css(html_file, tmp_path)

        result = html_file.read_text(encoding="utf-8")
        assert n == 0
        assert 'href="missing.css"' in result
        assert "missing.css" in caplog.text

    def test_multiple_links_inlined(self, tmp_path):
        (tmp_path / "a.css").write_text("a { color: blue; }", encoding="utf-8")
        (tmp_path / "b.css").write_text("b { font-weight: bold; }", encoding="utf-8")

        html = '<html><head><link rel="stylesheet" href="a.css"><link rel="stylesheet" href="b.css"></head></html>'
        html_file = tmp_path / "report.html"
        html_file.write_text(html, encoding="utf-8")

        n = report_utils._inline_css(html_file, tmp_path)

        result = html_file.read_text(encoding="utf-8")
        assert n == 2
        assert "color: blue" in result
        assert "font-weight: bold" in result

    def test_subdirectory_href_resolved(self, tmp_path):
        libs = tmp_path / "report_files" / "libs"
        libs.mkdir(parents=True)
        css = "pre { background: #eee; }"
        (libs / "theme.css").write_text(css, encoding="utf-8")

        html = '<html><head><link rel="stylesheet" href="report_files/libs/theme.css"></head></html>'
        html_file = tmp_path / "report.html"
        html_file.write_text(html, encoding="utf-8")

        n = report_utils._inline_css(html_file, tmp_path)

        result = html_file.read_text(encoding="utf-8")
        assert n == 1
        assert css in result
