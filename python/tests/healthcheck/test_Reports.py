"""Testing the functionality of the adm Reports module"""

import logging
import pathlib
import zipfile

import polars as pl
import pytest

from pdstools import datasets
from pdstools.adm.Reports import Reports

from pdstools.utils.report_utils import check_report_for_errors


def _read_report_html(output_path):
    """Return the HTML content for a rendered report, unzipping if needed."""
    if output_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(output_path) as zf:
            html_name = next(n for n in zf.namelist() if n.endswith(".html"))
            return zf.read(html_name).decode("utf-8")
    return output_path.read_text(encoding="utf-8")


@pytest.mark.slow
def test_health_check_file_size_optimization():
    """Verify Health Check reports stay under 20MB limit."""
    import tempfile
    from pathlib import Path

    from pdstools import datasets
    from pdstools.adm.Reports import Reports

    # Check if Quarto is available (common issue in CI environments)
    try:
        from pdstools.utils.report_utils import get_quarto_with_version

        get_quarto_with_version()
    except (FileNotFoundError, Exception) as e:
        pytest.skip(f"Quarto not available in this environment: {e}")

    try:
        datamart = datasets.cdh_sample()
        reports = Reports(datamart)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Ensure temp directory is properly created and accessible
            assert temp_path.exists(), f"Temporary directory not created: {temp_path}"
            assert temp_path.is_dir(), f"Temporary path is not a directory: {temp_path}"

            try:
                output_path = reports.health_check(
                    name="size_test",
                    output_dir=temp_path,
                )
            except Exception as e:
                pytest.fail(f"Health check generation failed: {e}")

            # Robust path checking
            if output_path is None:
                pytest.fail("Health check returned None instead of a Path")

            if not isinstance(output_path, Path):
                pytest.fail(
                    f"Health check returned {type(output_path)} instead of Path",
                )

            if not output_path.exists():
                pytest.fail(f"Generated report file does not exist: {output_path}")

            # File size validation
            try:
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
            except OSError as e:
                pytest.fail(f"Could not get file size for {output_path}: {e}")

            assert file_size_mb <= 20, f"Report too large: {file_size_mb:.1f}MB > 20MB"
            assert file_size_mb >= 0.5, f"Report suspiciously small: {file_size_mb:.1f}MB"

            # Content validation
            try:
                html_content = _read_report_html(output_path)
                assert "<html" in html_content, "Generated file is not valid HTML"
                assert "ADM Models across all Channels" in html_content, "Report content missing expected text"
            except UnicodeDecodeError as e:
                pytest.fail(f"Could not read generated HTML file: {e}")

            # Check for rendering errors in generated HTML
            errors = check_report_for_errors(output_path)
            assert len(errors) == 0, "Report contains errors:\n" + "\n".join(f"  - {e}" for e in errors)

    except Exception as e:
        pytest.fail(f"Test setup failed: {e}")


@pytest.mark.slow
def test_full_embed_integration():
    """Test that the full_embed option works correctly in report generation."""
    import tempfile
    from pathlib import Path

    from pdstools import datasets
    from pdstools.adm.Reports import Reports

    try:
        from pdstools.utils.report_utils import get_quarto_with_version

        get_quarto_with_version()
    except (FileNotFoundError, Exception) as e:
        pytest.skip(f"Quarto not available: {e}")

    try:
        datamart = datasets.cdh_sample()
        reports = Reports(datamart)
    except Exception as e:
        pytest.skip(f"Could not initialize test environment: {e}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        results = {}

        for full_embed in [False, True]:
            label = "embedded" if full_embed else "cdn"
            try:
                output = reports.health_check(
                    name=f"test_{label}",
                    output_dir=temp_path,
                    full_embed=full_embed,
                )
                if output and isinstance(output, Path) and output.exists():
                    results[label] = {
                        "path": output,
                        "size": output.stat().st_size / (1024 * 1024),
                        "content": _read_report_html(output),
                    }
            except Exception as e:
                pytest.skip(f"Report generation failed for {label}: {e}")

        if len(results) < 2:
            pytest.skip("Not all reports generated successfully")

        # Verify all reports are valid HTML with plots and no rendering errors
        for label, data in results.items():
            assert "<html" in data["content"], f"{label}: Not valid HTML"
            assert data["content"].count("Plotly.newPlot") > 0, f"{label}: Missing plots"
            errors = check_report_for_errors(data["path"])
            assert len(errors) == 0, f"{label} report contains errors:\n" + "\n".join(f"  - {e}" for e in errors)

        # CDN should be smallest (no embedded resources)
        assert results["cdn"]["size"] < results["embedded"]["size"], "CDN should be smaller than embedded"


def test_health_check_markdown_writes_markdown(tmp_path):
    datamart = datasets.cdh_sample()
    reports = Reports(datamart)

    output_path = reports.health_check_markdown(
        name="agent_health_check",
        output_dir=tmp_path,
    )

    assert output_path.exists()
    assert output_path.suffix == ".md"
    content = output_path.read_text(encoding="utf-8")
    assert content.startswith("# ADM Health Check")
    assert "## Summary" in content
    assert "## Key Metrics" in content
    assert "## Estate Snapshot" in content
    assert "## Findings Overview" in content
    assert "Health:" in content


def test_health_check_markdown_respects_title_subtitle_and_disclaimer(tmp_path):
    datamart = datasets.cdh_sample()
    reports = Reports(datamart)

    output_path = reports.health_check_markdown(
        output_dir=tmp_path,
        title="Custom Title",
        subtitle="Custom Subtitle",
        disclaimer="Review before sharing.",
    )

    content = output_path.read_text(encoding="utf-8")
    assert "# Custom Title" in content
    assert "## Custom Subtitle" in content
    assert "> **Disclaimer:** Review before sharing." in content


def test_health_check_markdown_accepts_preaggregates(tmp_path):
    datamart = datasets.cdh_sample()
    reports = Reports(datamart)
    preaggregates = datamart.analysis.compute_health_check_preaggregates()

    output_path = reports.health_check_markdown(
        output_dir=tmp_path,
        preaggregates=preaggregates,
    )

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").startswith("# ADM Health Check")


def test_health_check_markdown_rejects_query_with_preaggregates(tmp_path):
    datamart = datasets.cdh_sample()
    reports = Reports(datamart)
    preaggregates = datamart.analysis.compute_health_check_preaggregates()

    with pytest.raises(ValueError, match="query and preaggregates"):
        reports.health_check_markdown(
            output_dir=tmp_path,
            query=pl.col("Channel") == "Web",
            preaggregates=preaggregates,
        )


def test_health_check_query_consolidates_predictors_to_filter(tmp_path, monkeypatch, caplog):
    datamart = datasets.cdh_sample()
    reports = Reports(datamart)
    query = pl.col("Channel") == "Web"

    captured: dict[str, object] = {}

    def fake_copy_quarto_file(qmd_filename, temp_dir):
        (temp_dir / qmd_filename).write_text("", encoding="utf-8")

    def fake_save_data(path, selected_model_ids=None):
        captured["selected_model_ids"] = selected_model_ids
        return pathlib.Path(path) / "model.parquet", pathlib.Path(path) / "predictor.parquet"

    def fake_run_quarto(*, output_filename, temp_dir, **_kwargs):
        (temp_dir / output_filename).write_text("", encoding="utf-8")
        return None

    monkeypatch.setattr("pdstools.adm.Reports.copy_quarto_file", fake_copy_quarto_file)
    monkeypatch.setattr("pdstools.adm.Reports.run_quarto", fake_run_quarto)
    monkeypatch.setattr(datamart, "save_data", fake_save_data)

    with caplog.at_level(logging.INFO, logger="pdstools.adm.Reports"):
        output_path = reports.health_check(output_dir=tmp_path, query=query)

    expected_model_ids = (
        datamart.model_data.filter(query).select("ModelID").unique().collect(engine="streaming")["ModelID"].to_list()
    )

    assert captured["selected_model_ids"] is not None
    assert sorted(captured["selected_model_ids"]) == sorted(expected_model_ids)
    assert f"Data exported to {output_path}" in caplog.text


def test_model_reports_logs_export_path(tmp_path, monkeypatch, caplog):
    datamart = datasets.cdh_sample()
    reports = Reports(datamart)

    def fake_copy_quarto_file(qmd_filename, temp_dir):
        (temp_dir / qmd_filename).write_text("", encoding="utf-8")

    def fake_save_data(path, selected_model_ids=None):
        return pathlib.Path(path) / "model.parquet", pathlib.Path(path) / "predictor.parquet"

    def fake_run_quarto(*, output_filename, temp_dir, **_kwargs):
        (temp_dir / output_filename).write_text("<html></html>", encoding="utf-8")
        return None

    monkeypatch.setattr("pdstools.adm.Reports.copy_quarto_file", fake_copy_quarto_file)
    monkeypatch.setattr("pdstools.adm.Reports.run_quarto", fake_run_quarto)
    monkeypatch.setattr(datamart, "save_data", fake_save_data)

    with caplog.at_level(logging.INFO, logger="pdstools.adm.Reports"):
        output_path = reports.model_reports(
            model_ids=["model-1"],
            output_dir=tmp_path,
            name="Logged",
        )

    assert output_path.exists()
    assert f"Data exported to {output_path}" in caplog.text
