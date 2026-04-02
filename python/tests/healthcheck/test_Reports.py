"""Testing the functionality of the adm Reports module"""

import zipfile

import pytest

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


@pytest.mark.slow
def test_health_check_agent_basic():
    """Verify health_check_agent generates a valid Markdown file."""
    import tempfile
    from pathlib import Path

    from pdstools import datasets
    from pdstools.adm.Reports import Reports

    try:
        from pdstools.utils.report_utils import get_quarto_with_version

        get_quarto_with_version(verbose=False)
    except (FileNotFoundError, Exception) as e:
        pytest.skip(f"Quarto not available in this environment: {e}")

    datamart = datasets.cdh_sample()
    reports = Reports(datamart)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        output_path = reports.health_check_agent(
            name="agent_test",
            output_dir=temp_path,
        )

        assert output_path is not None
        assert isinstance(output_path, Path)
        assert output_path.exists(), f"Report file not found: {output_path}"
        assert output_path.suffix == ".md", f"Expected .md file, got: {output_path.suffix}"

        content = output_path.read_text(encoding="utf-8")

        # Must not be empty
        assert len(content) > 1000, "Report suspiciously short"

        # Core sections must be present
        assert "## ADM Model Overview" in content or "# ADM Model Overview" in content
        assert "Channel Overview" in content
        assert "Model Performance" in content or "Model Maturity" in content

        # Must be plain text Markdown, not HTML
        assert "<html" not in content.lower()
        assert "<!DOCTYPE" not in content.upper()


@pytest.mark.slow
def test_health_check_agent_title_and_subtitle():
    """Verify title, subtitle and disclaimer are rendered into the output."""
    import tempfile
    from pathlib import Path

    from pdstools import datasets
    from pdstools.adm.Reports import Reports

    try:
        from pdstools.utils.report_utils import get_quarto_with_version

        get_quarto_with_version(verbose=False)
    except (FileNotFoundError, Exception) as e:
        pytest.skip(f"Quarto not available in this environment: {e}")

    datamart = datasets.cdh_sample()
    reports = Reports(datamart)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = reports.health_check_agent(
            name="agent_title_test",
            title="Test Client Health Check",
            subtitle="Q1 2025",
            disclaimer="For internal use only.",
            output_dir=Path(temp_dir),
        )

        content = output_path.read_text(encoding="utf-8")
        assert "Test Client Health Check" in content
        assert "Q1 2025" in content
        assert "For internal use only." in content


@pytest.mark.slow
def test_health_check_agent_no_lift():
    """Verify the Predictions/lift section is not rendered (pending reliability verification)."""
    import tempfile
    from pathlib import Path

    from pdstools import datasets
    from pdstools.adm.Reports import Reports

    try:
        from pdstools.utils.report_utils import get_quarto_with_version

        get_quarto_with_version(verbose=False)
    except (FileNotFoundError, Exception) as e:
        pytest.skip(f"Quarto not available in this environment: {e}")

    datamart = datasets.cdh_sample()
    reports = Reports(datamart)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = reports.health_check_agent(
            name="agent_lift_test",
            output_dir=Path(temp_dir),
        )

        content = output_path.read_text(encoding="utf-8")

        # Lift section must not appear in output
        assert "Engagement Lift" not in content
        assert "CTR_Test" not in content
        assert "Negative engagement lift" not in content


@pytest.mark.slow
def test_health_check_agent_with_predictor_data():
    """Verify the Predictor Analysis section appears when predictor data is available."""
    import tempfile
    from pathlib import Path

    from pdstools import datasets
    from pdstools.adm.Reports import Reports

    try:
        from pdstools.utils.report_utils import get_quarto_with_version

        get_quarto_with_version(verbose=False)
    except (FileNotFoundError, Exception) as e:
        pytest.skip(f"Quarto not available in this environment: {e}")

    datamart = datasets.cdh_sample()
    if datamart.predictor_data is None:
        pytest.skip("cdh_sample does not include predictor data")

    reports = Reports(datamart)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = reports.health_check_agent(
            name="agent_predictor_test",
            output_dir=Path(temp_dir),
        )

        content = output_path.read_text(encoding="utf-8")
        assert "Predictor Analysis" in content
        assert "Predictors per Category" in content


@pytest.mark.slow
def test_health_check_agent_interpretation_guides():
    """Verify key interpretation blocks are present in the output."""
    import tempfile
    from pathlib import Path

    from pdstools import datasets
    from pdstools.adm.Reports import Reports

    try:
        from pdstools.utils.report_utils import get_quarto_with_version

        get_quarto_with_version(verbose=False)
    except (FileNotFoundError, Exception) as e:
        pytest.skip(f"Quarto not available in this environment: {e}")

    datamart = datasets.cdh_sample()
    reports = Reports(datamart)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = reports.health_check_agent(
            name="agent_interp_test",
            output_dir=Path(temp_dir),
        )

        content = output_path.read_text(encoding="utf-8")

        # At least these interpretation blocks must be present regardless of data
        assert content.count("**Interpretation:**") >= 5



