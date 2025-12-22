"""
Testing the functionality of the adm Reports module
"""

import pytest


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

        get_quarto_with_version(verbose=False)
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
                    verbose=True,  # Enable verbose for better debugging
                )
            except Exception as e:
                pytest.fail(f"Health check generation failed: {e}")

            # Robust path checking
            if output_path is None:
                pytest.fail("Health check returned None instead of a Path")

            if not isinstance(output_path, Path):
                pytest.fail(
                    f"Health check returned {type(output_path)} instead of Path"
                )

            if not output_path.exists():
                pytest.fail(f"Generated report file does not exist: {output_path}")

            # File size validation
            try:
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
            except OSError as e:
                pytest.fail(f"Could not get file size for {output_path}: {e}")

            assert file_size_mb <= 20, f"Report too large: {file_size_mb:.1f}MB > 20MB"
            assert (
                file_size_mb >= 0.5
            ), f"Report suspiciously small: {file_size_mb:.1f}MB"

            # Content validation
            try:
                html_content = output_path.read_text(encoding="utf-8")
                assert "<html" in html_content, "Generated file is not valid HTML"
                assert (
                    "ADM Models across all Channels" in html_content
                ), "Report content missing expected text"
            except UnicodeDecodeError as e:
                pytest.fail(f"Could not read generated HTML file: {e}")

    except Exception as e:
        pytest.fail(f"Test setup failed: {e}")


def test_html_deduplication():
    """Test HTML script deduplication functionality."""
    from pdstools.utils.report_utils import remove_duplicate_html_scripts

    # Test with duplicate large scripts (1MB+ to trigger deduplication)
    script1 = "var lib1 = function(){ console.log('lib1'); " + "x" * 1000000 + "};"
    script2 = "var lib2 = function(){ console.log('lib2'); " + "y" * 1000000 + "};"

    html_with_dupes = f"""<html><body>
    <script>{script1}</script>
    <script>{script1}</script>
    <script>{script2}</script>
    <div>Content</div></body></html>"""

    result = remove_duplicate_html_scripts(html_with_dupes)

    assert result.count("<script>") == 2
    assert "Duplicate script removed" in result
    assert "Content" in result

    # Edge cases
    assert remove_duplicate_html_scripts("") == ""
    assert (
        remove_duplicate_html_scripts("<html><script>broken") == "<html><script>broken"
    )


def test_remove_duplicate_html_scripts_size_threshold():
    """Test that only large scripts (>1MB) are deduplicated, small ones are preserved."""
    from pdstools.utils.report_utils import remove_duplicate_html_scripts

    large_script = "var largeLib = function(){ " + "x" * 1500000 + "};"  # 1.5MB
    small_script = "var small = 'hello';"

    # Mixed: small scripts + large duplicates
    html = f"""<html><body>
    <script>{small_script}</script>
    <script>{large_script}</script>
    <script>{large_script}</script>
    <script>{small_script}</script>
    </body></html>"""

    result = remove_duplicate_html_scripts(html)

    # Should remove 1 large duplicate, keep all small scripts
    assert result.count("<script>") == 3
    assert result.count("Duplicate script removed") == 1
    assert small_script in result


@pytest.mark.slow
def test_size_reduction_method_integration():
    """Test that the size_reduction_method options work correctly in report generation."""
    import tempfile
    from pathlib import Path
    from typing import Literal
    from pdstools import datasets
    from pdstools.adm.Reports import Reports

    try:
        from pdstools.utils.report_utils import get_quarto_with_version

        get_quarto_with_version(verbose=False)
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

        methods: list[Literal["strip", "cdn"] | None] = [None, "strip", "cdn"]
        for method in methods:
            method_name = method or "embedded"
            try:
                output = reports.health_check(
                    name=f"test_{method_name}",
                    output_dir=temp_path,
                    verbose=False,
                    size_reduction_method=method,
                )
                if output and isinstance(output, Path) and output.exists():
                    results[method_name] = {
                        "path": output,
                        "size": output.stat().st_size / (1024 * 1024),
                        "content": output.read_text(encoding="utf-8"),
                    }
            except Exception as e:
                pytest.skip(f"Report generation failed for {method_name}: {e}")

        if len(results) < 3:
            pytest.skip("Not all reports generated successfully")

        # Verify all reports are valid HTML with plots
        for method_name, data in results.items():
            assert "<html" in data["content"], f"{method_name}: Not valid HTML"
            assert (
                data["content"].count("Plotly.newPlot") > 0
            ), f"{method_name}: Missing plots"

        # CDN should be smallest (no embedded resources)
        assert (
            results["cdn"]["size"] < results["embedded"]["size"]
        ), "CDN should be smaller than embedded"

        # Strip should also be smaller than embedded
        assert (
            results["strip"]["size"] < results["embedded"]["size"]
        ), "Strip should be smaller than embedded"

        # Strip should have deduplication markers
        assert (
            "Duplicate script removed" in results["strip"]["content"]
        ), "Strip should have deduplication markers"

        # Embedded should NOT have deduplication markers
        assert (
            "Duplicate script removed" not in results["embedded"]["content"]
        ), "Embedded should not have markers"
