"""
Testing the functionality of the adm Reports module
"""

import pytest
from pdstools.utils.report_utils import get_output_filename


def test_get_output_filename():
    """Test the get_output_filename function"""
    # Test with name and model_id for ModelReport
    filename = get_output_filename(
        name="test_report",
        report_type="ModelReport",
        model_id="model1",
        output_type="html",
    )
    assert filename == "ModelReport_test_report_model1.html"

    # Test without name for ModelReport
    filename = get_output_filename(
        name=None, report_type="ModelReport", model_id="model1", output_type="html"
    )
    assert filename == "ModelReport_model1.html"

    # Test with name for HealthCheck
    filename = get_output_filename(
        name="test_report", report_type="HealthCheck", model_id=None, output_type="html"
    )
    assert filename == "HealthCheck_test_report.html"

    # Test without name for HealthCheck
    filename = get_output_filename(
        name=None, report_type="HealthCheck", model_id=None, output_type="html"
    )
    assert filename == "HealthCheck.html"

    # Test with spaces in name
    filename = get_output_filename(
        name="test report", report_type="HealthCheck", model_id=None, output_type="html"
    )
    assert filename == "HealthCheck_test_report.html"


@pytest.mark.skip(reason="Requires mocking yaml.dump")
def test_write_params_files(tmp_path):
    """Test the _write_params_files method"""
    # Skip this test for now since it requires mocking yaml.dump
    pass


@pytest.mark.skip(reason="Requires mocking subprocess")
def test_run_quarto_mock():
    """Test the run_quarto method"""
    # Skip this test for now since it requires mocking subprocess
    pass


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
                    verbose=True  # Enable verbose for better debugging
                )
            except Exception as e:
                pytest.fail(f"Health check generation failed: {e}")
            
            # Robust path checking
            if output_path is None:
                pytest.fail("Health check returned None instead of a Path")
            
            if not isinstance(output_path, Path):
                pytest.fail(f"Health check returned {type(output_path)} instead of Path")
            
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
                html_content = output_path.read_text(encoding='utf-8')
                assert "<html" in html_content, "Generated file is not valid HTML"
                assert "ADM Models across all Channels" in html_content, "Report content missing expected text"
            except UnicodeDecodeError as e:
                pytest.fail(f"Could not read generated HTML file: {e}")
    
    except Exception as e:
        pytest.fail(f"Test setup failed: {e}")


def test_html_deduplication():
    """Test HTML script deduplication functionality."""
    from pdstools.adm.Reports import Reports
    
    # Test with duplicate large scripts
    script1 = "var lib1 = function(){ console.log('lib1'); " + "x" * 2000 + "};"
    script2 = "var lib2 = function(){ console.log('lib2'); " + "y" * 2000 + "};"
    
    html_with_dupes = f"""<html><body>
    <script>{script1}</script>
    <script>{script1}</script>
    <script>{script2}</script>
    <div>Content</div></body></html>"""
    
    result = Reports._deduplicate_html_resources(html_with_dupes)
    
    assert result.count('<script>') == 2
    assert "Duplicate script removed" in result
    assert "Content" in result
    
    # Edge cases
    assert Reports._deduplicate_html_resources("") == ""
    assert Reports._deduplicate_html_resources("<html><script>broken") == "<html><script>broken"
