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
    
    assert result.count('<script>') == 2
    assert "Duplicate script removed" in result
    assert "Content" in result
    
    # Edge cases
    assert remove_duplicate_html_scripts("") == ""
    assert remove_duplicate_html_scripts("<html><script>broken") == "<html><script>broken"


def test_remove_duplicate_html_scripts_comprehensive():
    """Comprehensive test for remove_duplicate_html_scripts functionality."""
    from pdstools.utils.report_utils import remove_duplicate_html_scripts
    
    # Test 1: Large scripts should be deduplicated significantly
    large_script = "var largeLib = function(){ " + "x" * 1500000 + "};"  # 1.5MB
    html_large_dupes = f"""<html><body>
    <script>{large_script}</script>
    <script>{large_script}</script>
    <script>{large_script}</script>
    <div>Content</div>
    </body></html>"""
    
    result_large = remove_duplicate_html_scripts(html_large_dupes)
    original_size = len(html_large_dupes)
    result_size = len(result_large)
    
    # Should remove 2 of the 3 large scripts (keep 1, remove 2)
    assert result_large.count('<script>') == 1, "Should keep only one large script"
    assert result_large.count('Duplicate script removed') == 2, "Should remove 2 duplicates"
    assert "Content" in result_large, "Should preserve non-script content"
    
    # Should achieve significant size reduction (>50%)
    size_reduction = (original_size - result_size) / original_size
    assert size_reduction > 0.5, f"Should achieve >50% size reduction, got {size_reduction:.1%}"
    
    # Test 2: Small scripts should NOT be affected
    small_script1 = "var small1 = 'hello';"
    small_script2 = "var small2 = 'world';"
    html_small = f"""<html><body>
    <script>{small_script1}</script>
    <script>{small_script1}</script>
    <script>{small_script2}</script>
    <div>Content</div>
    </body></html>"""
    
    result_small = remove_duplicate_html_scripts(html_small)
    
    # Small scripts should be left untouched
    assert result_small.count('<script>') == 3, "Should not remove small scripts"
    assert 'Duplicate script removed' not in result_small, "Should not mark any small scripts as removed"
    assert len(result_small) == len(html_small), "Size should be unchanged for small scripts"
    
    # Test 3: Mixed scenario - only large duplicates removed
    mixed_html = f"""<html><body>
    <script>{small_script1}</script>
    <script>{large_script}</script>
    <script>{large_script}</script>
    <script>{small_script2}</script>
    <div>Content</div>
    </body></html>"""
    
    result_mixed = remove_duplicate_html_scripts(mixed_html)
    
    # Should remove 1 large duplicate but leave all small scripts
    assert result_mixed.count('<script>') == 3, "Should have 2 small + 1 large script"
    assert result_mixed.count('Duplicate script removed') == 1, "Should remove 1 large duplicate"
    assert small_script1 in result_mixed and small_script2 in result_mixed, "Should preserve small scripts"
    
    # Test 4: Threshold behavior - scripts just under 1MB are preserved
    wrapper_text = "var justUnder = function(){ };"
    # Calculate padding to make total script size just under 1MB (1,000,000 bytes)
    padding_size = 1000000 - len(wrapper_text) - 10  # Leave small buffer
    just_under_threshold = "var justUnder = function(){ " + "x" * padding_size + "};"
    html_threshold = f"""<html><body>
    <script>{just_under_threshold}</script>
    <script>{just_under_threshold}</script>
    <div>Content</div>
    </body></html>"""
    
    # Verify the script is actually under 1MB
    assert len(just_under_threshold) < 1000000, f"Test script should be <1MB, got {len(just_under_threshold)} bytes"
    
    result_threshold = remove_duplicate_html_scripts(html_threshold)
    
    # Scripts just under 1MB should not be deduplicated
    assert result_threshold.count('<script>') == 2, "Scripts under 1MB should not be deduplicated"
    assert 'Duplicate script removed' not in result_threshold, "Should not remove sub-1MB scripts"


@pytest.mark.slow
def test_remove_duplicate_html_scripts_flag_integration():
    """Test that the remove_duplicate_html_scripts flag works correctly in report generation."""
    import tempfile
    from pathlib import Path
    from pdstools import datasets
    from pdstools.adm.Reports import Reports
    
    # Skip if Quarto not available
    try:
        from pdstools.utils.report_utils import get_quarto_with_version
        get_quarto_with_version(verbose=False)
    except (FileNotFoundError, Exception) as e:
        pytest.skip(f"Quarto not available: {e}")
    
    datamart = datasets.cdh_sample()
    reports = Reports(datamart)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test with deduplication enabled (default)
        output_with_dedup = reports.health_check(
            name="test_with_dedup",
            output_dir=temp_path,
            verbose=False,
            remove_duplicate_html_scripts=True
        )
        
        # Test with deduplication disabled  
        output_without_dedup = reports.health_check(
            name="test_without_dedup", 
            output_dir=temp_path,
            verbose=False,
            remove_duplicate_html_scripts=False
        )
        
        # Verify both files exist
        assert output_with_dedup.exists(), "Report with deduplication should exist"
        assert output_without_dedup.exists(), "Report without deduplication should exist"
        
        # Check file sizes
        size_with_dedup = output_with_dedup.stat().st_size / (1024 * 1024)  # MB
        size_without_dedup = output_without_dedup.stat().st_size / (1024 * 1024)  # MB
        
        # File with deduplication should be significantly smaller
        size_reduction = (size_without_dedup - size_with_dedup) / size_without_dedup
        assert size_reduction > 0.8, f"Should achieve >80% size reduction, got {size_reduction:.1%}"
        assert size_with_dedup < 20, f"Deduplicated file should be <20MB, got {size_with_dedup:.1f}MB"
        
        # Verify both files have plots (functionality preserved)
        html_with_dedup = output_with_dedup.read_text(encoding='utf-8')
        html_without_dedup = output_without_dedup.read_text(encoding='utf-8')
        
        plots_with_dedup = html_with_dedup.count('Plotly.newPlot')
        plots_without_dedup = html_without_dedup.count('Plotly.newPlot')
        
        assert plots_with_dedup > 0, "Deduplicated file should still have plots"
        assert plots_with_dedup == plots_without_dedup, "Plot count should be identical"
        
        # Verify duplicate removal markers only in deduplicated version
        assert 'Duplicate script removed' in html_with_dedup, "Should have removal markers"
        assert 'Duplicate script removed' not in html_without_dedup, "Should not have markers without dedup"
