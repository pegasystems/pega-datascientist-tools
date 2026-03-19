"""Tests for GlobalExplanations Quarto report generation.

These tests require Quarto and Pandoc to be installed and available
on the system PATH. They are run in the 'Quarto report tests' CI
workflow alongside other quarto report generation tests.
"""

import shutil
from datetime import datetime
from pathlib import Path

import pytest

from pdstools.explanations import Explanations

basePath = Path(__file__).parent.parent.parent


def clean_up(root_dir):
    _root_dir = Path(f"{basePath}/{root_dir}")
    if _root_dir.exists():
        for file in _root_dir.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                # Remove subdirectories recursively
                shutil.rmtree(file)
        _root_dir.rmdir()


@pytest.fixture(scope="module")
def explanations():
    """Create an Explanations instance with sample data.

    The Explanations constructor automatically runs the preprocessing
    and aggregation pipeline, producing the aggregated parquet files
    and unique_contexts.json needed for report generation.
    """
    exp = Explanations(
        data_folder=f"{basePath}/data/explanations",
        model_name="AdaptiveBoostCT",
        from_date=datetime(2025, 3, 28),
        to_date=datetime(2025, 3, 28),
    )
    yield exp

    # Cleanup generated .tmp directory
    # cleanup .tmp folder
    clean_up(exp.root_dir)


def test_GenerateExplanationsReport(explanations: Explanations):
    """Test that the GlobalExplanations Quarto website renders successfully."""
    explanations.report.generate(
        top_n=5,
        top_k=3,
        zip_output=False,
    )

    output_dir = Path(explanations.report.report_output_dir)
    assert output_dir.exists(), f"Report output directory not found: {output_dir}"

    html_files = list(output_dir.rglob("*.html"))
    assert len(html_files) > 0, "No HTML files generated in report output"

    # The website should produce at least the getting-started and overview pages
    html_names = {f.stem for f in html_files}
    assert "getting-started" in html_names, f"Expected 'getting-started.html' in output, got: {html_names}"
    assert "overview" in html_names, f"Expected 'overview.html' in output, got: {html_names}"


def test_GenerateExplanationsReport_Zipped(explanations: Explanations):
    """Test that the GlobalExplanations report can be zipped."""
    zip_filename = "explanations_report.zip"
    explanations.report.generate(
        report_filename=zip_filename,
        top_n=5,
        top_k=3,
        zip_output=True,
    )

    # generate_zipped_report creates the zip in the current working directory
    zip_path = Path(zip_filename)
    assert zip_path.exists(), f"Zipped report not found: {zip_path}"
    assert zip_path.stat().st_size > 0, "Zipped report is empty"

    # Cleanup zip file
    zip_path.unlink()
