"""Tests for GlobalExplanations report generation."""

from pathlib import Path
from unittest.mock import patch

import pytest

from pdstools.explanations import Explanations

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "explanations" / "aggregated_data"


@pytest.fixture(scope="module")
def explanations():
    return Explanations.from_aggregates(
        aggregated_data_dir=DATA_DIR,
        model_name="AdaptiveBoostCT",
    )


def _configure_report_output(explanations: Explanations, tmp_path: Path) -> Path:
    aggregate_dir = tmp_path / "aggregated_data"
    aggregate_dir.mkdir()
    for filename in ("BY_CONTEXT.parquet", "OVERALL.parquet"):
        (aggregate_dir / filename).write_bytes((DATA_DIR / filename).read_bytes())

    explanations.aggregate.data_folderpath = aggregate_dir
    explanations.aggregate.context_operations.unique_contexts_file = aggregate_dir / "unique_contexts.json"
    explanations.report.aggregate_folder = aggregate_dir
    explanations.report.report_folderpath = str(tmp_path / "reports")
    explanations.report.report_output_dir = str(Path(explanations.report.report_folderpath) / "_site")
    explanations.report.params_file = str(Path(explanations.report.report_folderpath) / "scripts" / "params.yml")
    return aggregate_dir


def test_GenerateExplanationsReport(explanations: Explanations, tmp_path):
    """generate() creates the batch artifacts before invoking Quarto."""
    aggregate_dir = _configure_report_output(explanations, tmp_path)

    with patch("pdstools.explanations.Reports.run_quarto", return_value=0) as mock_run_quarto:
        explanations.report.generate(top_n=5, top_k=3, zip_output=False)

    assert (aggregate_dir / "unique_contexts.json").exists()
    assert list((aggregate_dir / "batches").glob("BATCH_*.parquet"))
    mock_run_quarto.assert_called_once()


def test_GenerateExplanationsReport_Zipped(explanations: Explanations, tmp_path):
    """zip_output=True delegates to generate_zipped_report after Quarto succeeds."""
    _configure_report_output(explanations, tmp_path)

    with (
        patch("pdstools.explanations.Reports.run_quarto", return_value=0),
        patch("pdstools.explanations.Reports.generate_zipped_report") as mock_generate_zipped_report,
    ):
        explanations.report.generate(
            report_filename="explanations_report.zip",
            top_n=5,
            top_k=3,
            zip_output=True,
        )

    mock_generate_zipped_report.assert_called_once_with(
        "explanations_report.zip",
        explanations.report.report_output_dir,
    )
