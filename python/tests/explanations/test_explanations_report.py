"""Tests for GlobalExplanations report generation."""

from pathlib import Path
from unittest.mock import patch

import pytest

from pdstools.explanations import Explanations

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "explanations" / "aggregated_data"


@pytest.fixture(scope="module")
def explanations():
    return Explanations.from_aggregates(
        base_path=DATA_DIR,
        model_name="AdaptiveBoostCT",
    )


def _configure_report_output(explanations: Explanations, tmp_path: Path) -> tuple[Path, Path]:
    aggregate_dir = tmp_path / "aggregated_data"
    aggregate_dir.mkdir()
    for filename in ("BY_CONTEXT.parquet", "OVERVIEW.parquet"):
        (aggregate_dir / filename).write_bytes((DATA_DIR / filename).read_bytes())

    explanations.data_folderpath = aggregate_dir
    output_dir = tmp_path / "reports"
    return aggregate_dir, output_dir


def test_GenerateExplanationsReport(explanations: Explanations, tmp_path):
    """generate() creates the batch artifacts before invoking Quarto."""
    aggregate_dir, output_dir = _configure_report_output(explanations, tmp_path)

    with patch("pdstools.explanations.Reports.run_quarto", return_value=0) as mock_run_quarto:
        explanations.report.generate(top_n=5, top_k=3, zip_output=False, output_dir=output_dir)

    assert (aggregate_dir / "unique_contexts.json").exists()
    assert {path.name for path in (aggregate_dir / "batches").glob("BATCH_*.parquet")} == {"BATCH_0.parquet"}
    mock_run_quarto.assert_called_once()


def test_GenerateExplanationsReport_Zipped(explanations: Explanations, tmp_path):
    """zip_output=True delegates to generate_zipped_report after Quarto succeeds."""
    aggregate_dir, output_dir = _configure_report_output(explanations, tmp_path)

    with (
        patch("pdstools.explanations.Reports.run_quarto", return_value=0),
        patch("pdstools.explanations.Reports.generate_zipped_report") as mock_generate_zipped_report,
    ):
        explanations.report.generate(
            report_filename="explanations_report.zip",
            top_n=5,
            top_k=3,
            zip_output=True,
            output_dir=output_dir,
        )

    mock_generate_zipped_report.assert_called_once_with(
        "explanations_report.zip",
        Path(output_dir) / "_site",
    )
