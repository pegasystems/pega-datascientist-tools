"""Build the published ADM report examples from the shared synthetic sample."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from pdstools import demo_data
from pdstools.utils.report_utils import check_report_for_errors

logger = logging.getLogger(__name__)

NAIVE_BAYES_MODEL_ID = "05085857-c52c-5daa-b51b-0af576a038af"


def generate_example_reports(output_dir: Path) -> None:
    """Render standalone HTML examples for Sphinx to publish as static files.

    Parameters
    ----------
    output_dir : Path
        Directory for the generated Health Check and Naive Bayes Model Report.

    Raises
    ------
    ValueError
        If a report is not standalone HTML or contains rendering errors.
    """
    sample = demo_data()
    output_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="pdstools-docs-reports-") as work_dir:
        report_dir = Path(work_dir)
        logger.info("Generating the example ADM Health Check")
        health_check = sample.datamart.generate.health_check(
            prediction=sample.prediction,
            output_dir=report_dir,
            full_embed=False,
            title="ADM Health Check example",
            subtitle="All data is synthetic and generated from pdstools.demo_data().",
        )
        logger.info("Generating the example Naive Bayes Model Report")
        model_report = sample.datamart.generate.model_reports(
            NAIVE_BAYES_MODEL_ID,
            output_dir=report_dir,
            full_embed=False,
            title="Naive Bayes Model Report example",
            subtitle=("All data is synthetic. This report uses the Fixed Rate Loan model for Web."),
        )

        for report, filename in (
            (health_check, "health-check.html"),
            (model_report, "naive-bayes-model-report.html"),
        ):
            if report.suffix != ".html":
                raise ValueError(f"Expected standalone HTML for the documentation example, got {report.name}")
            errors = check_report_for_errors(report)
            if errors:
                raise ValueError(f"Rendering errors in {report.name}: {'; '.join(errors)}")
            shutil.copyfile(report, output_dir / filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_example_reports(args.output_dir)
