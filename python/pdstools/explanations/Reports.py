from __future__ import annotations

__all__ = ["Reports"]

import logging
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

import yaml

from ..utils.namespaces import LazyNamespace
from ..utils.report_utils import (
    copy_report_resources,
    generate_zipped_report,
    run_quarto,
)
from ._constants import (
    CONTRIBUTION_LABELS,
    ContributionType,
    validate_contribution_type,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = ".tmp/reports"

if TYPE_CHECKING:
    from .Explanations import Explanations


class Reports(LazyNamespace):
    """Reports."""

    dependencies: ClassVar[list[str]] = ["yaml"]
    dependency_group = "explanations"

    def __init__(self, explanations: "Explanations"):
        self.explanations = explanations
        super().__init__()

    def generate(
        self,
        report_filename: str = "explanations_report.zip",
        top_n: int = 20,
        top_k: int = 20,
        zip_output: bool = False,
        *,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        sort_by: ContributionType = "contribution_abs",
        display_by: ContributionType = "contribution",
    ):
        """Generate the explanations report.

        Parameters
        ----------
        report_filename : str
            Name of the output report file.
        top_n : int
            Number of top explanations to include.
        top_k : int
            Number of top features to include in explanations.
        zip_output : bool
            Whether to zip the output report.
            The filename will be used as the zip file name.
        output_dir : str | Path, keyword-only
            Directory the Quarto project is rendered into, default
            ``".tmp/reports"``. The rendered site lands in ``<output_dir>/_site``.
        sort_by : str, keyword-only
            Column to rank/select top predictors. Default: ``"contribution_abs"``.
        display_by : str, keyword-only
            Column to use for the report axis values. Default: ``"contribution"``.

        Notes
        -----
        Progress and diagnostic information is logged at DEBUG level.
        Enable debug logging to see detailed report generation steps.

        """
        report_folder = Path(output_dir)
        co = self.explanations.aggregates.context_operations
        contexts = co.create_unique_contexts_file()
        co.create_batch_parquet_files(contexts)

        sort_by = validate_contribution_type(sort_by)
        display_by = validate_contribution_type(display_by)

        report_folder.mkdir(parents=True, exist_ok=True)
        self._copy_report_resources(report_folder)

        self._set_params(
            report_folder / "scripts" / "params.yml",
            top_n=top_n,
            top_k=top_k,
            from_date=self.explanations.from_date.strftime("%Y-%m-%d"),
            to_date=self.explanations.to_date.strftime("%Y-%m-%d"),
            sort_by=sort_by,
            display_by=display_by,
        )

        return_code = run_quarto(temp_dir=report_folder, output_type=None)

        if return_code != 0:
            raise RuntimeError(f"Quarto command failed with return code {return_code}")

        if zip_output:
            generate_zipped_report(report_filename, report_folder / "_site")

    @staticmethod
    def _copy_report_resources(report_folder: Path):
        logger.debug("Copying report resources to %s", report_folder)
        copy_report_resources(
            resource_dict=[
                ("GlobalExplanations", str(report_folder)),
                ("assets", str(report_folder / "assets")),
            ],
        )

    def _set_params(
        self,
        params_file: Path,
        top_n: int = 20,
        top_k: int = 20,
        from_date: str = "",
        to_date: str = "",
        sort_by: ContributionType = "contribution_abs",
        display_by: ContributionType = "contribution",
    ):
        params: dict[str, str | int] = {
            "top_n": top_n,
            "top_k": top_k,
            "from_date": from_date,
            "to_date": to_date,
            "sort_by": sort_by,
            "sort_by_text": CONTRIBUTION_LABELS[sort_by][1],
            "display_by": display_by,
            "display_by_text": CONTRIBUTION_LABELS[display_by][1],
            "data_folder": str(self.explanations.data_folderpath),
        }

        logger.debug("Writing report parameters to %s", params_file)
        params_file.parent.mkdir(parents=True, exist_ok=True)
        with open(params_file, "w", encoding="utf-8") as file:
            yaml.safe_dump(params, file)
