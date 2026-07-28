from __future__ import annotations

__all__ = ["Reports"]

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

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
        full_embed: bool = False,
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
        full_embed : bool, default False
            When True, fully embed JavaScript libraries into the generated
            Quarto website. Defaults to False, which keeps the smaller
            CDN-backed output.
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
        # Resolved eagerly: the Quarto pre-render script runs with the report
        # folder as its cwd and resolves a relative data_folder against the
        # folder's *parent*, so a relative output_dir would be doubled up.
        report_folder = Path(output_dir).resolve()
        # The Quarto subprocess reads plain local files, so materialise the
        # frames (which may be backed by a URL, or built in memory) alongside
        # the context artifacts it needs. Mirrors ADMDatamart.save_data usage.
        data_folder = report_folder / "data"
        self.explanations.save_data(data_folder)

        self.explanations.aggregates.context_operations.write_batches(data_folder)

        sort_by = validate_contribution_type(sort_by)
        display_by = validate_contribution_type(display_by)

        report_folder.mkdir(parents=True, exist_ok=True)
        self._copy_report_resources(report_folder)
        self._set_full_embed_options(report_folder, full_embed=full_embed)

        self._set_params(
            report_folder / "scripts" / "params.yml",
            top_n=top_n,
            top_k=top_k,
            data_folder=data_folder,
            from_date=self.explanations.from_date.strftime("%Y-%m-%d"),
            to_date=self.explanations.to_date.strftime("%Y-%m-%d"),
            sort_by=sort_by,
            display_by=display_by,
            full_embed=full_embed,
        )

        return_code = run_quarto(temp_dir=report_folder, output_type=None, full_embed=full_embed)

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

    @staticmethod
    def _set_full_embed_options(report_folder: Path, *, full_embed: bool) -> None:
        quarto_config_path = report_folder / "_quarto.yml"
        if not quarto_config_path.exists():
            logger.debug("Quarto config not found at %s; skipping embed option update", quarto_config_path)
            return

        with open(quarto_config_path, encoding="utf-8") as file:
            quarto_config = yaml.safe_load(file) or {}

        html_format = quarto_config.setdefault("format", {}).setdefault("html", {})
        html_format["embed-resources"] = full_embed
        html_format["plotly-connected"] = full_embed

        with open(quarto_config_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(quarto_config, file, sort_keys=False)

    def _set_params(
        self,
        params_file: Path,
        data_folder: Path,
        top_n: int = 20,
        top_k: int = 20,
        from_date: str = "",
        to_date: str = "",
        sort_by: ContributionType = "contribution_abs",
        display_by: ContributionType = "contribution",
        full_embed: bool = False,
    ):
        params: dict[str, str | int | bool] = {
            "top_n": top_n,
            "top_k": top_k,
            "from_date": from_date,
            "to_date": to_date,
            "sort_by": sort_by,
            "sort_by_text": CONTRIBUTION_LABELS[sort_by][1],
            "display_by": display_by,
            "display_by_text": CONTRIBUTION_LABELS[display_by][1],
            "data_folder": str(data_folder),
            "full_embed": full_embed,
        }

        logger.debug("Writing report parameters to %s", params_file)
        params_file.parent.mkdir(parents=True, exist_ok=True)
        with open(params_file, "w", encoding="utf-8") as file:
            yaml.safe_dump(params, file)
