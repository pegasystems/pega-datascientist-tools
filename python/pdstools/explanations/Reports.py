from __future__ import annotations

__all__ = ["Reports"]

import logging
import shutil
import subprocess
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

import yaml

from ..utils.namespaces import LazyNamespace
from ..utils.report_utils import (
    copy_report_resources,
    generate_zipped_report,
    run_quarto,
)
from .ExplanationsUtils import (
    _CONTRIBUTION_TYPE,
    DisplayBy,
    SortBy,
    _resolve_contribution_type,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .Explanations import Explanations


class Reports(LazyNamespace):
    """Reports."""

    dependencies: ClassVar[list[str]] = ["yaml"]
    dependency_group = "explanations"

    def __init__(self, explanations: "Explanations"):
        self.explanations = explanations

        self.report_foldername = "reports"
        self.report_folderpath = Path(self.explanations.root_dir) / self.report_foldername
        self.report_output_dir = self.report_folderpath / "_site"

        self.aggregate_folder = self.explanations.data_folder
        # Safeguard: aggregate_folder is guaranteed to be Path from Aggregate.data_folderpath
        self.params_file = self.report_folderpath / "scripts" / "params.yml"

        super().__init__()

    def generate(
        self,
        report_filename: str = "explanations_report.zip",
        top_n: int = 20,
        top_k: int = 20,
        zip_output: bool = False,
        full_embed: bool = False,
        *,
        sort_by: SortBy = "contribution_abs",
        display_by: DisplayBy = "contribution",
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
        full_embed : bool, default=False
            When True, fully embed JavaScript libraries into the generated Quarto
            website. Defaults to False to preserve the current CDN-based output.
        sort_by : str, keyword-only
            Column to rank/select top predictors. Default: ``"contribution_abs"``.
        display_by : str, keyword-only
            Column to use for the report axis values. Default: ``"contribution"``.

        Notes
        -----
        Progress and diagnostic information is logged at DEBUG level.
        Enable debug logging to see detailed report generation steps.

        """
        try:
            self.explanations.validate_data_folder()
            co = self.explanations.aggregate.context_operations
            contexts = co.create_unique_contexts_file()
            co.create_batch_parquet_files(contexts)
        except Exception as e:
            logger.error("Validation failed: %s", e)
            raise

        validated_sort_by = _resolve_contribution_type(sort_by)
        validated_display_by = _resolve_contribution_type(display_by)

        self._validate_report_dir()

        try:
            self._copy_report_resources()
            self._set_full_embed_options(full_embed=full_embed)
        except (OSError, shutil.Error) as e:
            logger.error("IO error during resource copy: %s", e)
            raise

        from_date = self.explanations.from_date.strftime("%Y-%m-%d") if self.explanations.from_date else ""
        to_date = self.explanations.to_date.strftime("%Y-%m-%d") if self.explanations.to_date else ""
        self._set_params(
            top_n=top_n,
            top_k=top_k,
            from_date=from_date,
            to_date=to_date,
            sort_by=validated_sort_by,
            display_by=validated_display_by,
            full_embed=full_embed,
        )

        try:
            return_code = run_quarto(
                temp_dir=Path(self.report_folderpath),
                output_type=None,
                full_embed=full_embed,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Quarto command failed: %s", e)
            raise

        if return_code != 0:
            logger.error("Quarto command failed with return code %s", return_code)
            raise RuntimeError(f"Quarto command failed with return code {return_code}")

        if zip_output:
            generate_zipped_report(report_filename, self.report_output_dir)

    def _validate_report_dir(self):
        self.report_folderpath.mkdir(parents=True, exist_ok=True)

    def _copy_report_resources(self):
        logger.debug("Copying report resources to %s", self.report_folderpath)
        copy_report_resources(
            resource_dict=[
                ("GlobalExplanations", str(self.report_folderpath)),
                ("assets", str(self.report_folderpath / "assets")),
            ],
        )

    def _set_full_embed_options(self, *, full_embed: bool) -> None:
        quarto_config_path = self.report_folderpath / "_quarto.yml"
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
        top_n: int = 20,
        top_k: int = 20,
        from_date: str = "",
        to_date: str = "",
        sort_by: _CONTRIBUTION_TYPE = _CONTRIBUTION_TYPE.CONTRIBUTION_ABS,
        display_by: _CONTRIBUTION_TYPE = _CONTRIBUTION_TYPE.CONTRIBUTION,
        full_embed: bool = False,
    ):
        params: dict[str, str | int | bool] = {}
        params["top_n"] = top_n
        params["top_k"] = top_k
        params["from_date"] = from_date
        params["to_date"] = to_date
        params["sort_by"] = sort_by.value
        params["sort_by_text"] = sort_by.text
        params["display_by"] = display_by.value
        params["display_by_text"] = display_by.text
        params["data_folder"] = str(self.aggregate_folder)
        params["full_embed"] = full_embed

        logger.debug("Writing report parameters to %s", self.params_file)
        with open(self.params_file, "w", encoding="utf-8") as file:
            yaml.safe_dump(params, file)
