__all__ = ["Reports"]

import logging
import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import yaml

from ..utils.namespaces import LazyNamespace
from ..utils.report_utils import (
    copy_report_resources,
    generate_zipped_report,
    run_quarto,
)
from .ExplanationsUtils import _DEFAULT

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .Explanations import Explanations


class Reports(LazyNamespace):
    dependencies = ["yaml"]
    dependency_group = "explanations"

    def __init__(self, explanations: "Explanations"):
        self.explanations = explanations

        self.report_foldername = "reports"
        self.report_folderpath = os.path.join(
            self.explanations.root_dir, self.report_foldername
        )
        self.report_output_dir = os.path.join(self.report_folderpath, "_site")

        self.aggregate_folder = self.explanations.aggregate.data_folderpath
        self.params_file = os.path.join(self.report_folderpath, "scripts", "params.yml")
        super().__init__()

    def generate(
        self,
        report_filename: str = "explanations_report.zip",
        top_n: int = _DEFAULT.TOP_N.value,
        top_k: int = _DEFAULT.TOP_K.value,
        zip_output: bool = False,
        verbose: bool = False,
    ):
        """Generate the explanations report.

        Args:
            report_filename (str):
                Name of the output report file.
            top_n (int):
                Number of top explanations to include.
            top_k (int):
                Number of top features to include in explanations.
            zip_output (bool):
                Whether to zip the output report.
                The filename will be used as the zip file name.
            verbose (bool):
                Whether to print verbose output during report generation.
        """
        try:
            self.explanations.aggregate.validate_folder()
        except Exception as e:
            logger.error("Validation failed: %s", e)
            raise

        self._validate_report_dir()

        try:
            self._copy_report_resources()
        except (OSError, shutil.Error) as e:
            logger.error("IO error during resource copy: %s", e)
            raise

        self._set_params(top_n=top_n, top_k=top_k, verbose=verbose)

        try:
            return_code = run_quarto(
                temp_dir=self.report_folderpath,
                verbose=verbose,
                output_type=None)
        except subprocess.CalledProcessError as e:
            logger.error("Quarto command failed: %s", e)
            raise

        if return_code != 0:
            logger.error("Quarto command failed with return code %s", return_code)
            raise RuntimeError(f"Quarto command failed with return code {return_code}")

        if zip_output:
            generate_zipped_report(report_filename, self.report_output_dir)

    def _validate_report_dir(self):
        if not os.path.exists(self.report_folderpath):
            os.makedirs(self.report_folderpath, exist_ok=True)

    def _copy_report_resources(self):
        copy_report_resources(
            resource_dict=[
                ("GlobalExplanations", self.report_folderpath),
                ("assets", os.path.join(self.report_folderpath, "assets")),
            ],
        )

    def _set_params(
        self,
        top_n: int = _DEFAULT.TOP_N.value,
        top_k: int = _DEFAULT.TOP_K.value,
        verbose: bool = False,
    ):
        params = {}
        params["top_n"] = top_n
        params["top_k"] = top_k
        params["verbose"] = verbose
        params["data_folder"] = self.aggregate_folder.name

        with open(self.params_file, "w", encoding="utf-8") as file:
            yaml.safe_dump(params, file)
