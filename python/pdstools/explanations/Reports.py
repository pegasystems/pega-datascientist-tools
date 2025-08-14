__all__ = ["Reports"]

import os
import shutil
import subprocess
import yaml
import logging

from typing import TYPE_CHECKING

from ..utils.namespaces import LazyNamespace
from ..utils.report_utils import (
    copy_report_resources,
    run_quarto,
    generate_zipped_report,
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
        try:
            self.explanations.aggregate.validate_folder()
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise

        self._validate_report_dir()

        try:
            self._copy_report_resources()
        except (OSError, shutil.Error) as e:
            logger.error(f"IO error during resource copy: {e}")
            raise

        self._set_params(top_n=top_n, top_k=top_k, verbose=verbose)

        try:
            return_code = run_quarto(temp_dir=self.report_folderpath, verbose=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Quarto command failed: {e}")
            raise

        if return_code != 0:
            logger.error(f"Quarto command failed with return code {return_code}")
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

        with open(self.params_file, "w") as file:
            yaml.safe_dump(params, file)
