__all__ = ["Reports"]

import os
import shutil
import subprocess
import yaml
import logging

from typing import TYPE_CHECKING

from ..utils.namespaces import LazyNamespace
from ..utils.quarto_utils import QuartoHelper as quarto

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .Explanations import Explanations


class Reports(LazyNamespace):
    dependencies = ["yaml"]
    dependency_group = "explanations"
    
    def __init__(self, explanations: "Explanations"):
        self.explanations = explanations

        self.report_dir = os.path.join(self.explanations.root_dir, self.explanations.report_folder)
        self.report_output_dir = os.path.join(self.report_dir, "_site")
        
        self.aggregates_folder = self.explanations.aggregates_folder
        self.params_file = os.path.join(self.report_dir, "scripts", "params.yml")

    @staticmethod
    def generate_zipped_report(output_filename: str, folder_to_zip: str):
        if not os.path.isdir(folder_to_zip):
            logger.error(f"The output path {folder_to_zip} is not a directory.")
            return

        if not os.path.exists(folder_to_zip):
            logger.warning(
                f"The {folder_to_zip} directory does not exist. Skipping zip creation."
            )
            return

        base_filename = os.path.splitext(output_filename)[0]
        zippy = shutil.make_archive(base_filename, "zip", folder_to_zip)
        logger.info(f"created zip file...{zippy}")

    def generate(
        self, 
        report_filename: str = "explanations_report.zip", 
        top_n: int = 10, 
        top_k: int = 10,
        zip_output: bool = False, 
        verbose: bool = False):
        
        self._validate_report_dir()
        
        self._copy_report_resources()

        self._set_params(
            top_n=top_n, top_k=top_k, 
            verbose=verbose)

        self._run_quarto()

        if zip_output:
            self.generate_zipped_report(report_filename, self.report_output_dir)

    def _validate_report_dir(self):
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir, exist_ok=True)
    
    def _copy_report_resources(self):
        try:
            quarto.copy_report_resources(
                resource_dict=[
                    ("GlobalExplanations", self.report_dir), 
                    ("assets", os.path.join(self.report_dir, "assets")),
                ],
            )
        except (OSError, shutil.Error) as e:
            logger.error(f"IO error during resource copy: {e}")
            raise

    def _set_params(
        self, top_n: int = 10, top_k: int = 10, verbose: bool = False):
        with open(self.params_file, "r") as file:
            params = yaml.safe_load(file)

        params["top_n"] = top_n
        params["top_k"] = top_k
        params["verbose"] = verbose
        params["data_folder"] = self.aggregates_folder

        with open(self.params_file, "w") as file:
            yaml.safe_dump(params, file)

        if verbose:
            print(f"""
Report generation params file initialized with the following:
- Aggregated Data folder: {self.aggregates_folder}
- Report directory: {self.report_dir}
- Report output directory: {self.report_output_dir}
- Top N: {top_n}
- Top K: {top_k}
- Verbose: {verbose}
            """)
            
    def _run_quarto(self):
        command = ["quarto", "render"]
        logger.debug(f"Executing: {' '.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            cwd=self.report_dir,
            text=True,
            bufsize=1,  # Line buffered
        )

        if process.stdout is not None:
            for line in iter(process.stdout.readline, ""):
                line = line.strip()
                print(line)
                logger.debug(line)
        else:  # pragma: no cover
            logger.warning("subprocess.stdout is None, unable to read output")

        return_code = process.wait()
        message = f"Quarto process exited with return code {return_code}"
        logger.info(message)

