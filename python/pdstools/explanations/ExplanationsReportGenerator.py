import os
import shutil
import pathlib
import subprocess
import yaml
import logging
from importlib import resources

logger = logging.getLogger(__name__)


class ExplanationsReportGenerator:
    def __init__(
        self,
        root_dir: str = ".tmp",
        data_folder: str = "aggregated_data",
        report_dir: str = "reports",
        report_output_dir: str = "_site",
        report_filename: str = "explanations_report.zip",
        top_n: int = 10,
        top_k: int = 10,
        zip_output: bool = True,
        verbose: bool = False,
    ):
        self.root_dir = root_dir

        self.report_dir = os.path.join(root_dir, report_dir)
        self.data_folder = data_folder
        self.report_output_dir = os.path.join(self.report_dir, report_output_dir)
        self.params_file = os.path.join(self.report_dir, "scripts", "params.yml")
        self.report_filename = report_filename

        self.zip_output = zip_output
        self.verbose = verbose

        self.top_n = top_n
        self.top_k = top_k

        self._validate_report_dir()

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

    def process(self):
        self._copy_report_resources()

        self._set_params()

        self._run_quarto()

        if self.zip_output:
            self.generate_zipped_report(self.report_filename, self.report_output_dir)

    def _validate_report_dir(self):
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir, exist_ok=True)

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

    def _copy_report_resources(self):
        from pdstools import __reports__
        
        shutil.copytree(__reports__ / "GlobalExplanations", self.report_dir, dirs_exist_ok=True)
        shutil.copytree(__reports__ / "assets", pathlib.Path(self.report_dir, "assets"), dirs_exist_ok=True)

    def _set_params(self):
        with open(self.params_file, "r") as file:
            params = yaml.safe_load(file)

        params["top_n"] = self.top_n
        params["top_k"] = self.top_k
        params["verbose"] = self.verbose
        params["data_folder"] = self.data_folder

        with open(self.params_file, "w") as file:
            yaml.safe_dump(params, file)

        if self.verbose:
            print(f"""
Report generation params file initialized with the following:
- Aggregated Data folder: {self.data_folder}
- Report directory: {self.report_dir}
- Report output directory: {self.report_output_dir}
- Top N: {self.top_n}
- Top K: {self.top_k}
- Zip output: {self.zip_output}
- Verbose: {self.verbose}
            """)
