import logging
import os
import shutil
import subprocess
import sys
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from ..utils import cdh_utils
from ..utils.namespaces import LazyNamespace

if TYPE_CHECKING:
    from .ADMDatamart import ADMDatamart
logger = logging.getLogger(__name__)


class Reports(LazyNamespace):
    dependencies = ["yaml"]
    dependency_group = "healthcheck"

    def __init__(self, datamart: "ADMDatamart"):
        self.datamart = datamart
        super().__init__()

    def model_reports(
        self,
        model_ids: List[str],
        *,
        name: Optional[str] = None,
        working_dir: Optional[PathLike] = None,
        only_active_predictors: bool = False,
        base_file_name: Optional[str] = None,
        output_type: str = "html",
        keep_temp_files: bool = False,
        progress_callback=None,  #:  Callable[[int, int], None] = None,
    ) -> Path:
        """
        Generates model reports.

        Parameters
        ----------
        name : str, optional
            The name of the report.
        model_list : List[str]
            The list of model IDs to generate reports for.
        working_dir : Union[str, Path, None], optional
            The working directory for the output. If None, uses current working directory.
        only_active_predictors : bool, default=False
            Whether to only include active predictor details.
        base_file_name : str, optional
            The base file name for the generated reports. Defaults to None.
        output_type : str, default='html'
            The type of the output file (e.g., "html", "pdf").
        keep_temp_files : bool, optional
            If True, the temporary directory with temp files will not be deleted after report generation.


        Returns
        -------
        Path
            The path to the generated report file.

        Raises
        ------
        ValueError
            If there's an error in report generation or invalid parameters.
        FileNotFoundError
            If required files are not found.
        subprocess.SubprocessError
            If there's an error in running external commands.
        """

        if (
            not model_ids
            or not isinstance(model_ids, list)
            or not all(isinstance(i, str) for i in model_ids)
        ):
            raise ValueError(
                "model_list argument is None, not a list, or contains non-string elements for generate_model_reports. Please provide a list of model_id strings to generate reports."
            )
        working_dir, temp_dir = cdh_utils.create_working_and_temp_dir(name, working_dir)

        try:
            qmd_file = "ModelReport.qmd"
            self._copy_quarto_file(qmd_file, temp_dir)
            self.datamart.save_data(temp_dir)
            output_file_paths = []
            for i, model_id in enumerate(model_ids):
                output_filename = self._get_output_filename(
                    name, "ModelReport", model_id, output_type
                )
                self._write_params_file(temp_dir, model_id, only_active_predictors)
                self._run_quarto_command(
                    temp_dir,
                    qmd_file,
                    output_type,
                    output_filename,
                )
                output_path = temp_dir / output_filename
                if not output_path.exists():
                    raise ValueError(f"Failed to write the report: {output_filename}")
                output_file_paths.append(output_path)
                if progress_callback:
                    progress_callback(i + 1, len(model_ids))
            base_file_name = kwargs.get(
                base_file_name,
            )  # TODO: strip this, let's just use a default name?
            file_data, file_name = cdh_utils.process_files_to_bytes(
                output_file_paths, base_file_name
            )
            output_path = working_dir / file_name
            with open(output_path, "wb") as f:
                f.write(file_data)
            if not output_path.exists():
                raise ValueError(f"Failed to generate report: {output_filename}")

            return output_path

        except Exception as e:
            logger.error(e)
            raise
        finally:
            if not keep_temp_files:
                if temp_dir.exists() and temp_dir.is_dir():
                    shutil.rmtree(temp_dir, ignore_errors=True)

    def health_check(
        self,
        name: Optional[str] = None,
        working_dir: Optional[os.PathLike] = None,
        *,
        output_type: str = "html",
        keep_temp_files: bool = False,
    ) -> Path:
        """
        Generates Health Check report based on the provided parameters.

        Parameters
        ----------
        name : str, optional
            The name of the report.
        working_dir : Union[str, Path, None], optional
            The working directory for the output. If None, uses current working directory.
        output_type : str, default='html'
            The type of the output file (e.g., "html", "pdf").
        keep_temp_files : bool, optional
            If True, the temporary directory with temp files will not be deleted after report generation.


        Returns
        -------
        Path
            The path to the generated report file.

        Raises
        ------
        ValueError
            If there's an error in report generation or invalid parameters.
        FileNotFoundError
            If required files are not found.
        subprocess.SubprocessError
            If there's an error in running external commands.
        """
        working_dir, temp_dir = cdh_utils.create_working_and_temp_dir(name, working_dir)
        try:
            qmd_file = "HealthCheck.qmd"
            output_filename = self._get_output_filename(
                name, "HealthCheck", None, output_type
            )

            self._copy_quarto_file(qmd_file, temp_dir)
            self.datamart.save_data(temp_dir)
            self._write_params_file(temp_dir, None, None)
            self._run_quarto_command(
                temp_dir,
                qmd_file,
                output_type,
                output_filename,
            )

            output_path = temp_dir / output_filename
            if not output_path.exists():
                raise ValueError(f"Failed to generate report: {output_filename}")

            final_path = working_dir / output_filename
            shutil.copy(output_path, final_path)
            return final_path

        except Exception as e:
            raise e
        finally:
            if not keep_temp_files and temp_dir.exists() and temp_dir.is_dir():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _get_output_filename(self, name, report_type, model_id, output_type):
        """Generate the output filename based on the report parameters."""
        name = name.replace(" ", "_") if name else None
        if report_type == "ModelReport":
            if not model_id:
                raise ValueError("model_id is required for a model report.")
            return (
                f"{report_type}_{name}_{model_id}.{output_type}"
                if name
                else f"{report_type}_{model_id}.{output_type}"
            )
        return (
            f"{report_type}_{name}.{output_type}"
            if name
            else f"{report_type}.{output_type}"
        )

    def _copy_quarto_file(self, qmd_file, temp_dir):
        """Copy the report quarto file to the temporary directory."""
        from pdstools import __reports__

        shutil.copy(__reports__ / qmd_file, temp_dir)

    def _verify_cached_files(self, temp_dir):
        """Verify that cached data files exist."""
        modeldata_files = list(temp_dir.glob("cached_modelData*"))
        predictordata_files = list(temp_dir.glob("cached_predictorData*"))

        if not modeldata_files:
            raise FileNotFoundError("No cached model data found.")
        if not predictordata_files:
            logger.warning("No cached predictor data found.")

    def _write_params_file(self, temp_dir, model_id, only_active_predictors):
        """Write parameters to a YAML file."""
        import yaml

        params = {
            "kwargs": {
                "subset": False,
                "model_id": model_id,
                "only_active_predictors": only_active_predictors,
            },
        }
        with open(temp_dir / "params.yaml", "w") as f:
            yaml.dump(params, f)

    def _run_quarto_command(
        self,
        temp_dir: Path,
        qmd_file: str,
        output_type: str,
        output_filename: str,
        verbose: bool = True,
    ) -> int:
        """Run the Quarto command to generate the report."""
        if verbose:
            print("Set verbose=False to hide output.")
        try:
            quarto_exec = self._find_quarto_executable()
        except FileNotFoundError as e:
            logger.error(e)
            raise

        # Check Quarto version
        try:
            version_result = subprocess.run(
                [str(quarto_exec), "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            quarto_version = version_result.stdout.strip()
            message = f"Quarto version: {quarto_version}"
            logger.info(message)
            if verbose:
                print(message)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to check Quarto version: {e}")

        command = [
            str(quarto_exec),
            "render",
            qmd_file,
            "--to",
            output_type,
            "--output",
            output_filename,
            "--execute-params",
            "params.yaml",
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            cwd=temp_dir,
            text=True,
            bufsize=1,  # Line buffered
        )

        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            logger.info(line)

        return_code = process.wait()
        message = f"Quarto process exited with return code {return_code}"
        logger.info(message)

        return return_code

    def _find_quarto_executable(self):
        """Find the Quarto executable on the system."""
        if sys.platform == "win32":
            possible_paths = [
                Path(os.environ.get("USERPROFILE", ""))
                / "AppData"
                / "Local"
                / "Programs"
                / "Quarto"
                / "bin"
                / "quarto.cmd",
                Path(os.environ.get("PROGRAMFILES", ""))
                / "Quarto"
                / "bin"
                / "quarto.cmd",
            ]
        else:
            possible_paths = [
                Path("/usr/local/bin/quarto"),
                Path("/opt/quarto/bin/quarto"),
                Path(os.environ.get("HOME", "")) / ".local" / "bin" / "quarto",
            ]

        for path in possible_paths:
            if path.exists():
                return path

        # If not found in common locations, try to find it in PATH
        quarto_in_path = shutil.which("quarto")
        if quarto_in_path:
            return Path(quarto_in_path)

        raise FileNotFoundError(
            "Quarto executable not found. Please ensure Quarto is installed and in the system PATH."
        )
