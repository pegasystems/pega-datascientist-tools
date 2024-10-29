import logging
import os
import shutil
import subprocess
import sys
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import polars as pl

from ..utils import cdh_utils
from ..utils.namespaces import LazyNamespace
from ..utils.types import QUERY

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
        query: Optional[QUERY] = None,
        only_active_predictors: bool = False,
        output_type: str = "html",
        keep_temp_files: bool = False,
        verbose: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
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
        output_type : str, default='html'
            The type of the output file (e.g., "html", "pdf").
        keep_temp_files : bool, optional
            If True, the temporary directory with temp files will not be deleted after report generation.
        verbose: bool, optional
            If True, prints detailed logs during execution.
        progress_callback : Callable[[int, int], None], optional
            A callback function to report progress. Used only in the Streamlit app.
            The function should accept two integers: the current progress and the total.


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
            model_file_path, predictor_file_path = self.datamart.save_data(temp_dir)
            output_file_paths = []
            for i, model_id in enumerate(model_ids):
                output_filename = self._get_output_filename(
                    name, "ModelReport", model_id, output_type
                )
                params = {
                    "model_file_path": str(model_file_path),
                    "predictor_file_path": str(predictor_file_path),
                    "report_type": "ModelReport",
                    "model_id": model_id,
                    "only_active_predictors": only_active_predictors,
                    "query": query,
                }

                self._write_params_file(temp_dir, params)
                self._run_quarto_command(
                    temp_dir,
                    qmd_file,
                    output_type,
                    output_filename,
                    verbose,
                )
                output_path = temp_dir / output_filename
                print(f"{output_path=}")
                if not output_path.exists():
                    raise ValueError(f"Failed to write the report: {output_filename}")
                output_file_paths.append(output_path)
                if progress_callback:
                    progress_callback(i + 1, len(model_ids))
            file_data, file_name = cdh_utils.process_files_to_bytes(
                output_file_paths, base_file_name=output_path
            )
            output_path = working_dir.joinpath(file_name)
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
        query: Optional[QUERY] = None,
        output_type: str = "html",
        keep_temp_files: bool = False,
        verbose: bool = False,
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
        verbose: bool, optional
            If True, prints detailed logs during execution.

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
            model_file_path, predictor_file_path = self.datamart.save_data(temp_dir)
            params = {
                "report_type": "HealthCheck",
                "model_file_path": str(model_file_path),
                "predictor_file_path": str(predictor_file_path),
                "query": query,
            }

            self._write_params_file(temp_dir, params)
            self._run_quarto_command(
                temp_dir,
                qmd_file,
                output_type,
                output_filename,
                verbose,
            )

            output_path = temp_dir / output_filename
            if not output_path.exists():
                raise ValueError(f"Failed to generate report: {output_filename}")

            final_path = working_dir / output_filename
            shutil.copy(output_path, final_path)
            return final_path

        finally:
            if not keep_temp_files and temp_dir.exists() and temp_dir.is_dir():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _get_output_filename(
        self,
        name: Optional[str],
        report_type: str,
        model_id: Optional[str] = None,
        output_type: str = "html",
    ) -> str:
        """Generate the output filename based on the report parameters."""
        name = name.replace(" ", "_") if name else None
        if report_type == "ModelReport":
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

    def _copy_quarto_file(self, qmd_file: str, temp_dir: Path) -> None:
        """Copy the report quarto file to the temporary directory."""
        from pdstools import __reports__

        shutil.copy(__reports__ / qmd_file, temp_dir)

    def _verify_cached_files(self, temp_dir: Path) -> None:
        """Verify that cached data files exist."""
        modeldata_files = list(temp_dir.glob("cached_modelData*"))
        predictordata_files = list(temp_dir.glob("cached_predictorData*"))

        if not modeldata_files:
            raise FileNotFoundError("No cached model data found.")
        if not predictordata_files:
            logger.warning("No cached predictor data found.")

    def _write_params_file(self, temp_dir: Path, params: Dict) -> None:
        """Write parameters to a YAML file."""
        import yaml

        yaml_params = {"kwargs": {key: value for key, value in params.items()}}

        with open(temp_dir / "params.yaml", "w") as f:
            yaml.dump(yaml_params, f)

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
        except FileNotFoundError as e:  # pragma: no cover
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
        except subprocess.CalledProcessError as e:  # pragma: no cover
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

        if process.stdout is not None:
            for line in iter(process.stdout.readline, ""):
                line = line.strip()
                if verbose:
                    print(line)
                logger.info(line)
        else:  # pragma: no cover
            logger.warning("subprocess.stdout is None, unable to read output")

        return_code = process.wait()
        message = f"Quarto process exited with return code {return_code}"
        logger.info(message)

        return return_code

    def _find_quarto_executable(self) -> Path:
        """Find the Quarto executable on the system."""
        if sys.platform == "win32":  # pragma: no cover
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
        else:  # pragma: no cover
            possible_paths = [
                Path("/usr/local/bin/quarto"),
                Path("/opt/quarto/bin/quarto"),
                Path(os.environ.get("HOME", "")) / ".local" / "bin" / "quarto",
            ]

        for path in possible_paths:
            if path.exists():
                return path

        # If not found in common locations, try to find it in PATH
        quarto_in_path = shutil.which("quarto")  # pragma: no cover
        if quarto_in_path:  # pragma: no cover
            return Path(quarto_in_path)

        raise FileNotFoundError(
            "Quarto executable not found. Please ensure Quarto is installed and in the system PATH."
        )  # pragma: no cover

    def excel_report(
        self,
        name: Union[Path, str] = Path("Tables.xlsx"),
        predictor_binning: bool = False,
        query: Optional[QUERY] = None,
    ) -> Optional[Path]:
        """
        Export aggregated data to an Excel file.
        This method exports the last snapshots of model_data, predictor summary,
        and optionally predictor_binning data to separate sheets in an Excel file.
        If a specific table is not available, it will be skipped without causing the export to fail.

        Parameters
        ----------
        name: Union[Path, str], optional
            The path where the Excel file will be saved.
            Defaults to Path("Tables.xlsx").
        predictor_binning: bool, optional
            If True, include predictor_binning data in the export.
            This is the last snapshot of the raw data, so it can be big.
            Defaults to False.

        Returns
        -------
        Union[Path, None]
            The path to the created Excel file if the export was successful,
            None if no data was available to export.
        """
        from xlsxwriter import Workbook

        name = Path(name)
        tabs = {
            "modeldata_last_snapshot": self.datamart.aggregates.last(table="model_data")
        }

        if self.datamart.predictor_data is not None:
            tabs["predictor_last_snapshot"] = (
                self.datamart.aggregates.predictor_last_snapshot()
            )

        if predictor_binning and self.datamart.predictor_data is not None:
            tabs["predictor_binning"] = self.datamart.aggregates.last(
                table="combined_data"
            ).filter(pl.col("PredictorName") != "Classifier")

        # Remove None values (tables that are not available)
        tabs = {k: v for k, v in tabs.items() if v is not None}

        if not tabs:  # pragma: no cover
            print("No data available to export.")
            return None

        with Workbook(
            name, options={"nan_inf_to_errors": True, "remove_timezone": True}
        ) as wb:
            for tab, data in tabs.items():
                data = data.with_columns(
                    pl.col(pl.List(pl.Categorical), pl.List(pl.Utf8))
                    .list.eval(pl.element().cast(pl.Utf8))
                    .list.join(", ")
                )
                data.collect().write_excel(workbook=wb, worksheet=tab)

        print(f"Data exported to {name}")
        return name
