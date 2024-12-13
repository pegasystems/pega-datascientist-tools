__all__ = ["Reports"]
import logging
import os
import re
import shutil
import subprocess
import sys
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import polars as pl

from ..utils import cdh_utils
from ..utils.namespaces import LazyNamespace
from ..utils.types import QUERY
from ..prediction import Prediction

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
        name: Optional[
            str
        ] = None,  # TODO when ends with .html assume its the full name but this could be in _get_output_filename
        title: str = "ADM Model Report",
        subtitle: str = "",
        output_dir: Optional[PathLike] = None,
        only_active_predictors: bool = False,
        output_type: str = "html",
        keep_temp_files: bool = False,
        verbose: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        model_file_path: Optional[PathLike] = None,
        predictor_file_path: Optional[PathLike] = None,
    ) -> Path:
        """
        Generates model reports.

        Parameters
        ----------
        model_ids : List[str]
            The list of model IDs to generate reports for.
        name : str, optional
            The (base) file name of the report.
        title : str, optional
            Title top put in the report, uses a default string if not given.
        subtitle : str, optional
            Subtitle top put in the report, empty if not given.
        output_dir : Union[str, Path, None], optional
            The directory for the output. If None, uses current working directory.
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
        model_file_path : Union[str, Path, None], optional
            Optional name of the actual model data file, so it does not get copied
        predictor_file_path : Union[str, Path, None], optional
            Optional name of the actual predictor data file, so it does not get copied


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
        output_dir, temp_dir = cdh_utils.create_working_and_temp_dir(name, output_dir)

        try:
            qmd_file = "ModelReport.qmd"
            self._copy_quarto_file(qmd_file, temp_dir)

            # Copy data to a temp dir only if the files are not passed in already
            if (
                (model_file_path is None) and (self.datamart.model_data is not None)
            ) or (
                (predictor_file_path is None)
                and (self.datamart.predictor_data is not None)
            ):
                model_file_path, predictor_file_path = self.datamart.save_data(
                    temp_dir, selected_model_ids=model_ids
                )

            output_file_paths = []
            for i, model_id in enumerate(model_ids):
                output_filename = self._get_output_filename(
                    name, "ModelReport", model_id, output_type
                )
                self.run_quarto(
                    qmd_file=qmd_file,
                    output_filename=output_filename,
                    output_type=output_type,
                    params={
                        "report_type": "ModelReport",
                        "model_file_path": str(model_file_path),
                        "predictor_file_path": str(predictor_file_path),
                        "model_id": model_id,
                        "only_active_predictors": only_active_predictors,
                        "title": title,
                        "subtitle": subtitle,
                    },
                    project={"title": title, "type": "default"},
                    analysis={
                        "predictions": False,
                        "predictors": (self.datamart.predictor_data is not None),
                        "models": (self.datamart.model_data is not None),
                    },
                    temp_dir=temp_dir,
                    verbose=verbose,
                )
                output_path = temp_dir / output_filename
                if verbose or not output_path.exists():
                    # print parameters so they can be copy/pasted into the quarto docs for debugging
                    if model_file_path is not None:
                        print(f'model_file_path = "{model_file_path}"')
                    if predictor_file_path is not None:
                        print(f'predictor_file_path = "{predictor_file_path}"')
                    print(f'model_id = "{model_id}"')
                    print(f"output_path = {output_path}")
                if not output_path.exists():
                    raise ValueError(f"Failed to write the report: {output_filename}")
                output_file_paths.append(output_path)
                if progress_callback:
                    progress_callback(i + 1, len(model_ids))
            # Is this just a difficult way to copy the file? Why not shutil.copy? Or
            # even pass in the output-dir property to the quarto project?
            file_data, file_name = cdh_utils.process_files_to_bytes(
                output_file_paths, base_file_name=output_path
            )
            output_path = output_dir.joinpath(file_name)
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
        name: Optional[
            str
        ] = None,  # TODO when ends with .html assume its the full name but this could be in _get_output_filename
        title: str = "ADM Model Overview",
        subtitle: str = "",
        output_dir: Optional[os.PathLike] = None,
        *,
        query: Optional[QUERY] = None,
        output_type: str = "html",
        keep_temp_files: bool = False,
        verbose: bool = False,
        model_file_path: Optional[PathLike] = None,
        predictor_file_path: Optional[PathLike] = None,
        prediction_file_path: Optional[PathLike] = None,
    ) -> Path:
        """
        Generates Health Check report based on the provided parameters.

        Parameters
        ----------
        name : str, optional
            The (base) file name of the report.
        title : str, optional
            Title top put in the report, uses a default string if not given.
        subtitle : str, optional
            Subtitle top put in the report, empty if not given.
        query : QUERY, optional
            Optional extra filter on the datamart data
        output_dir : Union[str, Path, None], optional
            The directory for the output. If None, uses current working directory.
        output_type : str, default='html'
            The type of the output file (e.g., "html", "pdf").
        keep_temp_files : bool, optional
            If True, the temporary directory with temp files will not be deleted after report generation.
        verbose: bool, optional
            If True, prints detailed logs during execution.
        model_file_path : Union[str, Path, None], optional
            Optional name of the actual model data file, so it does not get copied
        predictor_file_path : Union[str, Path, None], optional
            Optional name of the actual predictor data file, so it does not get copied
        prediction_file_path : Union[str, Path, None], optional
            Optional name of the actual predictions data file, so it does not get copied

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
        output_dir, temp_dir = cdh_utils.create_working_and_temp_dir(name, output_dir)
        try:
            qmd_file = "HealthCheck.qmd"
            output_filename = self._get_output_filename(
                name, "HealthCheck", None, output_type
            )

            self._copy_quarto_file(qmd_file, temp_dir)

            # Copy data to a temp dir only if the files are not passed in already
            if (
                (model_file_path is None) and (self.datamart.model_data is not None)
            ) or (
                (predictor_file_path is None)
                and (self.datamart.predictor_data is not None)
            ):
                model_file_path, predictor_file_path = self.datamart.save_data(temp_dir)

            self.run_quarto(
                qmd_file=qmd_file,
                output_filename=output_filename,
                output_type=output_type,
                params={
                    "report_type": "HealthCheck",
                    "model_file_path": str(model_file_path),
                    "predictor_file_path": str(predictor_file_path),
                    "prediction_file_path": str(prediction_file_path),
                    "query": query,
                    "title": title,
                    "subtitle": subtitle,
                },
                project={"title": title, "type": "default"},
                analysis={
                    "predictions": (prediction_file_path is not None),
                    "predictors": (self.datamart.predictor_data is not None),
                    "models": (self.datamart.model_data is not None),
                },
                temp_dir=temp_dir,
                verbose=verbose,
            )

            output_path = temp_dir / output_filename
            if verbose or not output_path.exists():
                if model_file_path is not None:
                    print(f'model_file_path = "{model_file_path}"')
                if predictor_file_path is not None:
                    print(f'predictor_file_path = "{predictor_file_path}"')
                if prediction_file_path is not None:
                    print(f'prediction_file_path = "{prediction_file_path}"')
                print(f"output_path = {output_path}")
            if not output_path.exists():
                raise ValueError(f"Failed to generate report: {output_filename}")

            # TODO consider passing in the output-dir property to the quarto project so quarto does the copying
            final_path = output_dir / output_filename
            shutil.copy(output_path, final_path)

            return final_path

        finally:
            if not keep_temp_files and temp_dir.exists() and temp_dir.is_dir():
                shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _get_output_filename(
        name: Optional[str],  # going to be the full file name
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

    @staticmethod
    def _copy_quarto_file(qmd_file: str, temp_dir: Path) -> None:
        """Copy the report quarto file to the temporary directory."""
        from pdstools import __reports__

        shutil.copy(__reports__ / qmd_file, temp_dir)

    # Never used?
    # def _verify_cached_files(self, temp_dir: Path) -> None:
    #     """Verify that cached data files exist."""
    #     modeldata_files = list(temp_dir.glob("cached_modelData*"))
    #     predictordata_files = list(temp_dir.glob("cached_predictorData*"))

    #     if not modeldata_files:
    #         raise FileNotFoundError("No cached model data found.")
    #     if not predictordata_files:
    #         logger.warning("No cached predictor data found.")

    @staticmethod
    def _write_params_files(
        temp_dir: Path,
        params: Dict = {},
        project: Dict = {"type": "default"},
        analysis: Dict = {},
    ) -> None:
        """Write parameters to a YAML file."""
        import yaml

        # Parameters to python code

        with open(temp_dir / "params.yml", "w") as f:
            yaml.dump(
                params,
                f,
            )

        # Project/rendering options to quarto

        with open(temp_dir / "_quarto.yml", "w") as f:
            yaml.dump(
                {
                    "project": project,
                    "analysis": analysis,
                },
                f,
            )

    @staticmethod
    def _find_executable(exec_name: str) -> Path:
        """Find the executable on the system."""

        # First find in path
        exec_in_path = shutil.which(exec_name)  # pragma: no cover
        if exec_in_path:  # pragma: no cover
            return Path(exec_in_path)

        # If not in path try find explicitly. TODO not sure this is wise
        # maybe we should not try be smart and assume quarto/pandoc are
        # properly installed.

        if sys.platform == "win32":  # pragma: no cover
            possible_paths = [
                Path(
                    os.environ.get("USERPROFILE", ""),
                    "AppData",
                    "Local",
                    "Programs",
                    f"{exec_name}",  # assume windows is still case insensitive (NTFS changes this...)
                    "bin",
                    f"{exec_name}.cmd",
                ),
                Path(
                    os.environ.get("PROGRAMFILES", ""),
                    f"{exec_name}",
                    "bin",
                    f"{exec_name}.cmd",
                ),
            ]
        else:  # pragma: no cover
            possible_paths = [
                Path(f"/usr/local/bin/{exec_name}"),
                Path(f"/opt/{exec_name}/bin/{exec_name}"),
                Path(os.environ.get("HOME", ""), ".local", "bin", exec_name),
            ]

        for path in possible_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            "Quarto executable not found. Please ensure Quarto is installed and in the system PATH."
        )  # pragma: no cover

    # TODO not conviced about below. This isn't necessarily the same path resolution
    # as the os does. What's wrong with just assuming quarto is in the path so we can
    # just test for version w code like
    # def get_cmd_output(args):
    # result = (
    #     subprocess.run(args, stdout=subprocess.PIPE).stdout.decode("utf-8").split("\n")
    # )
    # return result
    # get_version_only(get_cmd_output(["quarto", "--version"])[0])

    @staticmethod
    def _get_executable_with_version(
        exec_name: str, verbose: bool = False
    ) -> Tuple[Path, str]:
        def get_version_only(versionstr):
            return re.sub("[^.0-9]", "", versionstr)

        try:
            executable = Reports._find_executable(exec_name=exec_name)
        except FileNotFoundError as e:  # pragma: no cover
            logger.error(e)
            raise

        # Check version
        try:
            version_result = subprocess.run(
                [str(executable), "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            version_string = get_version_only(
                version_result.stdout.split("\n")[0].strip()
            )
            message = f"{exec_name} version: {version_string}"
            logger.info(message)
            if verbose:
                print(message)
        except subprocess.CalledProcessError as e:  # pragma: no cover
            logger.warning(f"Failed to check {exec_name} version: {e}")

        return (executable, version_string)

    @staticmethod
    def get_quarto_with_version(verbose: bool = True) -> Tuple[Path, str]:
        return Reports._get_executable_with_version("quarto", verbose=verbose)

    @staticmethod
    def get_pandoc_with_version(verbose: bool = True) -> Tuple[Path, str]:
        return Reports._get_executable_with_version("pandoc", verbose=verbose)

    @staticmethod
    def run_quarto(
        qmd_file: str,
        output_filename: str,
        output_type: str = "html",
        params: Dict = {},
        project: Dict = {"type": "default"},
        analysis: Dict = {},
        temp_dir: Path = Path("."),
        verbose: bool = False,
    ) -> int:
        """Run the Quarto command to generate the report."""

        Reports._write_params_files(
            temp_dir,
            params=params,
            project=project,
            analysis=analysis,
        )

        quarto_exec, _ = Reports.get_quarto_with_version(verbose)

        command = [
            str(quarto_exec),
            "render",
            qmd_file,
            "--to",
            output_type,
            "--output",
            output_filename,
            "--execute-params",
            "params.yml",
        ]

        if verbose:
            print(f"Executing: {' '.join(command)}")

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
            tabs["predictors_overview"] = (
                self.datamart.aggregates.predictors_overview()
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
