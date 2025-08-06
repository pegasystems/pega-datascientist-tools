__all__ = ["Reports"]
import logging
import os
import shutil
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import polars as pl

from ..utils import cdh_utils
from ..utils.namespaces import LazyNamespace
from ..utils.types import QUERY
from ..utils.report_utils import _serialize_query, get_quarto_with_version, run_quarto, copy_quarto_file, get_output_filename

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
        model_ids: Union[str,List[str]],
        *,
        name: Optional[
            str
        ] = None,  # TODO when ends with .html assume its the full name but this could be in get_output_filename
        title: str = "ADM Model Report",
        disclaimer: str = "",
        subtitle: str = "",
        output_dir: Optional[PathLike] = None,
        only_active_predictors: bool = True,
        output_type: str = "html",
        keep_temp_files: bool = False,
        verbose: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        model_file_path: Optional[PathLike] = None,
        predictor_file_path: Optional[PathLike] = None,
    ) -> Path:
        """
        Generates model reports for Naive Bayes ADM models.

        Parameters
        ----------
        model_ids : Union[str,List[str]]
            The model ID (or list of model IDs) to generate reports for.
        name : str, optional
            The (base) file name of the report.
        title : str, optional
            Title to put in the report, uses a default string if not given.
        subtitle : str, optional
            Subtitle to put in the report, empty if not given.
        disclaimer : str, optional
            Disclaimer blub to put in the report, empty if not given.
        output_dir : Union[str, Path, None], optional
            The directory for the output. If None, uses current working directory.
        only_active_predictors : bool, default=True
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

        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if (
            not model_ids
            or not isinstance(model_ids, list)
            or not all(isinstance(i, str) for i in model_ids)
        ):
            raise ValueError(
                "No valid model IDs"
            )
        output_dir, temp_dir = cdh_utils.create_working_and_temp_dir(name, output_dir)

        try:
            qmd_file = "ModelReport.qmd"
            copy_quarto_file(qmd_file, temp_dir)

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
                output_filename = get_output_filename(
                    name, "ModelReport", model_id, output_type
                )
                run_quarto(
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
                        "disclaimer": disclaimer,
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
        ] = None,  # TODO when ends with .html assume its the full name but this could be in get_output_filename
        title: str = "ADM Model Overview",
        subtitle: str = "",
        disclaimer: str = "",
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
        Generates Health Check report for ADM models, optionally including predictor and prediction sections.

        Parameters
        ----------
        name : str, optional
            The (base) file name of the report.
        title : str, optional
            Title to put in the report, uses a default string if not given.
        subtitle : str, optional
            Subtitle to put in the report, empty if not given.
        disclaimer : str, optional
            Disclaimer blub to put in the report, empty if not given.
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
            output_filename = get_output_filename(
                name, "HealthCheck", None, output_type
            )

            copy_quarto_file(qmd_file, temp_dir)

            # Copy data to a temp dir only if the files are not passed in already
            if (
                (model_file_path is None) and (self.datamart.model_data is not None)
            ) or (
                (predictor_file_path is None)
                and (self.datamart.predictor_data is not None)
            ):
                model_file_path, predictor_file_path = self.datamart.save_data(temp_dir)
            serialized_query = _serialize_query(query)
            run_quarto(
                qmd_file=qmd_file,
                output_filename=output_filename,
                output_type=output_type,
                params={
                    "report_type": "HealthCheck",
                    "model_file_path": str(model_file_path),
                    "predictor_file_path": str(predictor_file_path),
                    "prediction_file_path": str(prediction_file_path),
                    "query": serialized_query,
                    "title": title,
                    "subtitle": subtitle,
                    "disclaimer": disclaimer,
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

            # TODO why not print paths earlier, before the quarto call?
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


    def excel_report(
        self,
        name: Union[Path, str] = Path("Tables.xlsx"),
        predictor_binning: bool = False,
    ) -> tuple[Optional[Path], list[str]]:
        """
        Export raw data to an Excel file.

        This method exports the last snapshots of model_data, predictor summary,
        and optionally predictor_binning data to separate sheets in an Excel file.

        If a specific table is not available or too large, it will be skipped without
        causing the export to fail.

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
        tuple[Union[Path, None], list[str]]
            A tuple containing:
            - The path to the created Excel file if the export was successful, None if no data was available
            - A list of warning messages (empty if no warnings)
        """
        from xlsxwriter import Workbook

        EXCEL_ROW_LIMIT = 1048576
        # Standard ZIP format limit is 4gb but 3gb would already crash most laptops
        ZIP_SIZE_LIMIT_MB = 3000
        warning_messages = []

        name = Path(name)
        tabs = {
            "adm_models": self.datamart.aggregates.last(table="model_data")
            .with_columns(
                pl.col("ResponseCount", "Positives", "Negatives").cast(pl.Int64),
            )
            .select(
                ["ModelID"]
                + self.datamart.context_keys
                + [
                    col
                    for col in self.datamart.model_data.collect_schema().names()
                    if col != "ModelID" and col not in self.datamart.context_keys
                ]
            )
            .sort(self.datamart.context_keys)
        }

        if self.datamart.predictor_data is not None:
            tabs["predictors_detail"] = self.datamart.aggregates.predictors_overview()

        if self.datamart.predictor_data is not None:
            tabs["predictors_overview"] = (
                self.datamart.aggregates.predictors_global_overview()
            )

        if predictor_binning and self.datamart.predictor_data is not None:
            columns = [
                pl.col("ModelID").cast(pl.Utf8),
                pl.col("PredictorName").cast(pl.Utf8),
                pl.col("GroupIndex").cast(pl.Int16),
                pl.col("Performance").cast(pl.Float32),
                pl.col("FeatureImportance").cast(pl.Float32),
                pl.col("ResponseCount").cast(pl.Int64),
                pl.col("Positives").cast(pl.Int64),
                pl.col("BinIndex").cast(pl.Int16),
                pl.col("TotalBins").cast(pl.Int16),
                pl.col("BinType").cast(pl.Utf8),
                pl.col("EntryType").cast(pl.Utf8),
                pl.col("BinSymbol").cast(pl.Utf8),
                pl.col("BinPositives").cast(pl.Int64),
                pl.col("BinResponseCount").cast(pl.Int64),
                pl.col("PredictorCategory").cast(pl.Utf8),
            ]
            subset_columns = [
                col
                for col in columns
                if col.meta.output_name()
                in self.datamart.predictor_data.collect_schema().names()
            ]
            tabs["predictor_binning"] = (
                self.datamart.aggregates.last(table="predictor_data")
                .filter(pl.col("PredictorName") != "Classifier")
                .select(subset_columns)
            ).sort(
                ["GroupIndex", "EntryType", "Performance"],
                descending=[False, True, True],
                nulls_last=True,
            )

        # Remove None values (tables that are not available)
        tabs = {k: v for k, v in tabs.items() if v is not None}

        if not tabs:  # pragma: no cover
            print("No data available to export.")
            return None, warning_messages

        try:
            with Workbook(
                name, options={"nan_inf_to_errors": True, "remove_timezone": True}
            ) as wb:
                # Enable ZIP64 extensions to handle large files
                wb.use_zip64()

                for tab, data in tabs.items():
                    data = data.with_columns(
                        pl.col(pl.List(pl.Categorical), pl.List(pl.Utf8))
                        .list.eval(pl.element().cast(pl.Utf8))
                        .list.join(", ")
                    )
                    data = data.collect()

                    # Check data size (with a multiplication factor for Excel XML overhead)
                    estimated_size_mb = data.estimated_size(unit="mb") * 2.5
                    if estimated_size_mb > ZIP_SIZE_LIMIT_MB:
                        warning_msg = (
                            f"The data for sheet '{tab}' is too large (estimated {estimated_size_mb:.1f} MB). "
                            f"This exceeds the recommended size limit for Excel files ({ZIP_SIZE_LIMIT_MB} MB). "
                            "This sheet will not be written to the Excel file. "
                            "Consider exporting this data to CSV format instead."
                        )
                        warning_messages.append(warning_msg)
                        print(warning_msg)
                        continue

                    if data.shape[0] > EXCEL_ROW_LIMIT:
                        warning_msg = (
                            f"The data for sheet '{tab}' exceeds Excel's row limit "
                            f"({data.shape[0]:,} rows > {EXCEL_ROW_LIMIT:,} rows). "
                            "This sheet will not be written to the Excel file. "
                            "Please filter your data before generating the Excel report."
                        )
                        warning_messages.append(warning_msg)
                        print(warning_msg)
                        continue
                    else:
                        data.write_excel(workbook=wb, worksheet=tab)
        except Exception as e:
            warning_msg = (
                f"Error creating Excel file: {str(e)}. Try exporting to CSV instead."
            )
            warning_messages.append(warning_msg)
            print(warning_msg)
            return None, warning_messages

        print(f"Data exported to {name}")
        return name, warning_messages
