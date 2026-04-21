"""Report-generation namespace for :class:`ADMDatamart`.

Exposed via the ``dm.generate`` namespace facade. Wraps Quarto-driven
HealthCheck and ModelReport rendering plus an Excel export.
"""

from __future__ import annotations

__all__ = ["ReportOptions", "Reports"]

import logging
import shutil
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from typing_extensions import TypedDict, Unpack

from ..utils import cdh_utils
from ..utils.namespaces import LazyNamespace
from ..utils.report_utils import (
    bundle_quarto_resources,
    copy_quarto_file,
    get_output_filename,
    run_quarto,
    serialize_query,
)
from ..utils.types import QUERY

if TYPE_CHECKING:
    from ..prediction.Prediction import Prediction
    from .ADMDatamart import ADMDatamart

logger = logging.getLogger(__name__)

OutputType = Literal["html", "pdf"]


class ReportOptions(TypedDict, total=False):
    """Shared rendering/output options for report-generating methods.

    Passed via ``**options`` to :meth:`Reports.model_reports` and
    :meth:`Reports.health_check`. These eight keys are shared between
    both methods; each method also takes its own method-specific kwargs
    as normal parameters. All ``ReportOptions`` keys are optional —
    per-method defaults apply when a key is omitted.

    Keys
    ----
    title : str
        Title shown in the report.
    subtitle : str
        Subtitle shown under the title.
    disclaimer : str
        Disclaimer text included in the report.
    output_dir : str or path-like or None
        Directory for the output file. If None, uses the current working
        directory.
    output_type : {"html", "pdf"}
        Output format. Defaults to ``"html"``.
    qmd_file : str or path-like or None
        Path to a custom Quarto template. If None, the built-in template
        is used.
    full_embed : bool
        When True, fully embed all JavaScript libraries (Plotly, itables,
        etc.) into the HTML output (larger file, requires esbuild). When
        False (default), load JavaScript libraries from CDN. See issue
        #620.
    keep_temp_files : bool
        If True, the temporary working directory is preserved after the
        report is generated. Defaults to False.
    """

    title: str
    subtitle: str
    disclaimer: str
    output_dir: str | PathLike[str] | None
    output_type: OutputType
    qmd_file: str | PathLike[str] | None
    full_embed: bool
    keep_temp_files: bool


_VALID_REPORT_OPTION_KEYS: frozenset[str] = frozenset(ReportOptions.__annotations__)


def _resolve_report_options(defaults: ReportOptions, options: ReportOptions) -> dict:
    """Merge per-method defaults with caller-supplied options.

    Raises ``TypeError`` on unknown keys so typos surface at runtime
    (since ``TypedDict`` is only checked statically).
    """
    unknown = set(options) - _VALID_REPORT_OPTION_KEYS
    if unknown:
        raise TypeError(
            f"Unknown report option(s): {sorted(unknown)}. Valid options: {sorted(_VALID_REPORT_OPTION_KEYS)}"
        )
    return {**defaults, **options}


_MODEL_REPORT_DEFAULTS: ReportOptions = {
    "title": "ADM Model Report",
    "subtitle": "",
    "disclaimer": "",
    "output_dir": None,
    "output_type": "html",
    "qmd_file": None,
    "full_embed": False,
    "keep_temp_files": False,
}

_HEALTH_CHECK_DEFAULTS: ReportOptions = {
    "title": "ADM Model Overview",
    "subtitle": "",
    "disclaimer": "",
    "output_dir": None,
    "output_type": "html",
    "qmd_file": None,
    "full_embed": False,
    "keep_temp_files": False,
}


class Reports(LazyNamespace):
    """Report generation namespace attached to :class:`ADMDatamart` as ``dm.generate``."""

    dependencies = ["yaml"]
    dependency_group = "healthcheck"

    def __init__(self, datamart: ADMDatamart):
        self.datamart = datamart
        super().__init__()

    def model_reports(
        self,
        model_ids: str | list[str],
        *,
        name: str | None = None,
        only_active_predictors: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
        model_file_path: str | PathLike[str] | None = None,
        predictor_file_path: str | PathLike[str] | None = None,
        **options: Unpack[ReportOptions],
    ) -> Path:
        """Generate model reports for Naive Bayes ADM models.

        Parameters
        ----------
        model_ids : str or list of str
            The model ID (or list of model IDs) to generate reports for.
        name : str, optional
            Base file name of the report.
        only_active_predictors : bool, default=True
            Whether to only include active predictor details.
        progress_callback : callable, optional
            Function called as ``progress_callback(current, total)`` after each model
            report is rendered. Used by the Streamlit app.
        model_file_path : str or path-like, optional
            Path to an existing model-data file. If provided, skips re-exporting the
            datamart's model data.
        predictor_file_path : str or path-like, optional
            Path to an existing predictor-data file. If provided, skips re-exporting
            the datamart's predictor data.
        **options : Unpack[ReportOptions]
            Shared rendering/output options. See :class:`ReportOptions` for the full
            list (``title``, ``subtitle``, ``disclaimer``, ``output_dir``,
            ``output_type``, ``qmd_file``, ``full_embed``, ``keep_temp_files``).

        Returns
        -------
        Path
            The path to the generated report file (or zip if multiple model IDs).

        Raises
        ------
        ValueError
            If ``model_ids`` is empty or report generation fails.
        FileNotFoundError
            If required input files are not found.
        subprocess.SubprocessError
            If the Quarto subprocess fails.
        """
        opts = _resolve_report_options(_MODEL_REPORT_DEFAULTS, options)
        title = opts["title"]
        subtitle = opts["subtitle"]
        disclaimer = opts["disclaimer"]
        output_dir = opts["output_dir"]
        output_type = opts["output_type"]
        qmd_file = opts["qmd_file"]
        full_embed = opts["full_embed"]
        keep_temp_files = opts["keep_temp_files"]

        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if not model_ids or not all(isinstance(i, str) for i in model_ids):
            raise ValueError("No valid model IDs")

        output_dir, temp_dir = cdh_utils.create_working_and_temp_dir(name, output_dir)

        try:
            if qmd_file is None:
                qmd_filename = "ModelReport.qmd"
                copy_quarto_file(qmd_filename, temp_dir)
            else:
                qmd_filename = Path(qmd_file).name
                shutil.copy(qmd_file, temp_dir / qmd_filename)

            # Cache datamart frames to disk only when no pre-existing path was supplied.
            if (model_file_path is None and self.datamart.model_data is not None) or (
                predictor_file_path is None and self.datamart.predictor_data is not None
            ):
                model_file_path, predictor_file_path = self.datamart.save_data(
                    temp_dir,
                    selected_model_ids=model_ids,
                )

            output_file_paths: list[str | Path] = []
            output_path = temp_dir
            for i, model_id in enumerate(model_ids):
                output_filename = get_output_filename(
                    name,
                    "ModelReport",
                    model_id,
                    output_type,
                )
                run_quarto(
                    qmd_file=qmd_filename,
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
                    full_embed=full_embed,
                )
                output_path = temp_dir / output_filename
                if not output_path.exists():
                    if model_file_path is not None:
                        logger.info('model_file_path = "%s"', model_file_path)
                    if predictor_file_path is not None:
                        logger.info('predictor_file_path = "%s"', predictor_file_path)
                    logger.info('model_id = "%s"', model_id)
                    logger.info("output_path = %s", output_path)
                    raise ValueError(f"Failed to write the report: {output_filename}")
                output_path = bundle_quarto_resources(output_path)
                output_file_paths.append(output_path)
                if progress_callback:
                    progress_callback(i + 1, len(model_ids))

            file_data, file_name = cdh_utils.process_files_to_bytes(
                output_file_paths,
                base_file_name=output_path,
            )
            final_path = output_dir / file_name
            with open(final_path, "wb") as f:
                f.write(file_data)
            if not final_path.exists():
                raise ValueError(f"Failed to generate report: {file_name}")

            return final_path

        except Exception as e:
            logger.error(e)
            raise
        finally:
            if not keep_temp_files and temp_dir.exists() and temp_dir.is_dir():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def health_check(
        self,
        name: str | None = None,
        *,
        query: QUERY | None = None,
        prediction: Prediction | None = None,
        model_file_path: str | PathLike[str] | None = None,
        predictor_file_path: str | PathLike[str] | None = None,
        prediction_file_path: str | PathLike[str] | None = None,
        **options: Unpack[ReportOptions],
    ) -> Path:
        """Generate the ADM Health Check report.

        Optionally includes predictor and prediction sections when the corresponding
        data is available on the datamart or supplied via ``prediction``.

        Parameters
        ----------
        name : str, optional
            Base file name of the report.
        query : QUERY, optional
            Extra filter applied to the datamart data before rendering.
        prediction : Prediction, optional
            Prediction object to include in the report. If provided without
            ``prediction_file_path``, prediction data is cached to a temporary file.
        model_file_path : str or path-like, optional
            Path to an existing model-data file. If provided, skips re-exporting the
            datamart's model data.
        predictor_file_path : str or path-like, optional
            Path to an existing predictor-data file. If provided, skips re-exporting
            the datamart's predictor data.
        prediction_file_path : str or path-like, optional
            Path to an existing prediction-data file. If neither this nor ``prediction``
            is provided, the prediction section is omitted.
        **options : Unpack[ReportOptions]
            Shared rendering/output options. See :class:`ReportOptions` for the full
            list (``title``, ``subtitle``, ``disclaimer``, ``output_dir``,
            ``output_type``, ``qmd_file``, ``full_embed``, ``keep_temp_files``).

        Returns
        -------
        Path
            The path to the generated report file.

        Raises
        ------
        ValueError
            If report generation fails.
        FileNotFoundError
            If required input files are not found.
        subprocess.SubprocessError
            If the Quarto subprocess fails.
        """
        opts = _resolve_report_options(_HEALTH_CHECK_DEFAULTS, options)
        title = opts["title"]
        subtitle = opts["subtitle"]
        disclaimer = opts["disclaimer"]
        output_dir = opts["output_dir"]
        output_type = opts["output_type"]
        qmd_file = opts["qmd_file"]
        full_embed = opts["full_embed"]
        keep_temp_files = opts["keep_temp_files"]

        output_dir, temp_dir = cdh_utils.create_working_and_temp_dir(name, output_dir)
        try:
            if qmd_file is None:
                qmd_filename = "HealthCheck.qmd"
                copy_quarto_file(qmd_filename, temp_dir)
            else:
                qmd_filename = Path(qmd_file).name
                shutil.copy(qmd_file, temp_dir / qmd_filename)

            output_filename = get_output_filename(
                name,
                "HealthCheck",
                None,
                output_type,
            )

            if (model_file_path is None and self.datamart.model_data is not None) or (
                predictor_file_path is None and self.datamart.predictor_data is not None
            ):
                model_file_path, predictor_file_path = self.datamart.save_data(temp_dir)

            if prediction_file_path is None and prediction is not None:
                prediction_file_path = prediction.save_data(temp_dir)

            run_quarto(
                qmd_file=qmd_filename,
                output_filename=output_filename,
                output_type=output_type,
                params={
                    "report_type": "HealthCheck",
                    "model_file_path": str(model_file_path) if model_file_path is not None else "",
                    "predictor_file_path": str(predictor_file_path) if predictor_file_path is not None else "",
                    "prediction_file_path": str(prediction_file_path) if prediction_file_path is not None else "",
                    "query": serialize_query(query),
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
                full_embed=full_embed,
            )

            output_path = temp_dir / output_filename
            if not output_path.exists():
                if model_file_path is not None:
                    logger.info('model_file_path = "%s"', model_file_path)
                if predictor_file_path is not None:
                    logger.info('predictor_file_path = "%s"', predictor_file_path)
                if prediction_file_path is not None:
                    logger.info('prediction_file_path = "%s"', prediction_file_path)
                logger.info("output_path = %s", output_path)
                raise ValueError(f"Failed to generate report: {output_filename}")

            output_path = bundle_quarto_resources(output_path)

            final_path = output_dir / output_path.name
            shutil.copy(output_path, final_path)

            return final_path

        finally:
            if not keep_temp_files and temp_dir.exists() and temp_dir.is_dir():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def excel_report(
        self,
        name: str | PathLike[str] = Path("Tables.xlsx"),
        *,
        predictor_binning: bool = False,
    ) -> tuple[Path | None, list[str]]:
        """Export raw datamart tables to an Excel workbook.

        Writes the last snapshots of model data, predictor summaries, and optionally
        predictor binning data to separate sheets. Sheets that are unavailable or
        exceed Excel's row / size limits are skipped with a warning rather than
        failing the whole export.

        Parameters
        ----------
        name : str or path-like, default=Path("Tables.xlsx")
            Path where the Excel file will be written.
        predictor_binning : bool, default=False
            If True, include the (potentially large) predictor binning sheet.

        Returns
        -------
        tuple of (Path or None, list of str)
            The path to the created Excel file (``None`` if no data was available),
            and a list of warning messages (empty if nothing was skipped).
        """
        from xlsxwriter import Workbook

        EXCEL_ROW_LIMIT = 1048576
        # Standard ZIP format limit is 4 GB but 3 GB would already crash most laptops.
        ZIP_SIZE_LIMIT_MB = 3000
        warning_messages: list[str] = []

        output_path = Path(name)
        model_data = self.datamart.model_data
        predictor_data = self.datamart.predictor_data

        tabs: dict[str, pl.LazyFrame | None] = {}

        if model_data is not None:
            model_columns = model_data.collect_schema().names()
            tabs["adm_models"] = (
                self.datamart.aggregates.last(table="model_data")
                .with_columns(
                    pl.col("ResponseCount", "Positives", "Negatives").cast(pl.Int64),
                )
                .select(
                    ["ModelID"]
                    + self.datamart.context_keys
                    + [col for col in model_columns if col != "ModelID" and col not in self.datamart.context_keys],
                )
                .sort(self.datamart.context_keys)
            )

        if predictor_data is not None:
            tabs["predictors_detail"] = self.datamart.aggregates.predictors_overview()
            tabs["predictors_overview"] = self.datamart.aggregates.predictors_global_overview()

        if predictor_binning and predictor_data is not None:
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
            available = predictor_data.collect_schema().names()
            subset_columns = [c for c in columns if c.meta.output_name() in available]
            tabs["predictor_binning"] = (
                self.datamart.aggregates.last(table="predictor_data")
                .filter(pl.col("PredictorName") != "Classifier")
                .select(subset_columns)
                .sort(
                    ["GroupIndex", "EntryType", "Performance"],
                    descending=[False, True, True],
                    nulls_last=True,
                )
            )

        present_tabs: dict[str, pl.LazyFrame] = {k: v for k, v in tabs.items() if v is not None}

        if not present_tabs:  # pragma: no cover
            logger.warning("No data available to export.")
            return None, warning_messages

        try:
            with Workbook(
                output_path,
                options={"nan_inf_to_errors": True, "remove_timezone": True},
            ) as wb:
                # Enable ZIP64 extensions to handle large files.
                wb.use_zip64()

                for tab, data in present_tabs.items():
                    data = data.with_columns(
                        pl.col(pl.List(pl.Categorical), pl.List(pl.Utf8))
                        .list.eval(pl.element().cast(pl.Utf8))
                        .list.join(", "),
                    )
                    collected = data.collect()

                    # Multiplier accounts for Excel XML overhead vs in-memory size.
                    estimated_size_mb = collected.estimated_size(unit="mb") * 2.5
                    if estimated_size_mb > ZIP_SIZE_LIMIT_MB:
                        warning_msg = (
                            f"The data for sheet '{tab}' is too large (estimated {estimated_size_mb:.1f} MB). "
                            f"This exceeds the recommended size limit for Excel files ({ZIP_SIZE_LIMIT_MB} MB). "
                            "This sheet will not be written to the Excel file. "
                            "Consider exporting this data to CSV format instead."
                        )
                        warning_messages.append(warning_msg)
                        logger.warning(warning_msg)
                        continue

                    if collected.shape[0] > EXCEL_ROW_LIMIT:
                        warning_msg = (
                            f"The data for sheet '{tab}' exceeds Excel's row limit "
                            f"({collected.shape[0]:,} rows > {EXCEL_ROW_LIMIT:,} rows). "
                            "This sheet will not be written to the Excel file. "
                            "Please filter your data before generating the Excel report."
                        )
                        warning_messages.append(warning_msg)
                        logger.warning(warning_msg)
                        continue
                    collected.write_excel(workbook=wb, worksheet=tab)
        except Exception as e:
            warning_msg = f"Error creating Excel file: {e!s}. Try exporting to CSV instead."
            warning_messages.append(warning_msg)
            logger.warning(warning_msg)
            return None, warning_messages

        logger.info("Data exported to %s", output_path)
        return output_path, warning_messages
