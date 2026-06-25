"""I/O, logging, working-directory and version-check helpers."""

from __future__ import annotations

import datetime
import io
import logging
import tempfile
import zipfile
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from ._common import logger

if TYPE_CHECKING:
    import polars as pl
    from os import PathLike


def process_files_to_bytes(
    file_paths: list[str | Path],
    base_file_name: str | Path,
) -> tuple[bytes, str]:
    """Processes a list of file paths, returning file content as bytes and a corresponding file name.
    Useful for zipping muliple model reports and the byte object is used for downloading files in
    Streamlit app.

    This function handles three scenarios:
    1. Single file: Returns the file's content as bytes and the provided base file name.
    2. Multiple files: Creates a zip file containing all files, returns the zip file's content as bytes
       and a generated zip file name.
    3. No files: Returns empty bytes and an empty string.

    Parameters
    ----------
    file_paths : list[str | Path]
        A list of file paths to process. Can be empty, contain a single path, or multiple paths.
    base_file_name : str | Path
        The base name to use for the output file. For a single file, this name is returned as is.
        For multiple files, this is used as part of the generated zip file name.

    Returns
    -------
    tuple[bytes, str]
        A tuple containing:
        - bytes: The content of the single file or the created zip file, or empty bytes if no files.
        - str: The file name (either base_file_name or a generated zip file name), or an empty string if no files.

    """
    path_list: list[Path] = [Path(fp) for fp in file_paths]
    base_file_name = Path(base_file_name)

    if not path_list:
        return b"", ""

    if len(path_list) == 1:
        try:
            with path_list[0].open("rb") as file:
                return file.read(), base_file_name.name
        except OSError as e:
            logger.error(f"Error reading file {path_list[0]}: {e}")
            return b"", ""

    # Multiple files
    in_memory_zip = io.BytesIO()
    with zipfile.ZipFile(in_memory_zip, "w") as zipf:
        for file_path in path_list:
            try:
                zipf.write(
                    file_path,
                    file_path.name,
                    compress_type=zipfile.ZIP_DEFLATED,
                )
            except OSError as e:
                logger.error(f"Error adding file {file_path} to zip: {e}")

    time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    zip_file_name = f"{base_file_name.stem}_{time}.zip"
    in_memory_zip.seek(0)
    return in_memory_zip.getvalue(), zip_file_name


def get_latest_pdstools_version():
    import requests

    try:
        response = requests.get("https://pypi.org/pypi/pdstools/json", timeout=5)
        return response.json()["info"]["version"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.debug("Could not fetch latest pdstools version: %s", exc)
        return None


def setup_logger():
    """Return the ``pdstools`` logger and a log buffer it streams into.

    Targets the named ``pdstools`` logger rather than the root logger so we
    don't clobber the host application's logging config (Streamlit, Quarto,
    Jupyter, etc.). Idempotent: repeated calls return the same buffer
    instead of stacking new handlers, so re-running a notebook cell or
    bouncing a Streamlit page doesn't produce duplicated log lines.
    """
    logger = logging.getLogger("pdstools")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        existing_buffer = getattr(handler, "_pdstools_buffer", None)
        if existing_buffer is not None:
            return logger, existing_buffer
    log_buffer = StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler._pdstools_buffer = log_buffer
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, log_buffer


def create_working_and_temp_dir(
    name: str | None = None,
    working_dir: PathLike | None = None,
) -> tuple[Path, Path]:
    """Creates a working directory for saving files and a temp_dir"""
    # Create a temporary directory in working_dir
    working_dir = Path(working_dir) if working_dir else Path.cwd()
    working_dir.mkdir(parents=True, exist_ok=True)
    temp_dir_name = (
        tempfile.mkdtemp(prefix=f"tmp_{name}_", dir=working_dir)
        if name
        else tempfile.mkdtemp(prefix="tmp_", dir=working_dir)
    )
    return working_dir, Path(temp_dir_name)


# Full expected schema of the Databricks vw_gold_ml_models_predictions_summary view.
# When the view schema changes, update this set only.
_DATABRICKS_PREDICTION_COLUMNS = frozenset(
    {
        "PacID",
        "EnvironmentName",
        "Configuration",
        "AppliesToClass",
        "SnapshotDate",
        "Positives",
        "Negatives",
        "ResponseCount",
        "Performance",
        "ModelType",
    }
)

# Full expected schema of the Databricks vw_gold_ml_models_snapshots_summary view.
# When the view schema changes, update this set only.
_DATABRICKS_MODEL_SNAPSHOTS_COLUMNS = frozenset(
    {
        "PacID",
        "EnvironmentName",
        "ModelID",
        "Channel",
        "Direction",
        "Issue",
        "Group",
        "Name",
        "Treatment",
        "ExtraContextKeys",
        "Positives",
        "Negatives",
        "ResponseCount",
        "Performance",
        "SnapshotDate",
        "Configuration",
        "AppliesToClass",
        "ModelTechnique",
    }
)


def _validate_databricks_schema(
    df: pl.LazyFrame,
    schema: frozenset[str] | set[str],
    view_name: str,
    schema_const_name: str,
) -> pl.LazyFrame:
    df_cols = set(df.collect_schema().names())
    if missing := schema - df_cols:
        raise ValueError(f"Required columns missing from Databricks {view_name} view: {missing}")
    if extra := df_cols - schema:
        logger.warning(
            "Unexpected columns in Databricks %s view: %s. The view schema may have changed — update %s in pdstools.",
            view_name,
            extra,
            schema_const_name,
        )
    return df


def _validate_databricks_model_snapshots(df: pl.LazyFrame) -> pl.LazyFrame:
    """Validate the schema of the Databricks model snapshots view LazyFrame."""
    return _validate_databricks_schema(
        df, _DATABRICKS_MODEL_SNAPSHOTS_COLUMNS, "model snapshots", "_DATABRICKS_MODEL_SNAPSHOTS_COLUMNS"
    )


def _validate_databricks_predictions(df: pl.LazyFrame) -> pl.LazyFrame:
    """Validate the schema of the Databricks predictions view LazyFrame."""
    return _validate_databricks_schema(
        df, _DATABRICKS_PREDICTION_COLUMNS, "predictions", "_DATABRICKS_PREDICTION_COLUMNS"
    )


def _validate_databricks_rename_map(
    rename_map: dict[str, str],
    schema: frozenset[str] | set[str],
    view_name: str,
    schema_const_name: str,
) -> None:
    """Validate that a Databricks rename map only uses known source columns."""

    invalid_keys = set(rename_map) - schema
    if invalid_keys:
        raise ValueError(
            f"Databricks {view_name} rename map contains keys not present in {schema_const_name}: {sorted(invalid_keys)}"
        )
