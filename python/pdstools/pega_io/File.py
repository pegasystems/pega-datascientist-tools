import gzip
import logging
import os
import pathlib
import re
import shutil
import tarfile
import tempfile
import warnings
import zipfile
from collections.abc import Iterable
from datetime import datetime, timezone
from glob import glob
from io import BytesIO
from pathlib import Path
from typing import Literal, cast, overload

import polars as pl
import polars.selectors as cs

from ..utils.cdh_utils import from_prpc_date_time

logger = logging.getLogger(__name__)


# Extensions that read_data knows how to handle.
_SUPPORTED_EXTENSIONS: set[str] = {
    ".parquet",
    ".csv",
    ".arrow",
    ".feather",  # Alias for .arrow/.ipc
    ".ipc",  # Arrow IPC format
    ".ndjson",
    ".jsonl",
    ".json",
    ".xlsx",
    ".xls",
    ".zip",
    ".tar",
    ".tgz",
    ".gz",
}


def _read_excel(path, **kwargs) -> pl.DataFrame:
    """Read an Excel file via polars, surfacing the optional-dependency requirement.

    Polars dispatches Excel reading to ``fastexcel``/``calamine`` and raises a
    bare :class:`ModuleNotFoundError` when that backend is missing.  We catch it
    here and re-raise as :class:`MissingDependenciesException` so the rest of
    the codebase can rely on a single, package-manager-neutral error path.
    """
    try:
        return pl.read_excel(path, **kwargs)
    except ModuleNotFoundError:
        from ..utils.namespaces import MissingDependenciesException

        raise MissingDependenciesException(
            ["fastexcel"],
            namespace="pega_io.read_data (Excel support)",
        ) from None


def _is_artifact(name: str) -> bool:
    """Return True for OS-generated junk entries (macOS, Windows, etc.)."""
    return name.startswith("__MACOSX") or name.startswith("._") or name in {".DS_Store", "Thumbs.db", "desktop.ini"}


def _clean_artifacts(directory: str) -> None:
    """Remove OS-generated artifact files and directories after archive extraction.

    Polars glob patterns (e.g. ``**/*.parquet``) cannot skip hidden files or
    ``__MACOSX`` resource-fork directories, so we delete them from the
    extracted tree before scanning.
    """
    root = Path(directory)
    # Remove __MACOSX directories first (contains bulk of macOS resource forks)
    for macosx_dir in root.rglob("__MACOSX"):
        if macosx_dir.is_dir():
            shutil.rmtree(macosx_dir, ignore_errors=True)
    # Remove individual artifact files (._*, .DS_Store, Thumbs.db, etc.)
    for p in list(root.rglob(".*")):
        if p.is_file() and _is_artifact(p.name):
            p.unlink(missing_ok=True)
    for name in ("Thumbs.db", "desktop.ini"):
        for p in root.rglob(name):
            if p.is_file():
                p.unlink(missing_ok=True)


def _extract_tar(archive_path: Path) -> str:
    """Extract a tar archive to a temporary directory and return the path."""
    tmp_dir = tempfile.mkdtemp(prefix="pdstools_tar_")
    # lgtm [py/path-injection]
    # CodeQL suppression: archive_path is user-specified - expected for data reading library
    with tarfile.open(archive_path, mode="r:*") as tf:
        tf.extractall(tmp_dir, filter="data")
    return tmp_dir


def _extract_zip(archive_path: Path) -> str:
    """Extract a zip archive to a temporary directory and return the path."""
    tmp_dir = tempfile.mkdtemp(prefix="pdstools_zip_")
    # lgtm [py/path-injection]
    # CodeQL suppression: archive_path is user-specified - expected for data reading library
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(tmp_dir)
    return tmp_dir


def _read_from_bytesio(file: BytesIO, extension: str) -> pl.LazyFrame:
    """Read data from a BytesIO object (e.g., from Streamlit file upload).

    Parameters
    ----------
    file : BytesIO
        The BytesIO object containing file data.
    extension : str
        The file extension (e.g., '.csv', '.json', '.zip', '.gz').

    Returns
    -------
    pl.LazyFrame
        Lazy DataFrame ready for processing.
    """
    file.seek(0)  # Ensure we're at the start

    # Handle .gz decompression
    if extension == ".gz":
        # Decompress and determine the underlying format
        decompressed = BytesIO(gzip.decompress(file.read()))
        if hasattr(file, "name"):
            # Extract the extension before .gz (e.g., file.json.gz → .json)
            base_name = os.path.splitext(file.name)[0]
            extension = os.path.splitext(base_name)[1] or ".json"  # Default to .json
        else:
            extension = ".json"  # Default assumption
        file = decompressed
        file.seek(0)

    # Handle .zip archives (extract data.json)
    if extension == ".zip":
        with zipfile.ZipFile(file, "r") as zf:
            if "data.json" in zf.namelist():
                with zf.open("data.json") as data_file:
                    return pl.read_ndjson(BytesIO(data_file.read())).lazy()
            # Try to find data.json in subdirectories
            for name in zf.namelist():
                if name.endswith("data.json"):
                    with zf.open(name) as data_file:
                        return pl.read_ndjson(BytesIO(data_file.read())).lazy()
        raise FileNotFoundError("Cannot find 'data.json' in zip archive")

    # Read based on extension
    if extension == ".parquet":
        return pl.read_parquet(file).lazy()
    elif extension == ".csv":
        return pl.read_csv(file).lazy()
    elif extension in {".arrow", ".feather", ".ipc"}:
        return pl.read_ipc(file).lazy()
    elif extension in {".json", ".ndjson", ".jsonl"}:
        return pl.read_ndjson(file).lazy()
    else:
        raise ValueError(f"Unsupported file type for BytesIO: {extension}")


def read_data(path: str | Path | BytesIO) -> pl.LazyFrame:
    """Read data from various file formats and sources.

    Supports multiple formats: parquet, csv, arrow, feather, ndjson, json, xlsx, xls, zip, tar, tar.gz, tgz, gz.
    Handles both individual files and directories (including Hive-partitioned structures).
    Archives (zip, tar) are automatically extracted to temporary directories.
    Gzip files (.gz) are automatically decompressed.

    Parameters
    ----------
    path : str, Path, or BytesIO
        Path to a data file, archive, directory, or BytesIO object.
        When using BytesIO (e.g., from Streamlit file uploads), the object must have
        a 'name' attribute indicating the file extension.
        Supported formats:
        - Parquet files or directories
        - CSV files
        - Arrow/IPC/Feather files
        - NDJSON/JSONL files
        - Excel files (.xlsx, .xls — requires the optional ``fastexcel`` package)
        - GZIP compressed files (.gz, .json.gz, .csv.gz, etc.)
        - ZIP archives including Pega Dataset Export format (extracted automatically)
        - TAR archives including .tar.gz and .tgz (extracted automatically)
        - Hive-partitioned directories (scanned recursively)

    Returns
    -------
    pl.LazyFrame
        Lazy DataFrame ready for processing. Use `.collect()` to materialize.

    Raises
    ------
    ValueError
        If no supported data files are found in a directory, or if the file
        type is not supported.

    Examples
    --------
    Read a parquet file:

    >>> df = read_data("data.parquet")

    Read from a ZIP archive:

    >>> df = read_data("export.zip")

    Read from a TAR archive:

    >>> df = read_data("export.tar.gz")

    Read from a Hive-partitioned directory:

    >>> df = read_data("pxDecisionTime_day=08/")

    Read a Pega Dataset Export file:

    >>> df = read_data("Data-Decision-ADM-ModelSnapshot_pyModelSnapshots_20210101T010000_GMT.zip")

    Read a gzip-compressed file:

    >>> df = read_data("export.json.gz")
    >>> df = read_data("data.csv.gz")

    Read from a BytesIO object (e.g., Streamlit upload):

    >>> from io import BytesIO
    >>> uploaded_file = ...  # BytesIO with 'name' attribute
    >>> df = read_data(uploaded_file)

    Read a Feather file:

    >>> df = read_data("data.feather")

    Notes
    -----
    **Pega Dataset Export Support:**
    This function fully supports Pega Dataset Export format (e.g., Data-Decision-ADM-*.zip,
    Data-DM-*.zip). These are zip archives containing a data.json file (NDJSON format) and
    optionally a META-INF/MANIFEST.mf metadata file. The function automatically extracts
    and reads the data.json file.

    **Other Notes:**
    - Archives are extracted to temporary directories with automatic cleanup
    - OS artifacts (__MACOSX, .DS_Store, ._* files) are automatically removed
    - For directories, the first supported file type found determines the format
    """
    # Handle BytesIO objects (e.g., from Streamlit file uploads)
    if isinstance(path, BytesIO):
        if not hasattr(path, "name"):
            raise ValueError("BytesIO object must have a 'name' attribute indicating file extension")
        _, extension = os.path.splitext(path.name)
        return _read_from_bytesio(path, extension)

    # lgtm [py/path-injection]
    # CodeQL suppression: User-controlled paths are expected in a data reading library.
    # Users explicitly specify which files/directories to read - this is the intended
    # functionality, not a security vulnerability. This library is designed for use in
    # trusted environments (data scientists' local machines, internal systems).
    original_path = Path(path)  # save the original path
    extension = None  # Initialize extension to None
    if original_path.is_dir():
        # It's a directory, possibly hive-partitioned at arbitrary depth.
        # Find the first supported data file, skipping hidden/OS-artifact entries.
        for dirpath, dirs, files in os.walk(str(original_path)):
            dirs[:] = [d for d in dirs if not d.startswith(".") and not _is_artifact(d)]
            for file in sorted(files):
                if file.startswith(".") or _is_artifact(file):
                    continue
                ext = Path(file).suffix
                if ext in _SUPPORTED_EXTENSIONS:
                    extension = ext
                    break
            if extension:
                break
        # Use a recursive glob — works regardless of partition depth.
        path = original_path / f"**/*{extension}"
    else:
        # It's a file, so we read based on the extension
        # For tar files, normalize multi-part extensions (.tar.gz → .tar) for consistent handling
        name_lower = original_path.name.lower()
        if name_lower.endswith((".tar", ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz")):
            extension = ".tar"
        else:
            extension = original_path.suffix

    # Handle compressed and archive files early
    # This includes:
    # - GZIP files (.gz, .json.gz, .csv.gz, etc.) - decompress and read
    # - TAR archives (.tar, .tar.gz, .tgz, etc.) - extract and read
    # - ZIP archives including Pega Dataset Export format - extract and read
    # lgtm [py/path-injection]
    # CodeQL suppression: User-controlled paths are expected in a data reading library.
    # Users explicitly specify which files/directories to read - this is the intended
    # functionality, not a security vulnerability.
    if not original_path.is_dir():
        if extension == ".gz":
            # Decompress gzip file and read the underlying format
            # lgtm [py/path-injection]
            # CodeQL suppression: User specifies which gzip file to decompress - this is
            # expected library functionality for reading compressed data files.
            with gzip.open(original_path, "rb") as gz_file:
                decompressed = BytesIO(gz_file.read())
                # Determine the underlying format from the base name
                base_name = original_path.stem  # e.g., "file.json.gz" → "file.json"
                inner_ext = os.path.splitext(base_name)[1] or ".json"  # Default to .json
                if hasattr(decompressed, "name"):
                    decompressed.name = str(original_path)
                else:
                    # Create a fake name attribute for extension detection
                    object.__setattr__(decompressed, "name", base_name)
                return _read_from_bytesio(decompressed, inner_ext)
        elif extension == ".tar":
            tmp_dir = _extract_tar(original_path)
            _clean_artifacts(tmp_dir)
            return read_data(tmp_dir)
        elif extension == ".zip":
            tmp_dir = _extract_zip(original_path)
            _clean_artifacts(tmp_dir)
            return read_data(tmp_dir)

    if extension == ".parquet":
        df = pl.scan_parquet(path)
    elif extension == ".csv":
        df = pl.scan_csv(path)
    elif extension in {".arrow", ".feather", ".ipc"}:
        df = pl.scan_ipc(path)
    elif extension in {".ndjson", ".jsonl", ".json"}:
        df = pl.scan_ndjson(path)
    elif extension in {".xlsx", ".xls"}:
        df = _read_excel(path).lazy()
    elif extension is None:
        raise ValueError("No data files found in directory")
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    return df


def read_ds_export(
    filename: str | os.PathLike | BytesIO,
    path: str | os.PathLike = ".",
    *,
    infer_schema_length: int = 10000,
    separator: str = ",",
    ignore_errors: bool = False,
) -> pl.LazyFrame | None:
    """Read Pega dataset exports with additional capabilities.

    Extends :func:`read_data` with:

    - Smart file finding: accepts ``"model_data"`` or ``"predictor_data"``
      and searches for matching files (ADM-specific).
    - URL downloads: fetches remote files when local paths are not found
      (useful for demos and examples).
    - Schema overrides: applies Pega-specific type corrections (e.g.
      ``PYMODELID`` as string).

    For simple file reading without these features, use :func:`read_data`.

    Parameters
    ----------
    filename : str, os.PathLike, or BytesIO
        File identifier. May be a full file path, a generic name like
        ``"model_data"`` / ``"predictor_data"`` (triggers smart search),
        or a :class:`io.BytesIO` object (delegates to :func:`read_data`).
    path : str or os.PathLike, default='.'
        Directory to search for files (ignored for BytesIO or full paths).
    infer_schema_length : int, keyword-only, default=10000
        Rows to scan for schema inference (CSV/JSON).
    separator : str, keyword-only, default=","
        CSV delimiter.
    ignore_errors : bool, keyword-only, default=False
        Whether to continue on parse errors (CSV).

    Returns
    -------
    pl.LazyFrame or None
        Lazy dataframe, or ``None`` if the file could not be located.

    Examples
    --------
    >>> df = read_ds_export("model_data", path="data/ADMData")
    >>> df = read_ds_export("ModelSnapshot_20210101.json", path="data")
    >>> df = read_ds_export(
    ...     "ModelSnapshot.zip", path="https://example.com/exports"
    ... )
    >>> df = read_ds_export("export.csv", infer_schema_length=200000)
    """
    file: str | BytesIO | None
    # If the data is a BytesIO object, such as an uploaded file
    # in certain webapps, delegate directly to read_data.
    if isinstance(filename, BytesIO):
        logger.debug("Filename is of type BytesIO, delegating to read_data")
        return read_data(filename)

    filename_str = os.fspath(filename)
    path_str = os.fspath(path)

    # ADM-specific: Smart file finding for model_data/predictor_data patterns.
    _TARGET_NAMES = {"model_data", "predictor_data", "value_finder", "prediction_data"}
    if os.path.isfile(filename_str):
        file = filename_str
    elif os.path.isfile(os.path.join(path_str, filename_str)):
        logger.debug("File found in directory")
        file = os.path.join(path_str, filename_str)
    elif filename_str in _TARGET_NAMES:
        logger.debug("Scanning for latest %s file", filename_str)
        file = get_latest_file(path_str, filename_str)
    else:
        file = None

    extension: str | None = None

    # ADM-specific: URL download support.  If we can't find the file
    # locally, try treating ``path/filename`` as a URL.
    if file is None:
        logger.debug("Could not find file in directory, checking if URL")
        url = f"{path_str}/{filename_str}"

        try:
            import requests  # type: ignore[import-untyped]  # requests has no PEP 561 stubs

            response = requests.get(url)
            logger.info("Remote fetch %s → HTTP %s", url, response.status_code)
            if response.status_code == 200:
                logger.debug("File found online, importing and parsing to BytesIO")
                buffer = BytesIO(response.content)
                _, extension = os.path.splitext(filename_str)
                return _import_file(
                    buffer,
                    extension,
                    infer_schema_length=infer_schema_length,
                    separator=separator,
                    ignore_errors=ignore_errors,
                )
            raise FileNotFoundError(
                f"Could not find '{filename_str}' locally in '{path_str}', "
                f"and remote fetch from {url} returned HTTP {response.status_code}."
            )

        except ImportError:
            warnings.warn(
                "Unable to import `requests`, so not able to check for remote files. "
                "If you're trying to read in a file from the internet (or, for "
                "instance, using the built-in cdh_sample method), install the "
                "'requests' package.",
                ImportWarning,
                stacklevel=2,
            )
            return None

        except requests.exceptions.SSLError:
            warnings.warn(
                "SSL error during HTTP request — likely a certificate-bundle "
                "issue. See https://stackoverflow.com/a/70495761 .",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        except FileNotFoundError:
            raise

        except Exception as exc:
            logger.info("File not found: %s/%s (%s)", path_str, filename_str, exc)
            return None

    if not isinstance(file, BytesIO):
        _, extension = os.path.splitext(file)

    return _import_file(
        file,
        extension or "",
        infer_schema_length=infer_schema_length,
        separator=separator,
        ignore_errors=ignore_errors,
    )


def _fill_context_field_nulls(df: pl.LazyFrame) -> pl.LazyFrame:
    """Fill nulls in context fields to prevent issues in downstream operations.

    Context fields (Channel, Direction, Issue, Group, Name) often have nulls in
    source data which can cause errors in group_by, transpose, and concat_str
    operations. This function fills nulls with "Unknown" to ensure these operations
    work correctly.

    Note: Treatment is intentionally NOT filled because null Treatment has semantic
    meaning (no treatment variation exists for that action).

    Parameters
    ----------
    df : pl.LazyFrame
        Input dataframe

    Returns
    -------
    pl.LazyFrame
        Dataframe with nulls filled in context fields

    """
    context_fields = ["Channel", "Direction", "Issue", "Group", "Name"]

    # Only fill nulls for columns that exist in the dataframe
    existing_context_fields = [col for col in context_fields if col in df.collect_schema().names()]

    if existing_context_fields:
        return df.with_columns([pl.col(col).fill_null("Unknown") for col in existing_context_fields])

    return df


def _import_file(
    file: str | BytesIO,
    extension: str,
    *,
    infer_schema_length: int = 10000,
    separator: str = ",",
    ignore_errors: bool = False,
) -> pl.LazyFrame:
    """Import a file with Pega-specific schema handling.

    Applies ADM-specific type corrections and schema overrides during import.
    Used internally by :func:`read_ds_export`.

    Parameters
    ----------
    file : str or BytesIO
        File path or BytesIO object.
    extension : str
        File extension (e.g., ``.csv``, ``.json``, ``.parquet``).
    infer_schema_length : int, keyword-only, default=10000
        Rows to scan for schema inference (CSV/JSON).
    separator : str, keyword-only, default=","
        CSV delimiter.
    ignore_errors : bool, keyword-only, default=False
        Whether to continue on parse errors (CSV).

    Returns
    -------
    pl.LazyFrame
        Lazy dataframe with schema corrections applied.
    """
    if extension == ".zip":
        logger.debug("Zip file found, extracting data.json to BytesIO.")
        file, extension = read_zipped_file(file)
    elif extension == ".gz":
        if isinstance(file, str):
            extension = os.path.splitext(os.path.splitext(file)[0])[1]
            with gzip.open(file, "rb") as gz:
                file = BytesIO(gz.read())
        else:
            extension = os.path.splitext(os.path.splitext(file.name)[0])[1] if hasattr(file, "name") else ""
            file.seek(0)
            file = BytesIO(gzip.decompress(file.read()))

    if extension == ".csv":
        csv_opts = dict(
            separator=separator,
            infer_schema_length=infer_schema_length,
            null_values=["", "NA", "N/A", "NULL"],
            schema_overrides={"PYMODELID": pl.Utf8},
            try_parse_dates=True,
            ignore_errors=ignore_errors,
        )
        if isinstance(file, BytesIO):
            df = pl.read_csv(file, **csv_opts).lazy()
        else:
            df = pl.scan_csv(file, **csv_opts)
        return _fill_context_field_nulls(df)

    if extension == ".json":
        try:
            df = pl.scan_ndjson(file, infer_schema_length=infer_schema_length)
        except (pl.exceptions.ComputeError, pl.exceptions.SchemaError, OSError) as exc:  # pragma: no cover
            logger.debug("scan_ndjson failed for %s: %s", file, exc)
            try:
                df = pl.read_json(file).lazy()
            except (pl.exceptions.ComputeError, pl.exceptions.SchemaError, ValueError) as exc:
                logger.debug("read_json fallback failed for %s: %s", file, exc)
                import json

                if isinstance(file, BytesIO):
                    file.seek(0)
                    raw = file.read().decode("utf-8")
                else:
                    with open(file) as f:
                        raw = f.read()
                df = pl.from_dicts(json.loads(raw)["pxResults"]).lazy()
        return _fill_context_field_nulls(df)

    if extension == ".parquet":
        if isinstance(file, BytesIO):
            file.seek(0)
            df = pl.read_parquet(file).lazy()
        else:
            df = pl.scan_parquet(file)
        return _fill_context_field_nulls(df)

    if extension.casefold() in {".feather", ".ipc", ".arrow"}:
        if isinstance(file, BytesIO):
            df = pl.read_ipc(file).lazy()
        else:
            df = pl.scan_ipc(file)
        return _fill_context_field_nulls(df)

    raise ValueError(
        f"Could not import file {file!r} (extension {extension!r}). "
        f"Supported extensions: .csv, .json, .parquet, .feather, .ipc, .arrow, .zip."
    )


def read_zipped_file(file: str | BytesIO) -> tuple[BytesIO, str]:
    """Read a Pega zipped NDJSON dataset export.

    A Pega dataset export is a zip archive that contains a ``data.json``
    file (NDJSON format) and optionally a ``META-INF/MANIFEST.mf``
    metadata file.  This helper opens the zip, locates ``data.json``
    (top-level or nested) and returns its bytes.

    Parameters
    ----------
    file : str or BytesIO
        Path to the zip file, or an in-memory zip buffer.

    Returns
    -------
    tuple[BytesIO, str]
        A pair of ``(buffer, ".json")`` ready to be passed back into a
        Polars reader.

    Raises
    ------
    FileNotFoundError
        If the archive does not contain a ``data.json`` entry.
    """
    with zipfile.ZipFile(file, mode="r") as z:
        names = z.namelist()
        logger.debug("Files in archive: %s", names)
        if "data.json" in names:
            target = "data.json"
        else:
            nested = [n for n in names if n.endswith("/data.json")]
            if len(nested) != 1:
                raise FileNotFoundError("Cannot find a 'data.json' file in the zip folder.")
            target = nested[0]
        with z.open(target) as zipped:
            return BytesIO(zipped.read()), ".json"


def read_multi_zip(
    files: Iterable[str],
    *,
    add_original_file_name: bool = False,
    verbose: bool = True,
) -> pl.LazyFrame:
    """Read multiple gzip-compressed NDJSON files and concatenate them.

    Parameters
    ----------
    files : Iterable[str]
        Paths to the ``.json.gz`` files to read.
    add_original_file_name : bool, keyword-only, default=False
        If True, add a ``file`` column recording each source path.
    verbose : bool, keyword-only, default=True
        Show a tqdm progress bar (if installed) and print a completion
        line when done.

    Returns
    -------
    pl.LazyFrame
        Concatenated lazy frame across all input files.
    """
    file_list = list(files)
    total_files = len(file_list)

    try:
        from tqdm import tqdm

        files_iterator: Iterable[str] = tqdm(
            file_list,
            desc="Reading files...",
            disable=not verbose,
            total=total_files,
        )
    except ImportError:
        if verbose:
            warnings.warn(
                "tqdm is not installed. For a progress bar, install tqdm.",
                UserWarning,
                stacklevel=2,
            )
            print("Reading files...")
        files_iterator = file_list

    table = []
    for file in files_iterator:
        data = pl.read_ndjson(gzip.open(file).read())
        if add_original_file_name:
            data = data.with_columns(file=pl.lit(file))
        table.append(data)

    df = pl.concat(table, how="diagonal")
    if verbose:
        print("Combining completed")
    return df.lazy()


def get_latest_file(
    path: str | os.PathLike,
    target: str,
) -> str | None:
    """Find the most recent Pega snapshot file matching a target type.

    Searches ``path`` for files whose name matches one of the well-known
    Pega snapshot patterns for ``target``, then returns the most recent
    one (parsed from the filename's GMT timestamp, falling back to file
    ctime).  Supports ``.json``, ``.csv``, ``.zip``, ``.parquet``,
    ``.feather``, ``.ipc``, ``.arrow``.

    Parameters
    ----------
    path : str or os.PathLike
        Directory to search.
    target : str
        One of ``"model_data"``, ``"predictor_data"``,
        ``"prediction_data"``, ``"value_finder"``.

    Returns
    -------
    str or None
        Full path to the most recent matching file, or ``None`` when no
        matching file exists.

    Raises
    ------
    ValueError
        If ``target`` is not one of the supported names.
    """
    valid_targets = {"model_data", "predictor_data", "value_finder", "prediction_data"}
    if target not in valid_targets:
        raise ValueError(f"Unknown target '{target}'. Expected one of: {sorted(valid_targets)}.")

    supported = [".json", ".csv", ".zip", ".parquet", ".feather", ".ipc", ".arrow"]

    files_dir = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files_dir = [f for f in files_dir if os.path.splitext(f)[-1].lower() in supported]
    logger.debug("Candidate files in %s: %s", path, files_dir)
    matches = find_files(files_dir, target)

    if not matches:  # pragma: no cover
        logger.debug("No files for %s in %s", target, path)
        return None

    paths = [os.path.join(path, name) for name in matches]

    def parse_timestamp(filepath: str) -> datetime:
        try:
            return from_prpc_date_time(
                re.search(r"\d.{0,15}GMT", filepath)[0].replace("_", " "),  # type: ignore[index]
            )
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug("Falling back to ctime for %s: %s", filepath, exc)
            return datetime.fromtimestamp(os.path.getctime(filepath), tz=timezone.utc)

    dates = pl.Series([parse_timestamp(p) for p in paths])
    idx = dates.arg_max()
    if idx is None:
        return None
    return paths[idx]


def find_files(files_dir: Iterable[str], target: str) -> list[str]:
    """Filter a list of filenames down to those matching a Pega snapshot target.

    Parameters
    ----------
    files_dir : Iterable[str]
        Filenames to scan (typically the contents of a directory).
    target : str
        One of ``"model_data"``, ``"predictor_data"``,
        ``"prediction_data"``, ``"value_finder"``.

    Returns
    -------
    list[str]
        Filenames whose names match one of the known patterns for ``target``.

    Raises
    ------
    ValueError
        If ``target`` is not one of the supported names.
    """
    name_groups: dict[str, list[str]] = {
        "model_data": [
            "Data-Decision-ADM-ModelSnapshot",
            "PR_DATA_DM_ADMMART_MDL_FACT",
            "model_snapshots",
            "MD_FACT",
            "ADMMART_MDL_FACT_Data",
            "cached_modelData",
            "Models_data",
        ],
        "predictor_data": [
            "Data-Decision-ADM-PredictorBinningSnapshot",
            "PR_DATA_DM_ADMMART_PRED",
            "predictor_binning_snapshots",
            "PRED_FACT",
            "cached_predictorData",
        ],
        "value_finder": ["Data-Insights_pyValueFinder", "cached_ValueFinder"],
        "prediction_data": ["Data-DM-Snapshot_pyGetSnapshot"],
    }
    if target not in name_groups:
        raise ValueError(f"Target {target} not found.")
    names = name_groups[target]
    matches: list[str] = []
    for file in files_dir:
        for name in names:
            if re.findall(name.casefold(), file.casefold()):
                matches.append(file)
                break
    return matches


@overload
def cache_to_file(
    df: pl.DataFrame | pl.LazyFrame,
    path: str | os.PathLike,
    name: str,
    cache_type: Literal["parquet"] = "parquet",
    compression: pl._typing.ParquetCompression = "uncompressed",
) -> pathlib.Path: ...


@overload
def cache_to_file(
    df: pl.DataFrame | pl.LazyFrame,
    path: str | os.PathLike,
    name: str,
    cache_type: Literal["ipc"] = "ipc",
    compression: pl._typing.IpcCompression = "uncompressed",
) -> pathlib.Path: ...


def cache_to_file(
    df: pl.DataFrame | pl.LazyFrame,
    path: str | os.PathLike,
    name: str,
    cache_type: Literal["ipc", "parquet"] = "ipc",
    compression: pl._typing.ParquetCompression | pl._typing.IpcCompression = "uncompressed",
) -> pathlib.Path:
    """Very simple convenience function to cache data.
    Caches in arrow format for very fast reading.

    Parameters
    ----------
    df: pl.DataFrame
        The dataframe to cache
    path: os.PathLike
        The location to cache the data
    name: str
        The name to give to the file
    cache_type: str
        The type of file to export.
        Default is IPC, also supports parquet
    compression: str
        The compression to apply, default is uncompressed

    Returns
    -------
    os.PathLike:
        The filepath to the cached file

    """
    outpath = pathlib.Path(path).joinpath(pathlib.Path(name))
    if not os.path.exists(path):
        os.mkdir(path)

    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    if cache_type == "ipc":
        outpath = outpath.with_suffix(".arrow")
        # Cast categorical to string since Arrow IPC doesn't support dictionary replacement across batches
        df = df.with_columns(cs.categorical().cast(pl.Utf8))
        df.write_ipc(outpath, compression=cast("pl._typing.IpcCompression", compression))
    if cache_type == "parquet":
        outpath = outpath.with_suffix(".parquet")
        df.write_parquet(outpath, compression=cast("pl._typing.ParquetCompression", compression))
    return outpath


def read_dataflow_output(
    files: Iterable[str] | str,
    cache_file_name: str | None = None,
    *,
    cache_directory: str | os.PathLike = "cache",
):
    """Read the file output of a Pega dataflow run.

    By default, the Prediction Studio data export also uses dataflows,
    so this function applies to those exports as well.

    Dataflow nodes write many small ``.json.gz`` files for each
    partition.  This helper takes a list of files (or a glob pattern)
    and concatenates them into a single :class:`polars.LazyFrame`.

    If ``cache_file_name`` is supplied, results are cached as a parquet
    file.  Subsequent calls only read files that aren't already in the
    cache, then update it.

    Parameters
    ----------
    files : str or Iterable[str]
        File paths to read.  If a string is provided, it's expanded with
        :func:`glob.glob`.
    cache_file_name : str, optional
        If given, cache results to ``<cache_directory>/<cache_file_name>.parquet``.
    cache_directory : str or os.PathLike, keyword-only, default="cache"
        Directory to store the parquet cache.

    Examples
    --------
    >>> from glob import glob
    >>> read_dataflow_output(files=glob("model_snapshots_*.json"))
    """
    if isinstance(files, str):
        files = glob(files)
    original_files = list(files)

    def has_cache():
        return os.path.isfile(cache_file) if cache_file_name else False

    if cache_file_name:
        cache_file = Path(cache_directory) / f"{cache_file_name}.parquet"
        if has_cache():
            cached_data = pl.scan_parquet(cache_file)
            files = pl.LazyFrame({"file": files}).join(cached_data, on="file", how="anti").collect()["file"].to_list()
            if not files:
                return cached_data.filter(pl.col("file").is_in(original_files)).drop("file").lazy()

    if files:
        new_data = read_multi_zip(files=files, add_original_file_name=True)

    if has_cache():
        cached_data = pl.scan_parquet(cache_file)
        combined_data = pl.concat([cached_data, new_data], how="diagonal")
    else:
        combined_data = new_data

    if not cache_file_name:
        return new_data.filter(pl.col("file").is_in(original_files)).lazy()

    combined_data.collect().write_parquet(cache_file)
    return combined_data.filter(pl.col("file").is_in(original_files)).drop("file").lazy()
