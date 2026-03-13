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
from typing import Literal, overload

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
    ".zip",
    ".tar",
    ".tgz",
    ".gz",
}


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

    Supports multiple formats: parquet, csv, arrow, feather, ndjson, json, zip, tar, tar.gz, tgz, gz.
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
    elif extension is None:
        raise ValueError("No data files found in directory")
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    return df


def read_ds_export(
    filename: str | os.PathLike | BytesIO,
    path: str | os.PathLike = ".",
    verbose: bool = False,
    **reading_opts,
) -> pl.LazyFrame | None:
    """Read Pega dataset exports with additional capabilities.

    This function extends read_data() with:
    - Smart file finding: accepts 'modelData' or 'predictorData' and searches for matching files (ADM-specific)
    - URL downloads: fetches remote files when local paths are not found (useful for demos and examples)
    - Schema overrides: applies Pega-specific type corrections (e.g., PYMODELID as string)

    For simple file reading without these features, use read_data() instead.

    Parameters
    ----------
    filename : str, os.PathLike, or BytesIO
        File identifier. Can be:
        - Full file path
        - Generic name like 'modelData' or 'predictorData' (triggers smart search)
        - BytesIO object (delegates to read_data)
    path : str or os.PathLike, default='.'
        Directory to search for files (ignored for BytesIO or full paths)
    verbose : bool, default=False
        Print file selection details
    **reading_opts
        Additional Polars scan_* options. Common options include:
        - infer_schema_length (int, default=10000): Rows to scan for schema inference
        - separator (str): CSV delimiter
        - ignore_errors (bool): Continue on parse errors

    Returns
    -------
    pl.LazyFrame or None
        Lazy dataframe, or None if file not found

    Examples
    --------
    Smart file finding:

    >>> df = read_ds_export('modelData', path='data/ADMData')

    Specific file:

    >>> df = read_ds_export('ModelSnapshot_20210101.json', path='data')

    URL download:

    >>> df = read_ds_export('ModelSnapshot.zip', path='https://example.com/exports')

    Schema control:

    >>> df = read_ds_export('export.csv', infer_schema_length=200000)

    """
    file: str | BytesIO
    # If the data is a BytesIO object, such as an uploaded file
    # in certain webapps, delegate directly to read_data
    if isinstance(filename, BytesIO):
        logger.debug("Filename is of type BytesIO, delegating to read_data")
        # NOTE: read_data doesn't support **reading_opts yet, so warn if provided
        if reading_opts:
            logger.warning("reading_opts not supported when using BytesIO with read_ds_export, ignoring")
        return read_data(filename)

    # Convert PathLike to string for processing
    filename_str = str(filename) if isinstance(filename, os.PathLike) else filename
    path_str = str(path) if isinstance(path, os.PathLike) else path

    # ADM-specific: Smart file finding for modelData/predictorData patterns
    # If the filename is simply a string, then we first
    # extract the extension of the file, then look for
    # the file in the user's directory.
    if os.path.isfile(filename_str):
        file = filename_str
    elif os.path.isfile(os.path.join(path_str, filename_str)):
        logger.debug("File found in directory")
        file = os.path.join(path_str, filename_str)
    else:
        logger.debug("File not found in directory, scanning for latest file")
        file = get_latest_file(path_str, filename_str)  # type: ignore[assignment]

    # ADM-specific: URL download support
    # If we can't find the file locally, we can try
    # if the file's a URL. If it is, we need to wrap
    # the file in a BytesIO object, and read the file
    # fully to disk for pyarrow to read it.
    if file == "Target not found" or file is None:
        logger.debug("Could not find file in directory, checking if URL")

        try:
            import requests  # type: ignore[import-untyped]

            response = requests.get(f"{path_str}/{filename_str}")
            logger.info(f"Response: {response}")
            if response.status_code == 200:
                logger.debug("File found online, importing and parsing to BytesIO")
                file = f"{path_str}/{filename_str}"
                file = BytesIO(response.content)
                _, extension = os.path.splitext(filename_str)
                # Delegate to import_file for Pega-specific handling
                return import_file(file, extension, **reading_opts)

        except ImportError:
            warnings.warn(
                "Unable to import `requests`, so not able to check for remote files. If you're trying to read in a file from the internet (or, for instance, using the built-in cdh_sample method), try installing the 'requests' package (`uv pip install requests`)",
                ImportWarning,
            )

        except requests.exceptions.SSLError:
            warnings.warn(
                "There was an error making a HTTP request call. This is likely due to your certificates not being installed correctly. Please follow these instructions: https://stackoverflow.com/a/70495761",
                RuntimeWarning,
            )

        except Exception as e:
            logger.info(e)
            if verbose:
                print(f"File {filename_str} not found in dir {path_str}")
            logger.info(f"File not found: {path_str}/{filename_str}")
            return None

    # For local files, use import_file for Pega-specific schema handling
    if "extension" not in vars() and not isinstance(file, BytesIO):
        _, extension = os.path.splitext(file)

    # Delegate to import_file which handles Pega-specific features like schema overrides
    return import_file(file, extension, **reading_opts)


def import_file(
    file: str | BytesIO,
    extension: str,
    **reading_opts,
) -> pl.LazyFrame:
    """Import a file with Pega-specific schema handling.

    Applies ADM-specific type corrections and schema overrides during import.
    Used internally by read_ds_export() for backward compatibility with legacy code.

    Parameters
    ----------
    file : str or BytesIO
        File path or BytesIO object
    extension : str
        File extension (e.g., '.csv', '.json', '.parquet')
    **reading_opts
        Polars reading options (infer_schema_length, separator, ignore_errors, etc.)

    Returns
    -------
    pl.LazyFrame
        Lazy dataframe with schema corrections applied

    """
    if extension == ".zip":
        logger.debug("Zip file found, extracting data.json to BytesIO.")
        file, extension = read_zipped_file(file)
    elif extension == ".gz":
        import gzip

        if isinstance(file, str):
            extension = os.path.splitext(os.path.splitext(file)[0])[1]
            file = BytesIO(gzip.GzipFile(file).read())
        else:
            # For BytesIO objects, extract extension from name attribute if available
            if hasattr(file, "name"):
                extension = os.path.splitext(os.path.splitext(file.name)[0])[1]
            else:
                extension = ""  # Default to empty if we can't determine
            file.seek(0)
            file = BytesIO(gzip.decompress(file.read()))

    if extension == ".csv":
        csv_opts = dict(
            separator=reading_opts.get("sep", ","),
            infer_schema_length=reading_opts.pop("infer_schema_length", 10000),
            null_values=["", "NA", "N/A", "NULL"],
            schema_overrides={"PYMODELID": pl.Utf8},
            try_parse_dates=True,
            ignore_errors=reading_opts.get("ignore_errors", False),
        )
        if isinstance(file, BytesIO):
            return pl.read_csv(
                file,
                **csv_opts,
            ).lazy()
        return pl.scan_csv(file, **csv_opts)

    if extension == ".json":
        try:
            # if isinstance(file, BytesIO):
            #     from pyarrow import json

            #     return pl.LazyFrame(
            #         json.read_json(
            #             file,
            #         )
            #     )
            # else:
            return pl.scan_ndjson(
                file,
                infer_schema_length=reading_opts.pop("infer_schema_length", 10000),
            )
        except Exception:  # pragma: no cover
            try:
                return pl.read_json(file).lazy()
            except Exception:
                import json

                with open(file) as f:  # type: ignore[arg-type]
                    return pl.from_dicts(json.loads(f.read())["pxResults"]).lazy()

    if extension == ".parquet":
        if isinstance(file, BytesIO):
            file.seek(0)
            return pl.read_parquet(file).lazy()
        return pl.scan_parquet(file)

    if extension.casefold() in {".feather", ".ipc", ".arrow"}:
        if isinstance(file, BytesIO):
            return pl.read_ipc(file).lazy()
        return pl.scan_ipc(file)

    raise ValueError(f"Could not import file: {file}, with extension {extension}")


def read_zipped_file(
    file: str | BytesIO,
    verbose: bool = False,
) -> tuple[BytesIO, str]:
    """Read a zipped NDJSON file.
    Reads a dataset export file as exported and downloaded from Pega. The export
    file is formatted as a zipped multi-line JSON file. It reads the file,
    and then returns the file as a BytesIO object.

    Parameters
    ----------
    file : str
        The full path to the file
    verbose : str, default=False
        Whether to print the names of the files within the unzipped file for debugging purposes

    Returns
    -------
    os.BytesIO
        The raw bytes object to pass through to Polars

    """

    def get_valid_files(files: list[str]):
        logger.debug(f"Files found: {files}")
        if "data.json" in files:
            return "data.json"
        # pragma: no cover
        file = [file for file in files if file.endswith("/data.json")]
        if len(file) != 1:
            return None
        return file[0]

    with zipfile.ZipFile(file, mode="r") as z:
        logger.debug("Opened zip file.")
        zfile = get_valid_files(z.namelist())
        logger.debug(f"Opening file {file}")
        if zfile is not None:
            logger.debug("data.json found.")
            if verbose:
                print(
                    (
                        "Zipped json file found. For faster reading, we recommend",
                        "parsing the files to a format such as arrow or parquet. ",
                        "See example in docs #TODO",
                    ),
                )
            with z.open(zfile) as zippedfile:
                return (BytesIO(zippedfile.read()), ".json")
        else:  # pragma: no cover
            raise FileNotFoundError("Cannot find a 'data.json' file in the zip folder.")


def read_multi_zip(
    files: Iterable[str],
    zip_type: Literal["gzip"] = "gzip",
    add_original_file_name: bool = False,
    verbose: bool = True,
) -> pl.LazyFrame:
    """Reads multiple zipped ndjson files, and concats them to one Polars dataframe.

    Parameters
    ----------
    files : list
        The list of files to concat
    zip_type : Literal['gzip']
        At this point, only 'gzip' is supported
    verbose : bool, default = True
        Whether to print out the progress of the import

    """
    import gzip

    if zip_type != "gzip":
        raise NotImplementedError("Only supports gzip for now")

    table = []
    total_files = len(files) if isinstance(files, (list, tuple)) else None

    try:
        from tqdm import tqdm

        files_iterator = tqdm(
            files,
            desc="Reading files...",
            disable=not verbose,
            total=total_files,
        )
    except ImportError:
        if verbose:
            warnings.warn(
                "tqdm is not installed. For a progress bar, install tqdm: pip install tqdm",
                UserWarning,
            )
        files_iterator = files
        if verbose:
            print("Reading files...")

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
    verbose: bool = False,
) -> str | None:
    """Convenience method to find the latest model snapshot.
    It has a set of default names to search for and finds all files who match it.
    Once it finds all matching files in the directory, it chooses the most recent one.
    Supports [".json", ".csv", ".zip", ".parquet", ".feather", ".ipc"].
    Needs a path to the directory and a target of either 'modelData' or 'predictorData'.

    Parameters
    ----------
    path : str
        The filepath where the data is stored
    target : str in ['model_data', 'predictor_data', 'prediction_data']
        Whether to look for data about the predictive models ('model_data')
        or the predictor bins ('predictor_data')
    verbose : bool, default = False
        Whether to print all found files before comparing name criteria for debugging purposes

    Returns
    -------
    str
        The most recent file given the file name criteria.

    """
    if target not in {
        "model_data",
        "predictor_data",
        "value_finder",
        "prediction_data",
    }:
        return "Target not found"

    supported = [".json", ".csv", ".zip", ".parquet", ".feather", ".ipc", ".arrow"]

    files_dir = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files_dir = [f for f in files_dir if os.path.splitext(f)[-1].lower() in supported]
    if verbose:
        print(files_dir)  # pragma: no cover
    matches = find_files(files_dir, target)

    if len(matches) == 0:  # pragma: no cover
        if verbose:
            print(
                f"Unable to find data for {target}. Please check if the data is available.",
            )
        return None

    paths = [os.path.join(path, name) for name in matches]

    def f(x):
        try:
            return from_prpc_date_time(
                re.search(r"\d.{0,15}*GMT", x)[0].replace("_", " "),
            )
        except Exception:
            return datetime.fromtimestamp(os.path.getctime(x), tz=timezone.utc)

    dates = pl.Series([f(i) for i in paths])
    idx = dates.arg_max()
    if idx is None:
        return None
    return paths[idx]


def find_files(files_dir, target):
    matches = []
    default_model_names = [
        "Data-Decision-ADM-ModelSnapshot",
        "PR_DATA_DM_ADMMART_MDL_FACT",
        "model_snapshots",
        "MD_FACT",
        "ADMMART_MDL_FACT_Data",
        "cached_modelData",
        "Models_data",
    ]
    default_predictor_names = [
        "Data-Decision-ADM-PredictorBinningSnapshot",
        "PR_DATA_DM_ADMMART_PRED",
        "predictor_binning_snapshots",
        "PRED_FACT",
        "cached_predictorData",
    ]
    value_finder_names = ["Data-Insights_pyValueFinder", "cached_ValueFinder"]
    default_prediction_names = ["Data-DM-Snapshot_pyGetSnapshot"]

    if target == "model_data":
        names = default_model_names
    elif target == "predictor_data":
        names = default_predictor_names
    elif target == "value_finder":
        names = value_finder_names
    elif target == "prediction_data":
        names = default_prediction_names
    else:
        raise ValueError(f"Target {target} not found.")
    for file in files_dir:
        match = [file for name in names if re.findall(name.casefold(), file.casefold())]
        if len(match) > 0:
            matches.append(match[0])
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
        df.write_ipc(outpath, compression=compression)  # type: ignore[arg-type]
    if cache_type == "parquet":
        outpath = outpath.with_suffix(".parquet")
        df.write_parquet(outpath, compression=compression)
    return outpath


def read_dataflow_output(
    files: Iterable[str] | str,
    cache_file_name: str | None = None,
    *,
    extension: Literal["json"] = "json",
    compression: Literal["gzip"] = "gzip",
    cache_directory: str | os.PathLike = "cache",
):
    """Reads the file output of a dataflow run.

    By default, the Prediction Studio data export also uses dataflows,
    thus this function can be used for those use cases as well.

    Because dataflows have good resiliancy, they can produce a great number of files.
    By default, every few seconds each dataflow node writes a file for each partition.
    While this helps the system stay healthy, it is a bit more difficult to consume.
    This function can take in a list of files (or a glob pattern),
    and read in all of the files.

    If `cache_file_name` is specified, this function caches the data it read before
    as a `parquet` file. This not only reduces the file size, it is also very fast.
    When this function is run and there is a pre-existing parquet file with the name
    specified in `cache_file_name`, it will read all of the files that weren't read in
    before and add it to the parquet file. If no new files are found, it simply returns
    the contents of that parquet file - significantly speeding up operations.

    In a future version, the functionality of this function will be extended to also
    read from S3 or other remote file systems directly using the same caching method.

    Parameters
    ----------
    files : Union[str, Iterable[str]]
        An iterable (list or a glob) of file strings to read.
        If a string is provided, we call glob() on it to find all files corresponding
    cache_file_name : str, Optional
        If given, caches the files to a file with the given name.
        If None, does not use the cache at all
    extension : Literal["json"]
        The extension of the files, by default "json"
    compression : Literal["gzip"]
        The compression of the files, by default "gzip"
    cache_directory : os.PathLike
        The file path to cache the previously read files

    Usage
    -----
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
        new_data = read_multi_zip(
            files=files,
            zip_type=compression,
            add_original_file_name=True,
        )

    if has_cache():
        cached_data = pl.scan_parquet(cache_file)
        combined_data = pl.concat([cached_data, new_data], how="diagonal")
    else:
        combined_data = new_data

    if not cache_file_name:
        return new_data.filter(pl.col("file").is_in(original_files)).lazy()

    combined_data.collect().write_parquet(cache_file)
    return combined_data.filter(pl.col("file").is_in(original_files)).drop("file").lazy()
