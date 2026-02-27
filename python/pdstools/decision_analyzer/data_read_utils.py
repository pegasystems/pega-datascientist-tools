import gzip
import os
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

import polars as pl

from .column_schema import TableConfig
from .utils import ColumnResolver


def read_nested_zip_files(file_buffer) -> pl.DataFrame:
    """Reads a zip file buffer (uploaded from Streamlit) that contains .zip files,
    which are in fact gzipped ndjson files. Extracts, reads, and concatenates
    them into a single Polars DataFrame.

    Parameters
    ----------
    file_buffer : UploadedFile
        The uploaded zip file buffer from Streamlit.

    Returns
    -------
    pl.DataFrame
        A concatenated Polars DataFrame containing the data from all gzipped ndjson files.

    """
    dfs: list[pl.DataFrame] = []
    columns: list[str] = []

    with zipfile.ZipFile(file_buffer, "r") as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith(".zip") and not file_name.startswith("__MACOSX/._"):
                with zip_ref.open(file_name) as f:
                    data = BytesIO(f.read())
                    df = read_gzipped_data(data)
                    if df is not None and columns == []:
                        columns = df.columns
                    if df is not None:
                        dfs.append(df.select(columns))

    return pl.concat(dfs, rechunk=True)


def read_gzipped_data(data: BytesIO) -> pl.DataFrame | None:
    """Reads gzipped ndjson data from a BytesIO object and returns a Polars DataFrame.

    Parameters
    ----------
    data : BytesIO
        The gzipped ndjson data.

    Returns
    -------
    pl.DataFrame | None
        The Polars DataFrame containing the data, or None if reading fails.

    """
    try:
        with gzip.open(data, "rb") as file:
            file_content = file.read()
            return pl.read_ndjson(BytesIO(file_content)).lazy()  # type: ignore[return-value]
    except Exception as e:
        print(f"Error reading gzipped data: {e}")
        return None


def read_gzips_with_zip_extension(path: str) -> pl.DataFrame:
    """Iterates over all files with a .zip extension in the given directory, treats them
    as gzipped ndjson files, reads, and concatenates them into a single Polars DataFrame.

    Parameters
    ----------
    path : str
        The path to the directory containing the .zip files.

    Returns
    -------
    pl.DataFrame
        A concatenated Polars DataFrame containing the data from all gzipped ndjson files.

    """
    dfs: list[pl.DataFrame] = []
    columns: list[str] = []

    # Iterate over all files in the directory
    for filename in os.listdir(path):
        if filename.endswith(".zip"):
            # Construct the full file path
            file_path = os.path.join(path, filename)
            # Read the gzipped file
            with gzip.open(file_path, "rb") as file:
                file_content = file.read()
                df = pl.read_ndjson(BytesIO(file_content)).lazy()
                if columns == []:
                    columns = df.columns
                dfs.append(df.select(columns))  # type: ignore[arg-type]

    return pl.concat(dfs, rechunk=True)


def read_data(path):
    original_path = Path(path)  # save the original path
    extension = None  # Initialize extension to None
    if original_path.is_dir():
        # It's a directory, so we assume it's partitioned
        # Find the first real data file extension, skipping hidden files
        for dirpath, dirs, files in os.walk(str(original_path)):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for file in files:
                if not file.startswith("."):
                    extension = Path(file).suffix
                    if extension:
                        break
            if extension:
                break
        # Find the depth of the directory structure by finding the maximum
        # number of parts among all matching files
        data_files = [
            p for p in original_path.glob(f"**/*{extension}") if not any(part.startswith(".") for part in p.parts)
        ]
        if not data_files:
            raise ValueError("No data files found in directory")
        depth = max(len(p.parts) for p in data_files) - len(original_path.parts)
        partition_structure = Path("/".join(["*"] * (depth - 1)))
        # Append the file extension glob to only match actual data files
        path = original_path / partition_structure / f"*{extension}"
    else:
        # It's a file, so we read based on the extension
        extension = original_path.suffix
    if extension == ".parquet":
        df = pl.scan_parquet(path)
    elif extension == ".csv":
        df = pl.scan_csv(path)
    elif extension == ".arrow":
        df = pl.scan_ipc(path)
    elif extension in [".ndjson", ".json"]:
        df = pl.scan_ndjson(path)
    elif extension == ".zip":
        if original_path.is_file():
            # Single zip file: extract to temp dir and read the contents
            tmp_dir = tempfile.mkdtemp(prefix="pdstools_zip_")
            with zipfile.ZipFile(original_path, "r") as zf:
                zf.extractall(tmp_dir)
            df = read_data(tmp_dir)
        else:
            # Directory of .zip files (legacy gzipped ndjson)
            df = read_gzips_with_zip_extension(str(original_path))
    elif extension is None:
        raise ValueError("No files found in directory")
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    return df


def validate_columns(
    df: pl.LazyFrame,
    extract_type: dict[str, TableConfig],
) -> tuple[bool, str | None]:
    """Validate that default columns from table definition exist in the dataframe.

    This function checks if required columns exist in the data, accounting for
    the fact that columns may be present under either their source name or
    their target label name.

    Args:
        df: The dataframe to validate
        extract_type: Table configuration mapping column names to their properties

    Returns:
        tuple containing validation success (bool) and error message (str or None)

    """
    resolver = ColumnResolver(
        table_definition=extract_type,
        raw_columns=set(df.collect_schema().names()),
    )
    missing_columns = resolver.get_missing_columns()

    if missing_columns:
        return (
            False,
            f"The following default columns are missing: {', '.join(missing_columns)}",
        )
    return True, None
