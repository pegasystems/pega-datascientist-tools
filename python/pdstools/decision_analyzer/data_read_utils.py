import gzip
import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, List
import polars as pl

from .table_definition import TableConfig


def read_nested_zip_files(file_buffer) -> pl.DataFrame:
    """
    Reads a zip file buffer (uploaded from Streamlit) that contains .zip files,
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
    dfs: List[pl.DataFrame] = []
    columns: List[str] = []

    with zipfile.ZipFile(file_buffer, "r") as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith(".zip") and not file_name.startswith("__MACOSX/._"):
                with zip_ref.open(file_name) as f:
                    data = BytesIO(f.read())
                    df = read_gzipped_data(data)
                    if columns == []:
                        columns = (
                            df.columns
                        )  # Ensures columns in each DataFrame have the same order.
                    if df is not None:
                        dfs.append(df.select(columns))

    return pl.concat(dfs, rechunk=True)


def read_gzipped_data(data: BytesIO) -> Optional[pl.DataFrame]:
    """
    Reads gzipped ndjson data from a BytesIO object and returns a Polars DataFrame.

    Parameters
    ----------
    data : BytesIO
        The gzipped ndjson data.

    Returns
    -------
    Optional[pl.DataFrame]
        The Polars DataFrame containing the data, or None if reading fails.
    """
    try:
        with gzip.open(data, "rb") as file:
            file_content = file.read()
            return pl.read_ndjson(BytesIO(file_content)).lazy()
    except Exception as e:
        print(f"Error reading gzipped data: {e}")
        return None


def read_gzips_with_zip_extension(path: str) -> pl.DataFrame:
    """
    Iterates over all files with a .zip extension in the given directory, treats them
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
    dfs: List[pl.DataFrame] = []
    columns: List[str] = []

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
                dfs.append(df.select(columns))

    return pl.concat(dfs, rechunk=True)


def read_data(path):
    original_path = Path(path)  # save the original path
    extension = None  # Initialize extension to None
    if original_path.is_dir():
        # It's a directory, so we assume it's partitioned
        # Find the depth of the directory structure by finding the maximum number of parts among all files
        depth = max(
            len(p.parts) for p in original_path.glob("**/*") if p.is_file()
        ) - len(original_path.parts)
        partition_structure = Path("/".join(["*"] * depth))
        path = (
            original_path / partition_structure
        )  # now path points to the partition structure
        # Assume the first file extension is the same for all files in the directory
        for dirpath, dirs, files in os.walk(
            str(original_path)
        ):  # walk through the original directory
            for file in files:
                extension = Path(file).suffix
                if extension:
                    break
            if extension:
                break
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
        df = read_gzips_with_zip_extension(original_path)
    elif extension is None:
        raise ValueError("No files found in directory")
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    return df


# OneDrive seems to be using different paths on different systems even on the same OS. This
# way we just find the first valid one. Can be used to support other OS-es as well.
def get_da_data_path():
    onedrive_da_paths = [
        Path(p).expanduser()
        for p in [
            "~/Library/CloudStorage/OneDrive-SharedLibraries-PegasystemsInc/PRD - 1-1 Customer Engagement Alliance - AI Chapter/projects/Decision Analyzer (Insights)",
            "~/Library/CloudStorage/OneDrive-PegasystemsInc/AI Chapter/projects/Decision Analyzer (Insights)",
        ]
        if Path(p).expanduser().exists()
    ]
    if len(onedrive_da_paths) == 0:
        exit("No valid source path")
    return onedrive_da_paths[0]


def validate_columns(df: pl.LazyFrame, extract_type: Dict[str, TableConfig]):
    existing_columns = df.columns
    required_columns = [
        col for col, properties in extract_type.items() if properties["required"]
    ]
    missing_columns = [col for col in required_columns if col not in existing_columns]

    if missing_columns:
        raise ValueError(
            f"The following required columns are missing: {', '.join(missing_columns)}"
        )
