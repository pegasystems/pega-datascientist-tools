import logging
import os
import pathlib
import re
import urllib
import warnings
import zipfile
from datetime import datetime, timezone
from glob import glob
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple, Union, overload

import polars as pl

from ..utils.cdh_utils import from_prpc_date_time


def read_ds_export(
    filename: Union[str, BytesIO],
    path: Union[str, os.PathLike] = ".",
    verbose: bool = False,
    **reading_opts,
) -> Optional[pl.LazyFrame]:
    """Read in most out of the box Pega dataset export formats
    Accepts one of the following formats:
    - .csv
    - .json
    - .zip (zipped json or CSV)
    - .feather
    - .ipc
    - .parquet

    It automatically infers the default file names for both model data as well as predictor data.
    If you supply either 'modelData' or 'predictorData' as the 'file' argument, it will search for them.
    If you supply the full name of the file in the 'path' directory, it will import that instead.
    Since pdstools V3.x, returns a Polars LazyFrame. Simply call `.collect()` to get an eager frame.

    Parameters
    ----------
    filename : Union[str, BytesIO]
        Can be one of the following:
        - A string with the full path to the file
        - A string with the name of the file (to be searched in the given path)
        - A BytesIO object containing the file data (e.g., from an uploaded file in a webapp)
    path : str, default = '.'
        The location of the file
    verbose : bool, default = True
        Whether to print out which file will be imported

    Keyword arguments
    -----------------
    Any:
        Any arguments to plug into the scan_* function from Polars.

    Returns
    -------
    pl.LazyFrame
        The (lazy) dataframe

    Examples:
        >>> df = read_ds_export(filename='full/path/to/ModelSnapshot.json')
        >>> df = read_ds_export(filename='ModelSnapshot.json', path='data/ADMData')
        >>> df = read_ds_export(filename=uploaded_file)  # Where uploaded_file is a BytesIO object

    """
    file: Union[str, BytesIO]
    # If the data is a BytesIO object, such as an uploaded file
    # in certain webapps, then we can simply return the object
    # as is, while extracting the extension as well.
    if isinstance(filename, BytesIO):
        logging.debug("Filename is of type BytesIO, importing that directly")
        _, extension = os.path.splitext(filename.name)
        return import_file(filename, extension, **reading_opts)

    # If the filename is simply a string, then we first
    # extract the extension of the file, then look for
    # the file in the user's directory.
    if os.path.isfile(filename):
        file = filename
    elif os.path.isfile(os.path.join(path, filename)):
        logging.debug("File found in directory")
        file = os.path.join(path, filename)
    else:
        logging.debug("File not found in directory, scanning for latest file")
        file = get_latest_file(path, filename)

    # If we can't find the file locally, we can try
    # if the file's a URL. If it is, we need to wrap
    # the file in a BytesIO object, and read the file
    # fully to disk for pyarrow to read it.
    if file == "Target not found" or file is None:
        logging.debug("Could not find file in directory, checking if URL")

        try:
            import requests

            response = requests.get(f"{path}/{filename}")
            logging.info(f"Response: {response}")
            if response.status_code == 200:
                logging.debug("File found online, importing and parsing to BytesIO")
                file = f"{path}/{filename}"
                file = BytesIO(urllib.request.urlopen(file).read())
                _, extension = os.path.splitext(filename)

        except ImportError:
            warnings.warn(
                "Unable to import `requests`, so not able to check for remote files. If you're trying to read in a file from the internet (or, for instance, using the built-in cdh_sample method), try installing the 'requests' package (`uv pip install requests`)"
            )

        except Exception as e:
            logging.info(e)
            if verbose:
                print(f"File {filename} not found in dir {path}")
            logging.info(f"File not found: {path}/{filename}")
            return None

    if "extension" not in vars() and not isinstance(file, BytesIO):
        _, extension = os.path.splitext(file)

    # Now we should either have a full path to a file, or a
    # BytesIO wrapper around the file. Polars can read those both.
    return import_file(file, extension, **reading_opts)


def import_file(
    file: Union[str, BytesIO], extension: str, **reading_opts
) -> pl.LazyFrame:
    """Imports a file using Polars

    Parameters
    ----------
    File: str
        The path to the file, passed directly to the read functions
    extension: str
        The extension of the file, used to determine which function to use

    Returns
    -------
    pl.LazyFrame
        The (imported) lazy dataframe
    """
    if extension == ".zip":
        logging.debug("Zip file found, extracting data.json to BytesIO.")
        file, extension = read_zipped_file(file)
    elif extension == ".gz":
        import gzip

        extension = os.path.splitext(os.path.splitext(file)[0])[1]
        file = BytesIO(gzip.GzipFile(file).read())

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
        else:
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

                with open(file) as f:
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
    file: Union[str, BytesIO], verbose: bool = False
) -> Tuple[BytesIO, str]:
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

    def get_valid_files(files: List[str]):
        logging.debug(f"Files found: {files}")
        if "data.json" in files:
            return "data.json"
        else:  # pragma: no cover
            file = [file for file in files if file.endswith("/data.json")]
            if 0 < len(file) > 1:
                return None
            else:
                return file[0]

    with zipfile.ZipFile(file, mode="r") as z:
        logging.debug("Opened zip file.")
        zfile = get_valid_files(z.namelist())
        logging.debug(f"Opening file {file}")
        if file is not None:
            logging.debug("data.json found.")
            if verbose:
                print(
                    (
                        "Zipped json file found. For faster reading, we recommend",
                        "parsing the files to a format such as arrow or parquet. ",
                        "See example in docs #TODO",
                    )
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
            files, desc="Reading files...", disable=not verbose, total=total_files
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
    path: Union[str, os.PathLike], target: str, verbose: bool = False
) -> str:
    """Convenience method to find the latest model snapshot.
    It has a set of default names to search for and finds all files who match it.
    Once it finds all matching files in the directory, it chooses the most recent one.
    Supports [".json", ".csv", ".zip", ".parquet", ".feather", ".ipc"].
    Needs a path to the directory and a target of either 'modelData' or 'predictorData'.

    Parameters
    ----------
    path : str
        The filepath where the data is stored
    target : str in ['model_data', 'model_data']
        Whether to look for data about the predictive models ('model_data')
        or the predictor bins ('model_data')
    verbose : bool, default = False
        Whether to print all found files before comparing name criteria for debugging purposes

    Returns
    -------
    str
        The most recent file given the file name criteria.
    """
    if target not in {"model_data", "predictor_data", "value_finder"}:
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
                f"Unable to find data for {target}. Please check if the data is available."
            )
        return None

    paths = [os.path.join(path, name) for name in matches]

    def f(x):
        try:
            return from_prpc_date_time(
                re.search(r"\d.{0,15}*GMT", x)[0].replace("_", " ")
            )
        except Exception:
            return datetime.fromtimestamp(os.path.getctime(x), tz=timezone.utc)

    dates = pl.Series([f(i) for i in paths])
    return paths[dates.arg_max()]


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

    if target == "model_data":
        names = default_model_names
    elif target == "predictor_data":
        names = default_predictor_names
    elif target == "value_finder":
        names = value_finder_names
    else:
        raise ValueError(f"Target {target} not found.")
    for file in files_dir:
        match = [file for name in names if re.findall(name.casefold(), file.casefold())]
        if len(match) > 0:
            matches.append(match[0])
    return matches


@overload
def cache_to_file(
    df: Union[pl.DataFrame, pl.LazyFrame],
    path: Union[str, os.PathLike],
    name: str,
    cache_type: Literal["parquet"] = "parquet",
    compression: pl._typing.ParquetCompression = "uncompressed",
) -> pathlib.Path: ...


@overload
def cache_to_file(
    df: Union[pl.DataFrame, pl.LazyFrame],
    path: Union[str, os.PathLike],
    name: str,
    cache_type: Literal["ipc"] = "ipc",
    compression: pl._typing.IpcCompression = "uncompressed",
) -> pathlib.Path: ...


def cache_to_file(
    df: Union[pl.DataFrame, pl.LazyFrame],
    path: Union[str, os.PathLike],
    name: str,
    cache_type: Literal["ipc", "parquet"] = "ipc",
    compression: Union[
        pl._typing.ParquetCompression, pl._typing.IpcCompression
    ] = "uncompressed",
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
        df.write_ipc(outpath, compression=compression)
    if cache_type == "parquet":
        outpath = outpath.with_suffix(".parquet")
        df.write_parquet(outpath, compression=compression)
    return outpath


def read_dataflow_output(
    files: Union[Iterable[str], str],
    cache_file_name: Optional[str] = None,
    *,
    extension: Literal["json"] = "json",
    compression: Literal["gzip"] = "gzip",
    cache_directory: Union[str, os.PathLike] = "cache",
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
            files = (
                pl.LazyFrame({"file": files})
                .join(cached_data, on="file", how="anti")
                .collect()["file"]
                .to_list()
            )
            if not files:
                return (
                    cached_data.filter(pl.col("file").is_in(original_files))
                    .drop("file")
                    .lazy()
                )

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
    return (
        combined_data.filter(pl.col("file").is_in(original_files)).drop("file").lazy()
    )
