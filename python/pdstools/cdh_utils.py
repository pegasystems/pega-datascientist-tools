# -*- coding: utf-8 -*-
"""
cdhtools: Data Science add-ons for Pega.

Various utilities to access and manipulate data from Pega for purposes
of data analysis, reporting and monitoring.
"""

from typing import List, Union
import pandas as pd
import os
import zipfile
import re
import numpy as np
import datetime
from io import BytesIO
import urllib.request
import pytz
import requests
import logging


def readDSExport(
    filename: Union[pd.DataFrame, str],
    path: str = ".",
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Read a Pega dataset export file.
    Can accept either a Pandas DataFrame or one of the following formats:
    - .csv
    - .json
    - .zip (zipped json or CSV)

    It automatically infers the default file names for both model data as well as predictor data.
    If you supply either 'modelData' or 'predictorData' as the 'file' argument, it will search for them.
    If you supply the full name of the file in the 'path' directory, it will import that instead.

    Parameters
    ----------
    filename : [pd.DataFrame, str]
        Either a Pandas DataFrame with the source data (for compatibility),
        or a string, in which case it can either be:
        - The name of the file (if a custom name) or
        - Whether we want to look for 'modelData' or 'predictorData' in the path folder.
    path : str, default = '.'
        The location of the file
    verbose : bool, default = True
        Whether to print out which file will be imported

    Keyword arguments:
        Any arguments to plug into the read csv or json function, from either PyArrow or Pandas.

    Returns
    -------
    pd.DataFrame
        The read data from the given file

    Examples:
        >>> df = readDSExport(filename = 'modelData', path = './datamart')
        >>> df = readDSExport(filename = 'ModelSnapshot.json', path = 'data/ADMData')

        >>> df = pd.read_csv('file.csv')
        >>> df = readDSExport(filename = df)

    """

    # If a dataframe is supplied directly, we can just return it
    if isinstance(filename, pd.DataFrame):
        return filename

    # If the data is a BytesIO object, such as an uploaded file
    # in certain webapps, then we can simply return the object
    # as is, while extracting the extension as well.
    if isinstance(filename, BytesIO):
        name, extension = os.path.splitext(filename.name)
        return import_file(filename, extension, **kwargs)

    # If the filename is simply a string, then we first
    # extract the extension of the file, then look for
    # the file in the user's directory.
    if os.path.isfile(os.path.join(path, filename)):
        file = os.path.join(path, filename)
    else:
        file = get_latest_file(path, filename)

    # If we can't find the file locally, we can try
    # if the file's a URL. If it is, we need to wrap
    # the file in a BytesIO object, and read the file
    # fully to disk for pyarrow to read it.
    if file == "Target not found" or file == None:
        import requests

        try:
            response = requests.get(f"{path}/{filename}")
            logging.info(f"Response: {response}")
            if response.status_code == 200:
                file = f"{path}/{filename}"
                file = BytesIO(urllib.request.urlopen(file).read())
                name, extension = os.path.splitext(filename)

        except Exception as e:
            logging.info(e)
            if verbose:
                print(f"File {filename} not found in dir {path}")
            logging.info(f"File not found: {path}/{filename}")
            return None

    if "extension" not in vars():
        name, extension = os.path.splitext(file)

    # Now we should either have a full path to a file, or a
    # BytesIO wrapper around the file. Pyarrow can read those both.
    return import_file(file, extension, **kwargs)


def import_file(file, extension, **kwargs):
    import pyarrow

    if extension == ".zip":
        file = readZippedFile(file)
    elif extension == ".csv":
        file = pyarrow.csv.read_csv(
            file,
            parse_options=pyarrow.csv.ParseOptions(delimiter=kwargs.get("sep", ",")),
        )
    elif extension == ".json":
        file = pyarrow.json.read_json(file, **kwargs)
    elif extension == ".parquet":
        file = pyarrow.parquet.read_table(file)
    else:
        raise ValueError("Could not import file: {file}, with extension {extension}")

    if not kwargs.pop("return_pa", False) and isinstance(file, pyarrow.Table):
        return file.to_pandas()
    else:
        return file


def readZippedFile(file: str, verbose: bool = False, **kwargs) -> pd.DataFrame:
    """Read a zipped file.
    Reads a dataset export file as exported and downloaded from Pega. The export
    file is formatted as a zipped multi-line JSON file or CSV file
    and the data is read into a pandas dataframe.

    Parameters
    ----------
    file : str
        The full path to the file
    verbose : str, default=False
        Whether to print the names of the files within the unzipped file for debugging purposes

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with the contents.
    """

    with zipfile.ZipFile(file, mode="r") as z:
        files = z.namelist()
        if verbose:
            print(files)  # pragma: no cover
        if "data.json" in files:
            with z.open("data.json") as zippedfile:
                try:
                    from pyarrow import json

                    return json.read_json(zippedfile)
                except ImportError:  # pragma: no cover
                    try:
                        dataset = pd.read_json(zippedfile, lines=True)
                        return dataset
                    except ValueError:
                        dataset = pd.read_json(zippedfile)
                        return dataset
        if "csv.json" in files:  # pragma: no cover
            with z.open("data.csv") as zippedfile:
                try:
                    from pyarrow import csv

                    return csv.read_json(zippedfile).to_pandas()
                except ImportError:
                    return pd.read_csv(zippedfile)
        else:  # pragma: no cover
            raise FileNotFoundError("Cannot find a 'data' file in the zip folder.")


def get_latest_file(path: str, target: str, verbose: bool = False) -> str:
    """Convenience method to find the latest model snapshot.
    It has a set of default names to search for and finds all files who match it.
    Once it finds all matching files in the directory, it chooses the most recent one.
    It only looks at .json, .csv and .zip files for now, as they are supported.
    Needs a path to the directory and a target of either 'modelData' or 'predictorData'.

    Parameters
    ----------
    path : str
        The filepath where the data is stored
    target : str in ['modelData', 'predictorData']
        Whether to look for data about the predictive models ('modelData')
        or the predictor bins ('predictorData')
    verbose : bool, default = False
        Whether to print all found files before comparing name criteria for debugging purposes

    Returns
    -------
    str
        The most recent file given the file name criteria.
    """
    if target not in {"modelData", "predictorData", "ValueFinder"}:
        return f"Target not found"

    supported = [".json", ".csv", ".zip", ".parquet"]

    files_dir = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files_dir = [f for f in files_dir if os.path.splitext(f)[-1].lower() in supported]
    if verbose:
        print(files_dir)  # pragma: no cover
    matches = getMatches(files_dir, target)

    if len(matches) == 0:
        if verbose:  # pragma: no cover
            print(
                f"Unable to find data for {target}. Please check if the data is available."
            )
        return None

    paths = [os.path.join(path, name) for name in matches]

    def f(x):
        try:
            return fromPRPCDateTime(re.search("\d.*GMT", x)[0].replace("_", " "))
        except:
            return pytz.timezone("GMT").localize(
                datetime.datetime.fromtimestamp(os.path.getctime(x))
            )

    dates = np.array([f(i) for i in paths])
    return paths[np.argmax(dates)]


def getMatches(files_dir, target):
    matches = []
    default_model_names = [
        "Data-Decision-ADM-ModelSnapshot",
        "PR_DATA_DM_ADMMART_MDL_FACT",
        "model_snapshots",
        "MD_FACT",
        "ADMMART_MDL_FACT_Data",
    ]
    default_predictor_names = [
        "Data-Decision-ADM-PredictorBinningSnapshot",
        "PR_DATA_DM_ADMMART_PRED",
        "predictor_binning_snapshots",
        "PRED_FACT",
    ]
    ValueFinder_names = ["Data-Insights_pyValueFinder"]

    if target == "modelData":
        names = default_model_names
    elif target == "predictorData":
        names = default_predictor_names
    elif target == "ValueFinder":
        names = ValueFinder_names
    for file in files_dir:
        match = [file for name in names if re.findall(name.casefold(), file.casefold())]
        if len(match) > 0:
            matches.append(match[0])
    return matches


def safe_range_auc(auc: float) -> float:
    """Internal helper to keep auc a safe number between 0.5 and 1.0 always.

    Parameters
    ----------
    auc : float
        The AUC (Area Under the Curve) score

    Returns
    -------
    float
        'Safe' AUC score, between 0.5 and 1.0
    """

    if np.isnan(auc):
        return 0.5
    else:
        return 0.5 + np.abs(0.5 - auc)


def auc_from_probs(groundtruth: List[int], probs: List[float]) -> List[float]: # pragma: no cover
    """Calculates AUC from an array of truth values and predictions.
    Calculates the area under the ROC curve from an array of truth values and
    predictions, making sure to always return a value between 0.5 and 1.0 and
    will return 0.5 in case of any issues.

    Parameters
    ----------
    groundtruth : List[int]
        The 'true' values, Positive values must be represented as
        True or 1. Negative values must be represented as False or 0.
    probs : List[float]
        The predictions, as a numeric vector as the same length as groundtruth

    Returns : List[float]
        The AUC as a value between 0.5 and 1, return 0.5 if there are any issues
        with the data

    Examples:
        >>> auc_from_probs( [1,1,0], [0.6,0.2,0.2])
    """
    # Catching warning - since this is the only place sklearn is used.
    # This way we can remove it from the requirements.
    try:
        from sklearn import roc_auc_score
    except ImportError as e:
        raise ImportError('To calculate AUC, please install sklearn.', e)

    if len(set(groundtruth)) < 2:
        return 0.5
    auc = roc_auc_score(groundtruth, probs)
    return safe_range_auc(auc)


def auc_from_bincounts(pos: List[int], neg: List[int]) -> float:
    """Calculates AUC from counts of positives and negatives directly
    This is an efficient calculation of the AUC directly from an array of positives
    and negatives. It makes sure to always return a value between 0.5 and 1.0
    and will return 0.5 in case of any issues.

    Parameters
    ----------
    pos : List[int]
        Vector with counts of the positive responses
    neg: List[int]
        Vector with counts of the negative responses

    Returns
    -------
    float
        The AUC as a value between 0.5 and 1, return 0.5 if there are any issues
        with the data.

    Examples:
        >>> auc_from_bincounts([3,1,0], [2,0,1])
    """
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    o = np.argsort(-(pos / (pos + neg)))
    FPR = np.flip(np.cumsum(neg[o]) / np.sum(neg), axis=0)
    TPR = np.flip(np.cumsum(pos[o]) / np.sum(pos), axis=0)
    Area = (FPR - np.append(FPR[1:], 0)) * (TPR + np.append(TPR[1:], 0)) / 2
    return safe_range_auc(np.sum(Area))


def auc2GINI(auc: float) -> float:
    """
    Convert AUC performance metric to GINI

    Parameters
    ----------
    auc: float
        The AUC (number between 0.5 and 1)

    Returns
    -------
    float
        GINI metric, a number between 0 and 1

    Examples:
        >>> auc2GINI(0.8232)
    """
    return 2 * safe_range_auc(auc) - 1


def fromPRPCDateTime(
    x: str, return_string: bool = False
) -> Union[datetime.datetime, str]:
    """Convert from a Pega date-time string.

    Parameters
    ----------
    x: str
        String of Pega date-time
    return_string: bool, default=False
        If True it will return the date in string format. If
        False it will return in datetime type

    Returns
    -------
    Union[datetime.datetime, str]
        The converted date in datetime format or string.

    Examples:
        >>> fromPRPCDateTime("20180316T134127.847 GMT")
        >>> fromPRPCDateTime("20180316T134127.847 GMT", True)
        >>> fromPRPCDateTime("20180316T184127.846")
        >>> fromPRPCDateTime("20180316T184127.846", True)
    """

    timezonesplits = x.split(" ")

    if len(timezonesplits) > 1:
        x = timezonesplits[0]

    if "." in x:
        date_no_frac, frac_sec = x.split(".")
        # TODO: obtain only 3 decimals
        if len(frac_sec) > 3:
            frac_sec = frac_sec[:3]
        elif len(frac_sec) < 3:
            frac_sec = "{:<03d}".format(int(frac_sec))
    else:
        date_no_frac = x

    dt = datetime.datetime.strptime(date_no_frac, "%Y%m%dT%H%M%S")

    if len(timezonesplits) > 1:
        dt = dt.replace(tzinfo=pytz.timezone(timezonesplits[1]))

    if "." in x:
        dt = dt.replace(microsecond=int(frac_sec))

    if return_string:
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        return dt


def toPRPCDateTime(x: datetime.datetime) -> str:
    """Convert to a Pega date-time string

    Parameters
    ----------
    x: datetime.datetime
        A datetime object

    Returns
    -------
    str
        A string representation in the format used by Pega

    Examples:
        >>> toPRPCDateTime(datetime.datetime.now())
    """

    return x.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def readClientCredentialFile(credentialFile):
    outputdict = {}
    with open(credentialFile) as f:
        for idx, line in enumerate(f.readlines()):
            if (idx % 2) == 0:
                key = line.rstrip("\n")
            else:
                outputdict[key] = line.rstrip("\n")
        return outputdict


def getToken(credentialFile, verify=True, **kwargs):
    creds = readClientCredentialFile(credentialFile)
    return requests.post(
        url=kwargs.get("URL", creds["Access token endpoint"]),
        data={"grant_type": "client_credentials"},
        auth=(creds["Client ID"], creds["Client Secret"]),
        verify=verify,
    ).json()["access_token"]
