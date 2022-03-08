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
from sklearn.metrics import roc_auc_score
import datetime
from io import BytesIO
import urllib.request
import http

def readDSExport(filename: Union[pd.DataFrame, str], path: str='.', verbose: bool=True, force_pandas=False, **kwargs) -> pd.DataFrame:
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
    path : str, default='.'
        The location of the file
    verbose : bool, default=True
        Whether to print out which file will be imported
    
    Keyword arguments:
        Any arguments to plug into the read csv or json function, from either PyArrow or Pandas.
    
    Returns
    -------
    pd.DataFrame
        The read data from the given file
            
    Examples: 
        >>> df = readDSExport(file = 'modelData', path = './datamart')
        >>> df = readDSExport(file = 'ModelSnapshot.json', path = 'data/ADMData')

        >>> df = pd.read_csv('file.csv')
        >>> df = readDSExport(file = df)
    
    """
    if isinstance(filename, pd.DataFrame):
        return filename

    is_url = False

    if os.path.isfile(path + '/' + filename):
        file = f"{path}/{filename}"
    else:
        file = get_latest_file(path, filename)
        if file == 'dir_not_found':
            import requests
            try: 
                response = requests.get(f'{path}/{filename}')
                is_url = True if response.status_code == 200 else False
            except:
                is_url = False
            if is_url: 
                file = f"{path}/{filename}"
                if file.split(".")[-1] == 'zip':
                    file = urllib.request.urlopen(f"{path}/{filename}")
                if verbose: print('File found through URL')
    if file in [None, 'dir_not_found']:
        if verbose: print(f'File {filename} not found in dir {path}')
        return None

    if isinstance(file, str):
        extension = file.split(".")[-1]
    elif isinstance(file, http.client.HTTPResponse):
        extension = 'zipped'

    if verbose: print(f"Importing: {path}/{filename}") if is_url else print(f"Importing: {file}")

    if extension == 'parquet':
        try: 
            import pyarrow.parquet as pq
            return pq.read_table(file).to_pandas()
        except ImportError:
            print("You need to import pyarrow to read parquet files.")
    if extension == 'csv':
        try:
            if force_pandas or is_url: raise ImportError('Forcing pandas.')
            from pyarrow import csv
            return csv.read_csv(file, parse_options=csv.ParseOptions(delimiter=kwargs.get('sep', ','))).to_pandas()
        except ImportError:
            if not is_url:
                if verbose: print("Can't import pyarrow, so defaulting to pandas. For faster imports, please install pyarrow.")
            return pd.read_csv(file, **kwargs)
        except OSError:
            raise FileNotFoundError(f"File {file} is not found.")
    elif extension == 'json':
        try:
            if force_pandas: raise ImportError('Forcing pandas.')
            from pyarrow import json
            return json.read_json(file, **kwargs).to_pandas()
        except ImportError:
            if verbose: print("Can't import pyarrow, so defaulting to pandas. For faster imports, please install pyarrow.")

            try: 
                return pd.read_json(file, lines=True, **kwargs)
            except ValueError:
                return pd.read_json(file, **kwargs)
        except OSError:
            raise FileNotFoundError(f"File {file} is not found.")
    else: 
        try:
            if is_url and extension == 'zipped':
                return readZippedFile(file=BytesIO(file.read()))
            elif extension == 'zip':
                return readZippedFile(file=file)
            else: return FileNotFoundError(f"File {file} is not found.")
        except OSError:
            raise FileNotFoundError(f"File {file} is not found.")


def readZippedFile(file: str, verbose: bool = False) -> pd.DataFrame:
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

    with zipfile.ZipFile(file, mode='r') as z:
        files = z.namelist()
        if verbose: print(files)
        if 'data.json' in files:
            with z.open('data.json') as zippedfile:
                try:
                    from pyarrow import json
                    return json.read_json(zippedfile).to_pandas()
                except ImportError:
                    try: 
                        dataset = pd.read_json(zippedfile, lines=True)
                        return dataset
                    except ValueError:
                        dataset = pd.read_json(zippedfile)
                        return dataset
        if 'csv.json' in files:
            with z.open('data.csv') as zippedfile:
                try: 
                    from pyarrow import csv
                    return csv.read_json(zippedfile).to_pandas()
                except ImportError:
                    return pd.read_csv(zippedfile)
        else:
            raise FileNotFoundError("Cannot find a 'data' file in the zip folder.")


def get_latest_file(path: str, target: str, verbose: bool=False) -> str:
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
    verbose : bool, default=False
        Whether to print all found files before comparing name criteria for debugging purposes
    
    Returns
    -------
    str
        The most recent file given the file name criteria.
    """
    if target not in {'modelData', 'predictorData'}: 
        return 'dir_not_found'

    #NOTE remove some default names
    default_model_names = [
        'Data-Decision-ADM-ModelSnapshot',
        'PR_DATA_DM_ADMMART_MDL_FACT',
        'model_snapshots',
        'MD_FACT',
        'ADMMART_MDL_FACT_Data'
    ]
    default_predictor_names = [
        'Data-Decision-ADM-PredictorBinningSnapshot',
        'PR_DATA_DM_ADMMART_PRED',
        'predictor_binning_snapshots',
        'PRED_FACT'
    ]
    supported = ['.json', '.csv', '.zip', '.parquet']


    files_dir = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files_dir = [f for f in files_dir if os.path.splitext(f)[-1].lower() in supported]
    if verbose: print(files_dir)
    matches = []

    if target == 'modelData':
        for file in files_dir:
            match = [file for name in default_model_names if re.findall(name.casefold(), file.casefold())]
            if len(match)>0:
                matches.append(match[0])
    elif target == 'predictorData':
        for file in files_dir:
            match = [file for name in default_predictor_names if re.findall(name.casefold(), file.casefold())]
            if len(match)>0:
                matches.append(match[0])
    if len(matches) == 0:
        if verbose: print(f"Unable to find data for {target}. Please check if the data is available.")
        return None

    paths = [os.path.join(path, name) for name in matches]
    return max(paths, key=os.path.getctime) #TODO check for latest timestamp

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
        return (0.5 + np.abs(0.5-auc))
    

def auc_from_probs(groundtruth: List[int], probs: List[float]) -> List[float]:
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



def auc2GINI(auc:float) -> float:
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
    return (2*safe_range_auc(auc) - 1)
    

def fromPRPCDateTime(x: str, return_string: bool = False) -> Union[datetime.datetime, str]:
    """ Convert from a Pega date-time string.
    
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
    import pytz
    
    timezonesplits = x.split(' ')
    
    if len(timezonesplits) > 1:
        x = timezonesplits[0]
        
        
    if '.' in x:
        date_no_frac, frac_sec = x.split('.')
        # TODO: obtain only 3 decimals
        if len(frac_sec) > 3:
            frac_sec = frac_sec[:3]
        elif len(frac_sec) < 3:
            frac_sec = '{:<03d}'.format(int(frac_sec))
    else:
        date_no_frac = x
        
    dt = datetime.datetime.strptime(date_no_frac, "%Y%m%dT%H%M%S")
    
    if len(timezonesplits) > 1:
        dt = dt.replace(tzinfo=pytz.timezone(timezonesplits[1]))
        
    
    if '.' in x:
        dt = dt.replace(microsecond=int(frac_sec))
    
    if return_string:
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        return dt
    

def toPRPCDateTime(x: datetime.datetime) -> str:
    """ Convert to a Pega date-time string
    
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