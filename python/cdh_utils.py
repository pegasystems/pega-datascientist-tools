# -*- coding: utf-8 -*-
"""
cdhtools: Data Science add-ons for Pega.

Various utilities to access and manipulate data from Pega for purposes
of data analysis, reporting and monitoring.


"""

import pandas as pd
import os 
import errno
import zipfile
import re
import numpy as np
from sklearn.metrics import roc_auc_score
import datetime

def readDSExport(instanceName, srcFolder='.', tmpFolder='.', verbose=True):
    """Read a Pega dataset export file.
    Reads a dataset export file as exported and downloaded from Pega. The export
    file is formatted as a zipped multi-line JSON file, unzip it into a temp 
    folder and read the data into a pandas dataframe.
  
    Args:
        instancename: Name of the file w/o the timestamp, in Pega format 
            <Applies To>_<Instance Name>, or the complete filename including
            timestamp and zip extension as exported from Pega.
        srcFolder: Optional folder to look for the file (defaults to the
             current folder)
        tmpFolder: Optional folder to store the unzipped data (defaults to the
               source folder)

    Returns:
        A pandas dataframe with the contents.
        
    Raises:
        Exception: does not find any file for the given instanceName 
        
    Examples: 
        >>> df = readDSExport("Data-Decision-ADM-ModelSnapshot_AllModelSnapshots", srcFolder="inst/extdata", tmpFolder="tmp3")
        >>> df = readDSExport("Data-Decision-ADM-ModelSnapshot_AllModelSnapshots_20180316T134315_GMT.zip", srcFolder="inst/extdata", tmpFolder="tmp3")
    """
    
    # Check if filename exists or we need to find the most recent
    if os.path.isfile(srcFolder + '/' + instanceName):
        mostRecentZip = instanceName
    else:
        files_dir = [f for f in os.listdir(srcFolder) if os.path.isfile(os.path.join(srcFolder, f))]
        regex = re.compile('^' + instanceName + '_.*\\.zip$')
        instance_files = [f for f in files_dir if regex.search(f)]
        if len(instance_files) == 0:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), instanceName)
        mostRecentZip = sorted(instance_files, reverse=True)[0]
             
    if verbose: print(mostRecentZip)
    # Remove json file if already exists
    jsonFile = tmpFolder + "/data.json"
    if os.path.exists(jsonFile):
        os.remove(jsonFile)
    # Extract zip file   
    zip_ref = zipfile.ZipFile(srcFolder + '/' + mostRecentZip, 'r')  
    zip_ref.extractall(tmpFolder)
    zip_ref.close()   
    # Read json and transform to pandas dataframe
    df = pd.read_json(jsonFile, lines=True)
    return df


def safe_range_auc(auc):
    """Internal helper to keep auc a safe number between 0.5 and 1.0 always.
    
    """
    
    if np.isnan(auc):
        return 0.5
    else:
        return (0.5 + np.abs(0.5-auc))
    

def auc_from_probs(groundtruth, probs):
    """ Calculates AUC from an array of truth values and predictions.
    Calculates the area under the ROC curve from an array of truth values and 
    predictions, making sure to always return a value between 0.5 and 1.0 and 
    will return 0.5 in case of any issues.
    
    Args:
        groundtruth: The 'true' values, Positive values must be represented as 
            True or 1. Negative values must be represented as False or 0.
        probs: The predictions, as a numeric vector as the same length as groundtruth
    
    Returns:
        The AUC as a value between 0.5 and 1, return 0.5 if there are any issues
        with the data
        
    Examples:
        >>> auc_from_probs( [1,1,0], [0.6,0.2,0.2])
    """
    if len(set(groundtruth)) < 2:
        return 0.5
    auc = roc_auc_score(groundtruth, probs)
    return safe_range_auc(auc)
    

def auc_from_bincounts(pos, neg):
    """
    Calculates AUC from counts of positives and negatives directly
    This is an efficient calculation of the AUC directly from an array of positives
    and negatives. It makes sure to always return a value between 0.5 and 1.0
    and will return 0.5 in case of any issues.
    
    Args:
        pos: Vector with counts of the positive responses
        neg: Vector with counts of the negative responses
    
    Returns:
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



def auc2GINI(auc):
    """
    Convert AUC performance metric to GINI
    
    Args:
        auc: The AUC (number between 0.5 and 1)
        
    Returns:
        GINI metric, a number between 0 and 1
        
    Examples:
        >>> auc2GINI(0.8232)
    """
    return (2*safe_range_auc(auc) - 1)
    

def fromPRPCDateTime(x, return_string = False):
    """ Convert from a Pega date-time string.
    
    Args:
        x: String of Pega date-time
        return_string: If True it will return the date in string format. If 
            False it will return in datetime type
    Returns:
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
    

def toPRPCDateTime(x):
    """ Convert to a Pega date-time string
    
    Args:
        x: A datetime object
    
    Returns:
        A string representation in the format used by Pega
        
    Examples:
        >>> toPRPCDateTime(datetime.datetime.now())
        
    """
    
    return x.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    















