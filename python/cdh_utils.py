# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:41:44 2018

@author: santd1
"""

import pandas as pd
import os 
import zipfile
import re

def readDSExport(instanceName, srcFolder='.', tmpFolder='.'):
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
    """
    
    # Check if filename exists or we need to find the most recent
    if os.path.isfile(srcFolder + '/' + instanceName):
        mostRecentZip = instanceName
    else:
        files_dir = [f for f in os.listdir(srcFolder) if os.path.isfile(os.path.join(srcFolder, f))]
        regex = re.compile('^' + instanceName + '_.*\\.zip$')
        instance_files = [f for f in files_dir if regex.search(f)]
        if len(instance_files) == 0:
            #print(instanceName + ' instance not found')
            #return None
            raise Exception(instanceName + ' not found')
        mostRecentZip = sorted(instance_files, reverse=True)[0]
             
    print(mostRecentZip)        
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























