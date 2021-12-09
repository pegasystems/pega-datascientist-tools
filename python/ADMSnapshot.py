import cdh_utils
import pandas as pd
import re
import copy
from typing import Optional, Tuple

class ADMSnapshot:
    """Main class for importing, preprocessing and structuring Pega snapshot data."""

    def __init__(self, path, overwrite_mapping:Optional[dict] = None, **kwargs):
        """Gets all available data, properly names and merges into one main dataframe

        Parameters
        ----------
        path : str
            The path of the data files
            Default = current path (',')
        overwrite_mapping : dict
            A dictionary to overwrite default feature names in the input data
            Default = None
        
        Keyword arguments
        -----------------
        model_filename : str
            The name, or extended filepath, towards the model file
        predictor_filename : str
            The name, or extended filepath, towards the predictors file

        Examples
        --------
        >>> Data =  ADMSnapshot(f"/CDHSample")
        >>> Data =  ADMSnapshot(f"Data/Adaptive Models & Predictors Export",
                    model_filename = "Data-Decision-ADM-ModelSnapshot_AdaptiveModelSnapshotRepo20201110T085543_GMT/data.json",
                    predictor_filename = "Data-Decision-ADM-PredictorBinningSnapshot_PredictorBinningSnapshotRepo20201110T084825_GMT/data.json")
        >>> Data =  ADMSnapshot(f"Data/files",
                    model_filename = "ModelData.csv",
                    predictor_filename = "PredictorData.csv")

        """
        self.modelData, self.predictorData = self.import_data(path, overwrite_mapping=overwrite_mapping, **kwargs)
        if self.modelData is not None and self.predictorData is not None:
            self.combinedData = self.get_combined_data()
        else:
            print("Could not be combined. Do you have both model data and predictor data?")

    def import_data(self, 
                    path: Optional[str]='.', 
                    overwrite_mapping: Optional[dict]= None,
                    subset: bool = True,
                    model_df: Optional[pd.DataFrame] = None,
                    predictor_df: Optional[pd.DataFrame] = None,
                    **kwargs) -> pd.DataFrame:
        """Method to automatically import & format the relevant data.

        Parameters
        ----------
        path : str
            The path of the data files
            Default = current path (',')
        overwrite_mapping : dict
            A dictionary to overwrite default feature names in the input data
            Default = None
        
        Returns
        -------
        pd.DataFrame
            A merged dataframe with all available data
        """
        if model_df is not None: 
            df1, self.renamed_model, self.missing_model = self._import_utils(name=model_df, subset=subset)
        else:
            model_filename = kwargs.pop('model_filename', 'modelData')
            df1, self.renamed_model, self.missing_model = self._import_utils(model_filename, path, overwrite_mapping, subset)
        if df1 is not None:
            df1['SuccesRate'] = df1['Positives'] / df1['ResponseCount'] if df1 is not None else None
        
        if predictor_df is not None:
            df2, self.renamed_preds, self.missing_preds = self._import_utils(name=predictor_df)
        else:
            predictor_filename = kwargs.pop('predictor_filename', 'predictorData')
            df2, self.renamed_preds, self.missing_preds = self._import_utils(predictor_filename, path, overwrite_mapping, subset)
        if df2 is not None:
            try: 
                df2['SuccesRate'] = df2['BinPositives'] / df2['BinResponseCount'] if df2 is not None else None
            except KeyError:
                df2['SuccesRate'] = df2['BinPositives'] / df2['ResponseCount'] if df2 is not None else None

        if df1 is not None and df2 is not None:
            total_missing = set(self.missing_model) & set(self.missing_preds) - set(df1.columns) - set(df2.columns) 
            if len(total_missing) > 0:
                print(f"""Missing required field values. 
                Please check if they are available in the data, 
                and supply a custom mapping if the naming is different from default. 
                Missing values: {total_missing}""")
                
        return df1, df2
    
    def _import_utils(self, name, path=None, overwrite_mapping=None, subset=True):
        if isinstance(name, str):
            df = cdh_utils.readDSExport(file=name, path=path)
        else: df = name

        if isinstance(df, pd.DataFrame):
            self.model_snapshots = True
            df.columns = self._capitalize(list(df.columns))
            df, renamed, missing = self._available_columns(df, overwrite_mapping) 
            if subset: df = df[renamed.values()]
            df = self._set_types(df)
            return df, renamed, missing

        else: 
            return None, None, None

    def _available_columns(self, df:pd.DataFrame, overwrite_mapping:Optional[dict]=None) -> Tuple[pd.DataFrame, dict, list]:
        """Based on the default names for variables, rename available data to proper formatting

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        overwrite_mapping : dict
            If given, adds 'search terms' to the default names to look for
            If an extra variable is given which is not in default_names, it will also be included
        
        Returns
        -------
        (pd.DataFrame, dict, list)
            The original dataframe, but renamed for the found columns &
            The original and updated names for all renamed columns &
            The variables that were not found in the table
        """
        default_names = {
            'ModelID': ['ModelID'],
            'Issue': ['Issue'],
            'Group': ['Group'], 
            'Channel': ['Channel'], 
            'Direction': ['Direction'], 
            'ModelName': ['Name'], 
            'Positives': ['Positives'], 
            'ResponseCount': ['Response', 'Responses', 'ResponseCount'], 
            'SnapshotTime': ['ModelSnapshot', 'SnapshotTime'],
            'PredictorName': ['PredictorName'],
            'Performance': ['Performance'],
            'EntryType': ['EntryType'],
            'PredictorName': ['PredictorName'],
            'BinSymbol': ['BinSymbol'],
            'BinIndex': ['BinIndex'], 
            'BinType': ['BinType'],
            'BinPositives': ['BinPositives'],
            'BinresponseCount': ['BinresponseCount']
        } #NOTE: these default names are already capitalized properly, with py/px/pz removed.
        
        if overwrite_mapping is not None:
            old_keys = list(overwrite_mapping.keys())
            new_keys = self._capitalize(list(old_keys))
            for i, _ in enumerate(new_keys):
                overwrite_mapping[new_keys[i]] = overwrite_mapping.pop(old_keys[i]) 
        
            for key, name in overwrite_mapping.items():
                if key not in default_names.keys():
                    default_names[key] = [name]
                else: 
                    default_names[key].insert(0, name)
        
        variables = copy.deepcopy(default_names)
        for key, values in default_names.items():
            variables[key] = [name for name in values if name in df.columns]
        missing = [x for x,y in variables.items() if len(y)==0]
        variables = {y[0]:x for x,y in variables.items() if len(y)>0}
        df = df.rename(columns=variables)
            
        return df, variables, missing

    @staticmethod
    def _capitalize(fields: list) -> list:
        """Applies automatic capitalization, aligned with the R couterpart.

        Parameters
        ----------
        fields : list
            A list of names
        
        Returns
        -------
        fields : list
            The input list, but each value properly capitalized
        """
        capitalizeEndWords = [  "ID", "Key", "Name", "Count", "Time", "DateTime", "UpdateTime",
                                "ToClass", "Version", "Predictor", "Predictors", "Rate", "Ratio",
                                "Negatives", "Positives", "Threshold", "Error", "Importance",
                                "Type", "Percentage", "Index", "Symbol",
                                "LowerBound", "UpperBound", "Bins", "GroupIndex",
                                "ResponseCount", "NegativesPercentage", "PositivesPercentage",
                                "BinPositives", "BinNegatives", "BinResponseCount", "BinResponseCount",
                                "ResponseCountPercentage"]
        fields = [re.sub("^p(x|y|z)", "", field.lower()) for field in fields]
        fields = [field.title() for field in fields]
        for word in capitalizeEndWords:
            fields = [re.sub(f"({word.casefold()})", word, field) for field in fields]
        return fields
    

    @staticmethod
    def _set_types(df: pd.DataFrame) -> pd.DataFrame:
        """A method to change columns to their proper type
        
        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe
        
        Returns
        -------
        pd.DataFrame
            The input dataframe, but the proper typing applied
        """
        for col in {'Issue', 'Group', 'Channel', 'Direction', 'ModelName'} & set(df.columns):
            df[col] = df[col].astype(str) 
        
        for col in {'Positives', 'Negatives'} & set(df.columns):
            df[col] = df[col].astype(int) 

        for col in {'Performance'} & set(df.columns):
            df[col] = df[col].astype(float) 

        try: 
            df['SnapshotTime'] = pd.to_datetime(df['SnapshotTime'])
        except Exception:
            print("Warning: Unable to format timestamps.")

        return df
    
    @staticmethod
    def last(df: pd.DataFrame) -> pd.DataFrame:
        """Property to retrieve only the last values for a given dataframe."""
        if 'ModelName' in df.columns: 
            return df.sort_values('SnapshotTime').groupby(['ModelID']).last()
        if 'PredictorName' in df.columns:
            return df.sort_values('SnapshotTime').groupby(['ModelID', 'PredictorName']).last()
        
    def get_combined_data(self, modelData:pd.DataFrame = None, predictorData:pd.DataFrame = None) -> pd.DataFrame:
        """Combines the model data and predictor data into one dataframe.
        
        Parameters
        ----------
        modelData : pd.DataFrame
            Optional dataframe to override 'self.modelData' for merging
        predictorData : pd.DataFrame
            Optional dataframe to override 'self.predictorData' for merging
        
        Returns
        -------
        pd.DataFrame
            The combined dataframe
        """

        #TODO: support multiple snapshots for combined data
        lastPreds = self.last(self.predictorData) if predictorData is None else predictorData
        lastModels = self.last(self.modelData) if modelData is None else modelData
        combined = lastModels.merge(lastPreds, on='ModelID', how='right', suffixes=('', 'Bin'))
        return combined
    
    @staticmethod
    def defaultPredictorCategorization(name:str) -> str:
        raise NotImplemented