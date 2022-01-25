from datetime import timedelta
import numpy as np
import cdh_utils
from plots import ADMVisualisations
import pandas as pd
import re
import copy
from typing import Optional, Tuple, Union
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import seaborn as sns

class ADMDatamart(ADMVisualisations):
    """Main class for importing, preprocessing and structuring Pega ADM Datamart snapshot data."""

    def __init__(self, path:str=".", overwrite_mapping:Optional[dict] = None, query=None, **kwargs):
        """Gets all available data, properly names and merges into one main dataframe

        Parameters
        ----------
        path : str
            The path of the data files
            Default = current path ('.')
        overwrite_mapping : dict
            A dictionary to overwrite default feature names in the input data
            Default = None
        
        Keyword arguments
        -----------------
        model_filename : str
            The name, or extended filepath, towards the model file
        predictor_filename : str
            The name, or extended filepath, towards the predictors file
        model_df : pd.DataFrame
            Optional override to supply a dataframe instead of a file
        predictor_df : pd.DataFrame
            Optional override to supply a dataframe instead of a file
        subset : bool
            Whether to only select the renamed columns or retain them all

        Examples
        --------
        >>> Data =  ADMDatamart(f"/CDHSample")
        >>> Data =  ADMDatamart(f"Data/Adaptive Models & Predictors Export",
                    model_filename = "Data-Decision-ADM-ModelSnapshot_AdaptiveModelSnapshotRepo20201110T085543_GMT/data.json",
                    predictor_filename = "Data-Decision-ADM-PredictorBinningSnapshot_PredictorBinningSnapshotRepo20201110T084825_GMT/data.json")
        >>> Data =  ADMDatamart(f"Data/files",
                    model_filename = "ModelData.csv",
                    predictor_filename = "PredictorData.csv")

        """
        self.modelData, self.predictorData = self.import_data(path, overwrite_mapping=overwrite_mapping, query=query, **kwargs)
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
                    query: Union[str, dict] = None,
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
        verbose = kwargs.pop('verbose', True)
        if model_df is not None: 
            df1, self.renamed_model, self.missing_model = self._import_utils(name=model_df, subset=subset, query=query, verbose=verbose)
        else:
            model_filename = kwargs.pop('model_filename', 'modelData')
            df1, self.renamed_model, self.missing_model = self._import_utils(model_filename, path, overwrite_mapping, subset, query=query, verbose=verbose)
        if df1 is not None:
            df1['SuccessRate'] = df1['Positives'] / df1['ResponseCount'] if df1 is not None else None
        
        if predictor_df is not None:
            df2, self.renamed_preds, self.missing_preds = self._import_utils(name=predictor_df, subset=subset, query=query, verbose=verbose)
        else:
            predictor_filename = kwargs.pop('predictor_filename', 'predictorData')
            df2, self.renamed_preds, self.missing_preds = self._import_utils(predictor_filename, path, overwrite_mapping, subset, query=query, verbose=verbose)
        if df2 is not None:
            if 'BinResponseCount' not in df2.columns:
                df2['BinResponseCount'] = df2['BinPositives'] + df2['BinNegatives']
            df2['BinPropensity'] = df2['BinPositives'] / df2['BinResponseCount'] if df2 is not None else None
            df2['BinAdjustedPropensity'] = (0.5+df2['BinPositives']) / (1 + df2['BinResponseCount'])

        if df1 is not None and df2 is not None:
            total_missing = set(self.missing_model) & set(self.missing_preds) - set(df1.columns) - set(df2.columns) 
            if len(total_missing) > 0:
                print(f"""Missing required field values. 
                Please check if they are available in the data, 
                and supply a custom mapping if the naming is different from default. 
                Missing values: {total_missing}""")
                
        return df1, df2
    
    def _import_utils(self, name, path=None, overwrite_mapping=None, subset=True, query=None, verbose=True):
        if isinstance(name, str):
            df = cdh_utils.readDSExport(file=name, path=path, verbose=verbose)
        else: df = name

        if isinstance(df, pd.DataFrame):
            self.model_snapshots = True
            df.columns = self._capitalize(list(df.columns))
            df, renamed, missing = self._available_columns(df, overwrite_mapping) 
            if subset: df = df[renamed.values()]
            df = self._set_types(df)
            if query is not None:
                try:
                    df = self._apply_query(df, query)
                    if verbose: print(f"Query succesful for {name}.")
                except:
                    if verbose: print(f"""Query unsuccesful for {name}.
                    Maybe the filter you selected only applies to either model data or predictor data
                    and thus can't be succesful for the other one. That should be fine
                    as the other table is likely queried correctly.""")
            
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
            'Configuration': ['ConfigurationName'],
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
            'BinNegatives': ['BinNegatives'],
            'BinResponseCount': ['BinResponseCount'],
            'Type': ['Type'],
            'BinPositivesPercentage': ['BinPositivesPercentage'],
            'BinNegativesPercentage': ['BinNegativesPercentage'],
            'BinResponseCountPercentage': ['BinResponseCountPercentage']
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
        capitalizeEndWords = [  "ID", "Key", "Name", "Count", "Category",
                            "Time", "DateTime", "UpdateTime",
                            "ToClass", "Version", "Predictor", "Predictors", "Rate", "Ratio",
                            "Negatives", "Positives", "Threshold", "Error", "Importance",
                            "Type", "Percentage", "Index", "Symbol",
                            "LowerBound", "UpperBound", "Bins", "GroupIndex",
                            "ResponseCount", "NegativesPercentage", "PositivesPercentage",
                            "BinPositives", "BinNegatives", "BinResponseCount", "BinSymbol", 
                            "ResponseCountPercentage", "ConfigurationName" ]
        fields = [re.sub("^p(x|y|z)", "", field.lower()) for field in fields] 
        for word in capitalizeEndWords:
            fields = [re.sub(word, word, field, flags=re.I) for field in fields]
            fields = [field[:1].upper() + field[1:] for field in fields]
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
        flag=False
        for col in {'Issue', 'Group', 'Channel', 'Direction', 'ModelName'} & set(df.columns):
            df[col] = df[col].astype(str) 
        
        for col in {'Positives', 'Negatives'} & set(df.columns):
            try: 
                df[col] = df[col].astype(float).astype(int) 
            except:
                flag=True
                pass
        if flag: print("""Warning: there were some issues casting the Positives/Negatives values to int.
        Please make sure to check missing values or otherwise incorrect values for those columns.
        If any issues arise you may use the 'query' argument to filter those out.""")

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
        #NOTE Maybe we don't need to groupby predictorname
        if 'PredictorName' in df.columns:
            return df.sort_values('SnapshotTime').groupby(['ModelID', 'PredictorName', 'BinIndex']).last().reset_index()
        if 'ModelName' in df.columns: 
            return df.sort_values('SnapshotTime').groupby(['ModelID']).last()
        
    def get_combined_data(self, modelData:pd.DataFrame = None, predictorData:pd.DataFrame = None, last=True) -> pd.DataFrame:
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
        #TODO: actives only as parameter
        models = self.last(self.modelData) if last else self.modelData if modelData is None else modelData
        preds = self.last(self.predictorData) if last else self.predictorData if predictorData is None else predictorData
        combined = models.merge(preds, on='ModelID', how='right', suffixes=('', 'Bin'))
        return combined
    
    @staticmethod
    def defaultPredictorCategorization(name:str) -> str:
        raise NotImplemented

    @staticmethod
    def _apply_query(df, query: Union[str, dict] = None) -> pd.DataFrame:
        """Given an input pandas dataframe, it filters the dataframe based on input query
        
        Parameters
        ----------
        query: Union[str or dict]
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        Returns
        -------
        pd.DataFrame
            Filtered Pandas DataFrame
        """
        if query is not None:
            if isinstance(query, str):
                return df.query(query)

            df = df.reset_index(drop=True)
            if not isinstance(query, dict):
                raise TypeError('query must be a dict where values are lists')
            for key, val in query.items():
                if not type(val)==list:
                    raise ValueError('query values must be list')

            for col, val in query.items():
                df = df[df[col].isin(val)]
        return df

    def _subset_data(self, table:str, required_columns:set, query: Union[str, dict] = None, multi_snapshot:bool = False, last:bool = False, active_only:bool = False) -> pd.DataFrame:
        """Retrieves and subsets the data and performs some assertion checks

        Parameters
        ----------
        table : str
            Which table to retrieve from the ADMDatamart object 
            (modelData, predictorData or combinedData)
        required_columns : set
            Which columns we want to use for the visualisation
            Asserts those columns are in the data, and returns only those columns for efficiency
        query : Union[str, dict]
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        multi_snapshot : bool
            Whether to make sure there are multiple snapshots present in the data
            Sometimes required for visualisations over time
        last : bool
            Whether to subset on just the last known value for each model ID/predictor/bin
        active_only : bool
            Whether to subset on just the active predictors
        
        Returns
        -------
        pd.DataFrame
            The subsetted dataframe
        """
        df = getattr(self, table)

        assert required_columns.issubset(df.columns), f"The following columns are missing in the data: {required_columns - set(df.columns)}"
        
        df = self._apply_query(df, query)

        if multi_snapshot and not last: 
            assert df['SnapshotTime'].nunique() > 1, "There is only one snapshot, so this visualisation doesn't make sense."

        if last:
            df = self.last(df)
        
        if active_only and 'PredictorName' in df.columns:
            df = self._apply_query(df, "EntryType == 'Active'")

        return df[required_columns]
    
    @staticmethod
    def _create_sign_df(df:pd.DataFrame) -> pd.DataFrame:
        """Generates dataframe to show whether responses decreased/increased from day to day
        For a given dataframe where columns are dates and rows are model names,
        subtracts each day's value from the previous day's value per model. Then masks the data.
        If decreased (desired situtation), it will put 1 in the cell, if no change, it will
        put 0, and if decreased it will put -1. This dataframe then could be used in the heatmap
        
        Parameters
        ----------
        df: pd.DataFrame
            This typically is pivoted ModelData
        Returns
        -------
        pd.DataFrame
            The dataframe with signs for increase or decrease in day to day

        """
        vals = df.reset_index().values
        cols = df.columns
        _df = pd.DataFrame(np.hstack(
                (np.array([[vals[i,0]] for i in range(vals.shape[0])]),
                            np.array([vals[i,2:]-vals[i,1:-1]  for i in range(vals.shape[0])]))))
        _df.columns = cols
        _df.rename(columns={cols[0]:'ModelID'}, inplace=True)
        df_sign = _df.set_index('ModelID').mask(_df.set_index('ModelID')>0, 1)
        df_sign = df_sign.mask(df_sign<0, -1)
        df_sign[cols[0]] = 1
        return df_sign[cols].fillna(1)

    def _create_heatmap_df(self, df:pd.DataFrame, lookback:int=5, query:Union[str, dict]=None, fill_null_days:bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generates dataframes needed to plot calendar heatmap
        The method generates two dataframes where one is used to annotate the heatmap
        and the other is used to apply colors based on the sign dataframe.
        If there are multiple snapshots per day, the latest one will be selected
        
        Parameters
        ----------
        lookback : int
            Defines how many days to look back at data from the last snapshot
        query : Union[str, dict]
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        fill_null_days : bool
            If True, null values will be generated in the dataframe for
            days where there is no model snapshot

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Tuple of annotate and sign dataframes
        """

        df = self._apply_query(df, query)
        required_columns = {'ModelID', 'SnapshotTime', 'ResponseCount'}
        assert required_columns.issubset(df.columns)
        assert df['SnapshotTime'].nunique() > 1, "There is only one snapshot, so this visualisation doesn't make sense."
        
        df = df[['ModelID', 'SnapshotTime', 'ResponseCount']].sort_values('SnapshotTime').reset_index(drop=True)
        df['Date'] = pd.Series([i.date() for i in df['SnapshotTime']])
        df = df[df['Date']>(df['Date'].max()-timedelta(lookback))]
        if df.shape[0]<1:
            print("no data within lookback range")
            return pd.DataFrame()
        
        idx = df.groupby(['ModelID', 'Date'])['SnapshotTime'].transform(max)==df['SnapshotTime']
        df = df[idx]
        if fill_null_days:
            idx_date = pd.date_range(df['Date'].min(), df['Date'].max())
            df = df.set_index('Date').groupby('ModelID').apply(lambda d: d.reindex(idx_date)).drop(
                'ModelID', axis=1).reset_index('ModelID').reset_index().rename(columns={'index':'Date'})
            df['Date'] = df['Date'].dt.date
        df_annot = df.pivot(columns='Date', values='ResponseCount', index='ModelID')
        df_sign = self._create_sign_df(df_annot)
        return (df_annot, df_sign)

    @staticmethod
    def distribution_graph(df:pd.DataFrame, title:str, figsize:tuple) -> plt.figure:
        """Generic method to generate distribution graphs given data and graph size
        
        Parameters
        ----------
        df : pd.DataFrame
            The input data
        title : str
            Title of graph
        figsize : tuple
            Size of graph
            
        Returns
        -------
        plt.figure
            The generated figure
        """
        required_columns = {'BinIndex', 'BinSymbol', 'BinResponseCount', 'BinPropensity'}
        assert required_columns.issubset(df.columns)

        order = df.sort_values('BinIndex')['BinSymbol']
        fig, ax = plt.subplots(figsize=figsize)
        df['BinPropensity'] *= 100
        sns.barplot(x='BinSymbol', y='BinResponseCount', data=df, ax=ax, color='blue', order=order)
        ax1 = ax.twinx()
        ax1.plot(df.sort_values('BinIndex')['BinSymbol'], df.sort_values('BinIndex')['BinPropensity'], color='orange', marker='o')
        for i in ax.get_xmajorticklabels():
            i.set_rotation(90)
        labels = [i.get_text()[0:24]+'...' if len(i.get_text())>25 else i.get_text() for i in ax.get_xticklabels()]
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(labels)
        ax.set_ylabel('Responses')
        ax.set_xlabel('Range')
        ax1.set_ylabel('Propensity (%)')
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        patches = [mpatches.Patch(color='blue', label='Responses'), mpatches.Patch(color='orange', label='Propensity')]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.5, frameon=True)
        ax.set_title(title)
    
    @staticmethod
    def _calculate_impact_influence(df:pd.DataFrame, ModelID:str=None):
        def _ImpactInfluence(X):
            d = {}
            d['Impact(%)'] = X['absIc'].max()
            d['Influence(%)'] = (X['BinResponseCountPercentage']*X['absIc']/100).sum()
            return pd.Series(d)
        df = df.query("PredictorName != 'Classifier'").reset_index(drop=True)
        if ModelID is not None:
            df = df.query("ModelID == @ModelID")
        df['absIc'] = np.abs(df['BinPositivesPercentage'] - df['BinNegativesPercentage'])
        df = df.groupby(['ModelID', 'PredictorName']).apply(_ImpactInfluence).reset_index().merge(
            df[['ModelID', 'Issue', 'Group', 'Channel', 'Direction', 'ModelName']].drop_duplicates(), on='ModelID')
        return df.sort_values(['PredictorName', 'Impact(%)'], ascending=[False, False])
    
    def select_n():
        raise NotImplemented