import copy
import json
import logging
import os
from datetime import timedelta
from typing import Dict, NoReturn, Optional, Tuple, Union, Literal
from io import BytesIO
import pandas as pd
import polars as pl

from ..utils import cdh_utils
from ..utils.types import any_frame
from .ADMTrees import ADMTrees
from ..plots.plot_base import Plots
from ..plots.plots_mpl import ADMVisualisations as mpl_plot
from ..plots.plots_plotly import ADMVisualisations as plotly_plot

pl.toggle_string_cache = True


class ADMDatamart(Plots):
    """Main class for importing, preprocessing and structuring Pega ADM Datamart snapshot data.
    Gets all available data, properly names and merges into one main dataframe

    Parameters
    ----------
    path : str, default = "."
        The path of the data files
    overwrite_mapping : dict, default = None
        A dictionary to overwrite default feature names in the input data
    import_strategy: Literal['eager', 'lazy'], default = 'eager'
        Whether to import the file fully, or scan the file
        When data fits into memory, 'eager' is typically more efficient
        However, when data does not fit, the lazy methods typically allow
        you to still use the data.
    query : Union[str, dict], default = None
        The query to supply to _apply_query
        If a string, uses the default Pandas query function
        Else, a dict of lists where the key is the column name of the dataframe
        and the corresponding value is a list of values to keep in the dataframe
        Can be applied to an individual function, or used here to apply to the whole dataset
    plotting_engine : str, default = "plotly"
        Either 'mpl' for matplotlib, or 'plotly' for plotly.
        Determines what package to use for plotting.
        Can also be supplied to most plotting functions directly.


    Keyword arguments
    -----------------
    verbose : bool
        Whether to print out information during importing
    model_filename : str
        The name, or extended filepath, towards the model file
    predictor_filename : str
        The name, or extended filepath, towards the predictors file
    model_df : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame]
        Optional override to supply a dataframe instead of a file
    predictor_df : Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame]
        Optional override to supply a dataframe instead of a file
    subset : bool, default = True
        Whether to only select the renamed columns,
        set to False to keep all columns
    drop_cols = list
        Supply columns to drop from the dataframe
    include_cols = list
        Supply columns to include with the dataframe
    prequery : str
        A way to apply a query to the data before any preprocessing
        Uses the Pandas querying function, and beware that the columns
        have not been renamed, so use the original naming
    context_keys : list
        Which columns to use as context keys
    extract_treatment : str
        Treatments are typically hidden within the pyName column,
        extract_treatment can expand that cell to also show treatments.
        To extract, give the column name as the 'extract_treatment' argument

    Notes
    ----------------------------
    Depending on the importing function, typically it is possible to
    supply more arguments. For instance, you can set the csv separator
    for Polars' `scan_csv()` method.

    Attributes
    ----------
    modelData : pl.LazyFrame
        If available, holds the preprocessed data about the models
    predictorData : pl.LazyFrame
        If available, holds the preprocessed data about the predictor binning
    combinedData : pl.LazyFrame
        If both modelData and predictorData are available,
        holds the merged data about the models and predictors


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

    def __init__(
        self,
        path: str = ".",
        overwrite_mapping: Optional[dict] = None,
        import_strategy: Literal["eager", "lazy"] = "eager",
        query: Union[str, Dict[str, list]] = None,
        plotting_engine="plotly",
        **kwargs,
    ):

        self.context_keys = kwargs.pop(
            "context_keys", ["Channel", "Direction", "Issue", "Group"]
        )
        self.import_strategy = import_strategy
        self.modelData, self.predictorData = self.import_data(
            path,
            overwrite_mapping=overwrite_mapping,
            query=query,
            **kwargs,
        )

        if self.modelData is not None and self.predictorData is not None:
            self.combinedData = self.get_combined_data()
        else:
            if kwargs.get("verbose", True):
                print(
                    "Could not be combined. Do you have both model data and predictor data?"
                )
        self.context_keys = [
            key for key in self.context_keys if key in self.modelData.columns
        ]
        self.plotting_engine = plotting_engine
        super().__init__()

    @staticmethod
    def get_engine(plotting_engine):
        if not isinstance(plotting_engine, str):
            return plotting_engine
        elif plotting_engine == "plotly":
            return plotly_plot
        elif plotting_engine in {"mpl", "matplotlib", "seaborn"}:
            return mpl_plot
        else:
            msg = (
                f"Plotting engine {plotting_engine} not known. "
                "Please supply your own class with function names corresponding to plot_base.py "
                "or give one of the following strings: {'plotly', 'mpl', 'matplotlib', 'seaborn'}."
            )
            raise ValueError(msg)

    def import_data(
        self,
        path: Optional[str] = ".",
        overwrite_mapping: Optional[dict] = None,
        subset: bool = True,
        model_df: Optional[any_frame] = None,
        predictor_df: Optional[any_frame] = None,
        query: Union[str, Dict[str, list]] = None,
        **kwargs,
    ) -> pl.LazyFrame:
        """Method to automatically import & format the relevant data.

        The method first imports the model data, and then the predictor data.
        If model_df or predictor_df is supplied, it will use those instead
        After reading, some additional values (such as success rate) are
        automatically computed.
        Lastly, if there are missing columns from both datasets,
        this will be printed to the user if verbose is True.

        Parameters
        ----------
        path : str
            The path of the data files
            Default = current path (',')
        overwrite_mapping : dict
            A dictionary to overwrite default feature names in the input data
            Default = None
        subset : bool, default = True
            Whether to only select the renamed columns,
            set to False to keep all columns
        model_df : pd.DataFrame
            Optional override to supply a dataframe instead of a file
        predictor_df : pd.DataFrame
            Optional override to supply a dataframe instead of a file
        query : Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            The model data and predictor binning data as dataframes
        """
        verbose = kwargs.pop("verbose", True)
        model_filename = kwargs.pop("model_filename", "modelData")
        predictor_filename = kwargs.pop("predictor_filename", "predictorData")

        if model_df is not None:
            df1, self.renamed_model, self.missing_model = self._import_utils(
                name=model_df, subset=subset, query=query, verbose=verbose, **kwargs
            )
        else:
            df1, self.renamed_model, self.missing_model = self._import_utils(
                model_filename,
                path,
                overwrite_mapping,
                subset,
                query=query,
                verbose=verbose,
                **kwargs,
            )
        if df1 is not None:
            df1 = df1.with_column(
                (pl.col("Positives") / pl.col("ResponseCount"))
                .fill_nan(pl.lit(0))
                .alias("SuccessRate")
            )

        if predictor_df is not None:
            df2, self.renamed_preds, self.missing_preds = self._import_utils(
                name=predictor_df,
                subset=subset,
                query=query,
                verbose=verbose,
                **kwargs,
            )
        else:
            df2, self.renamed_preds, self.missing_preds = self._import_utils(
                predictor_filename,
                path,
                overwrite_mapping,
                subset,
                query=query,
                verbose=verbose,
                **kwargs,
            )
        if df2 is not None:
            if "BinResponseCount" not in df2.columns:
                df2 = df2.with_column(
                    (pl.col("BinPositives") + pl.col("BinNegatigves")).alias(
                        "BinResponseCount"
                    )
                )
            df2 = df2.with_columns(
                [
                    (pl.col("BinPositives") / pl.col("BinResponseCount")).alias(
                        "BinPropensity"
                    ),
                    (
                        (pl.col("BinPositives") + pl.lit(0.5))
                        / (pl.col("BinResponseCount") + pl.lit(1))
                    ).alias("BinAdjustedPropensity"),
                ]
            )

        if df1 is not None and df2 is not None:
            total_missing = set(self.missing_model) & set(self.missing_preds) - set(
                df1.columns
            ) - set(df2.columns)
            if len(total_missing) > 0 and verbose:
                print(
                    "Missing expected field values.\n",
                    "Please check if they are available in the data,\n",
                    "and supply a custom mapping if the naming is different from default.\n",
                    f"Missing values: {total_missing}",
                )
        if self.import_strategy == "eager":
            with pl.StringCache():
                if df1 is not None:
                    df1 = df1.collect().lazy()
                if df2 is not None:
                    df2 = df2.collect().lazy()
        return df1, df2

    def _import_utils(
        self,
        name: Union[str, any_frame],
        path: str = None,
        overwrite_mapping: dict = None,
        subset: bool = True,
        query: Union[str, Dict[str, list]] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[pl.LazyFrame, dict, dict]:
        """Handler function to interface to the cdh_utils methods

        Parameters
        ----------
        name : Union[str, pl.DataFrame]
            One of {modelData, predictorData}
            or a dataframe
        path: str, default = None
            The path of the data file
        overwrite_mapping : dict, default = None
            A dictionary to overwrite default feature names in the input data
        subset : bool, default = True
            Whether to only select the renamed columns,
            set to False to keep all columns
        query : Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        verbose : bool
            Whether to print out information during importing

        Keyword arguments
        -----------------
        drop_cols : list
            Supply columns to drop from the dataframe
        include_cols : list
            Supply columns to include with the dataframe
        prequery : str
            A way to apply a query to the data before any preprocessing
            Uses the Pandas querying function, and beware that the columns
            have not been renamed, so use the original naming
        extract_treatment : str
            Treatments are typically hidden within the pyName column,
            extract_treatment can expand that cell to also show treatments.
            To extract, give the column name as the 'extract_treatment' argument

        Additional keyword arguments
        -----------------
        See readDSExport in cdh_utils

        Returns
        -------
        (pl.LazyFrame, dict, dict)
            The requested dataframe,
            The renamed columns
            The columns missing in both dataframes
        """

        extract_col = kwargs.pop("extract_treatment", None)
        drop_cols = kwargs.pop("drop_cols", None)
        if isinstance(name, str) or isinstance(name, BytesIO):
            df = cdh_utils.readDSExport(
                filename=name, path=path, verbose=verbose, **kwargs
            )
        else:
            df = name
        if not isinstance(df, pl.LazyFrame):
            return None, None, None

        if not isinstance(overwrite_mapping, dict):
            overwrite_mapping = {}
        self.model_snapshots = True

        if kwargs.get("prequery", None) is not None:
            raise NotImplementedError("Not yet implemented for Polars version.")

        if drop_cols is not None:
            try:
                df = df.drop(drop_cols)
            except:
                if verbose:
                    print("Error dropping one or more columns")
                pass
        if extract_col is not None and extract_col in df.columns:
            df, overwrite_mapping = self.extract_treatments(
                df, overwrite_mapping, extract_col
            )
        df = df.rename(dict(zip(df.columns, cdh_utils._capitalize(df.columns))))
        df, renamed, missing = self._available_columns(df, overwrite_mapping, **kwargs)
        if subset:
            df = df.select(renamed.values())
        df = self._set_types(df)

        if query is not None:
            try:
                df = self._apply_query(df, query)
                if verbose:
                    print(f"Query succesful for {name}.")
            except:
                if verbose:
                    print(
                        f"""Query unsuccesful for {name}.
                Maybe the filter you selected only applies to either model data or predictor data
                and thus can't be succesful for the other one. That should be fine
                as the other table is likely queried correctly."""
                    )
        return df, renamed, missing

    def _available_columns(
        self, df: pl.LazyFrame, overwrite_mapping: Optional[dict] = None, **kwargs
    ) -> Tuple[pl.LazyFrame, dict, list]:
        """Based on the default names for variables, rename available data to proper formatting

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe
        overwrite_mapping : dict
            If given, adds 'search terms' to the default names to look for
            If an extra variable is given which is not in default_names, it will also be included

        Keyword arguments
        -----------------
        include_cols : list
            Supply columns to include with the dataframe

        Returns
        -------
        (pl.LazyFrame, dict, list)
            The original dataframe, but renamed for the found columns &
            The original and updated names for all renamed columns &
            The variables that were not found in the table
        """
        default_names = {
            "ModelID": ["ModelID"],
            "Issue": ["Issue"],
            "Group": ["Group"],
            "Channel": ["Channel"],
            "Direction": ["Direction"],
            "ModelName": ["Name", "ModelName"],
            "Positives": ["Positives"],
            "Configuration": ["ConfigurationName", "Configuration"],
            "ResponseCount": ["Response", "Responses", "ResponseCount"],
            "SnapshotTime": ["ModelSnapshot", "SnapshotTime"],
            "PredictorName": ["PredictorName"],
            "Performance": ["Performance"],
            "EntryType": ["EntryType"],
            "PredictorName": ["PredictorName"],
            "BinSymbol": ["BinSymbol"],
            "BinIndex": ["BinIndex"],
            "BinType": ["BinType"],
            "BinPositives": ["BinPositives"],
            "BinNegatives": ["BinNegatives"],
            "BinResponseCount": ["BinResponseCount"],
            "Type": ["Type"],
            "Contents": ["Contents"],
        }  # NOTE: these default names are already capitalized properly, with py/px/pz removed.

        if len(kwargs.get("include_cols", [])) > 0:
            for i in kwargs.pop("include_cols", []):
                overwrite_mapping[i] = i

        if overwrite_mapping is not None and len(overwrite_mapping) > 0:
            old_keys = list(overwrite_mapping.keys())
            new_keys = cdh_utils._capitalize(list(old_keys))
            for i, _ in enumerate(new_keys):
                overwrite_mapping[new_keys[i]] = overwrite_mapping.pop(old_keys[i])

            for key, name in overwrite_mapping.items():
                name = cdh_utils._capitalize([name])[0]
                default_names[key] = [name]

        variables = copy.deepcopy(default_names)
        for key, values in default_names.items():
            variables[key] = [name for name in values if name in df.columns]
        missing = [x for x, y in variables.items() if len(y) == 0]
        variables = {y[0]: x for x, y in variables.items() if len(y) > 0}
        df = df.rename(variables)

        return df, variables, missing

    @staticmethod
    def _set_types(df: any_frame) -> any_frame:
        """A method to change columns to their proper type

        Parameters
        ----------
        df : pl.LazyFrame
            The input dataframe
        verbose: bool, default = True
            Whether to print out issues with casting variable types

        Returns
        -------
        pl.LazyFrame
            The input dataframe, but the proper typing applied
        """
        retype = {
            pl.Categorical: ["Issue", "Group", "Channel", "Direction"],
            # pl.Int64: ["Positives", "Negatives", "ResponseCount"],
            pl.Float64: ["Performance"],
        }
        to_retype = []
        for type, cols in retype.items():
            for col in cols:
                if col in df.columns:
                    to_retype.append(pl.col(col).cast(type))
        df = df.with_columns(to_retype)
        if df.schema["SnapshotTime"] == pl.Utf8:
            df = df.with_column(
                pl.col("SnapshotTime").str.strptime(pl.Datetime, "%Y%m%dT%H%M%S.%f %Z")
            )
        return df

    def last(
        self, table="modelData", strategy: Literal["eager", "lazy"] = "eager"
    ) -> any_frame:
        """Convenience function to get the last values for a table

        Parameters
        ----------
        table : str, default = modelData
            Which table to get the last values for
            One of {modelData, predictorData, combinedData}
        strategy: Literal['eager', 'lazy']
            Wheter to return eager or lazy frame

        Returns
        -------
        pl.DataFrame | pl.LazyFrame
            The last snapshot for each model
        """

        if isinstance(table, pl.LazyFrame):
            df = self._last(table)

        elif isinstance(table, str):
            assert table in {"modelData", "predictorData", "combinedData"}
            df = self._last(getattr(self, table))
        else:
            print(table)
            print("wat", df)
        with pl.StringCache():
            return df if not strategy == "eager" else df.collect()

    @staticmethod
    def _last(df: any_frame) -> any_frame:
        """Method to retrieve only the last snapshot."""
        return df.filter(pl.col("SnapshotTime") == pl.col("SnapshotTime").max())

    def get_combined_data(
        self, last=True, strategy: Literal["eager", "lazy"] = "eager"
    ) -> any_frame:
        """Combines the model data and predictor data into one dataframe.

        Parameters
        ----------
        last:bool, default=True
            Whether to only use the last snapshot for each table

        Returns
        -------
        pl.LazyFrame
            The combined dataframe
        """
        models = self.last(self.modelData, "lazy") if last else self.modelData
        preds = self.last(self.predictorData, "lazy") if last else self.predictorData
        combined = models.join(preds, on="ModelID", how="inner", suffix="Bin")
        if strategy == "eager":
            with pl.StringCache():
                return combined.collect().lazy()
        else:
            return combined

    def save_data(self, path: str = ".") -> Tuple[os.PathLike, os.PathLike]:
        """Cache modelData and predictorData to files.

        Parameters
        ----------
        path : str
            Where to place the files

        Returns
        -------
        (os.PathLike, os.PathLike):
            The paths to the model and predictor data files
        """
        from datetime import datetime

        time = cdh_utils.toPRPCDateTime(datetime.now())
        if self.modelData is not None:
            modeldata_cache = cdh_utils.cache_to_file(
                self.modelData, path, name=f"cached_modelData_{time}"
            )
        if self.predictorData is not None:
            predictordata_cache = cdh_utils.cache_to_file(
                self.predictorData, path, name=f"cached_predictorData_{time}"
            )
        return modeldata_cache, predictordata_cache

    def _apply_query(self, df: any_frame, query: Union[str, dict] = None) -> any_frame:
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

            if isinstance(query, pl.Expr):
                return df.filter(query)

            if isinstance(query, str):
                print(
                    "Pandas query detected. This will be slow, and only works in eager mode ",
                    "because we turn the dataframe to pandas, query, and then turn it back ",
                    "to Polars. For faster performance, please pass a Polars Expression ",
                    "which can be used in the `filter()` method.",
                )
                if isinstance(df, pl.LazyFrame):
                    raise self.NotEagerError("Applying pandas queries")

            if not isinstance(query, dict):
                raise TypeError("query must be a dict where values are lists")
            for key, val in query.items():
                if not type(val) == list:
                    raise ValueError("query values must be list")

            for col, val in query.items():
                df = df[df[col].isin(val)]
        return df

    def extract_treatments(
        self,
        df,
        overwrite_mapping,
        extract_col,
        verbose=True,
    ):
        if verbose:
            print("Extracting treatments...")
        if self.import_strategy != "eager":
            raise self.NotEagerError("Extracting treatments")
        with pl.StringCache():
            df = df.collect()
        self.extracted = self._extract(df, extract_col)
        df = df.rename(dict(zip(df.columns, [i.lower() for i in df.columns])))
        df = df.drop(extract_col.lower())
        for column in self.extracted.columns:
            df.hstack([self.extracted.get_column(column)], in_place=True)
            if column.lower() != "pyname":
                overwrite_mapping[column] = column
        return df.lazy(), overwrite_mapping

    def _extract(self, df, extract_col):
        """Simple function to extract treatments from column"""

        try:
            df = df.select(pl.col(extract_col).apply(json.loads)).unnest(extract_col)
            df.columns = [col.lower() for col in df.columns]
            return df
        except:
            return df.select(pl.col(extract_col).apply(self.load_if_json)).unnest(
                extract_col
            )

    @staticmethod
    def load_if_json(input, defaultName="pyname"):
        """Either extracts the whole column, or just the json strings"""
        try:
            return json.loads(input)
        except:
            return {defaultName, input}

    class NotApplicableError(ValueError):
        pass

    class NotEagerError(ValueError):
        """Operation only possible in eager mode."""

        def __init__(
            self,
            operationType=None,
            defaultmsg="This operation is only possible in eager mode.",
        ):
            if operationType is not None:
                msg = f"{operationType} is only possible in eager mode"
            else:
                msg = defaultmsg
            super().__init__(msg)

    def discover_modelTypes(self, df: pl.LazyFrame, by="Configuration"):
        if self.import_strategy != "eager":
            raise self.NotEagerError("Discovering AGB models")

        def _getType(val):
            import zlib
            import base64

            return next(
                line.split('"')[-2].split(".")[-1]
                for line in zlib.decompress(base64.b64decode(val)).decode().split("\n")
                if line.startswith('  "_serialClass"')
            )

        if isinstance(df, pl.DataFrame):
            df = df.lazy()
        with pl.StringCache():
            types = (
                df.filter(pl.col("Modeldata").is_not_null())
                .groupby(by)
                .agg(pl.col("Modeldata").last())
                .collect()
                .with_column(pl.col("Modeldata").apply(lambda v: _getType(v)))
                .to_dicts()
            )
        return {key: value for key, value in [i.values() for i in types]}

    def get_AGB_models(
        self,
        last: bool = False,
        by: str = "Configuration",
        n_threads: int = 6,
        query: Union[str, dict] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Dict:
        """Method to automatically extract AGB models.

        Recommended to subset using the querying functionality
        to cut down on execution time, because it checks for each
        model ID. If you only have AGB models remaining after the query,
        it will only return proper AGB models.

        Parameters
        ----------
        last: bool, default = False
            Whether to only look at the last snapshot for each model
        by: str, default = 'Configuration'
            Which column to determine unique models with
        n_threads: int, default = 6
            The number of threads to use for extracting the models.
            Since we use multithreading, setting this to a reasonable value
            helps speed up the import.
        query: Union[str, dict], default = None
            The pandas query to apply to the modelData frame
        verbose: bool, default = False
            Whether to print out information while importing

        """
        df = self.modelData
        if "Modeldata" not in df.columns:
            raise ValueError(
                (
                    "Modeldata column not in the data. "
                    "Please make sure to include it by using the 'include_cols' "
                    "argument with your ADMDatamart call, or setting 'subset' to False."
                )
            )
        if query is not None:
            df = self._apply_query(df, query)

        modelTypes = self.discover_modelTypes(df)
        AGB_models = [
            model for model, type in modelTypes.items() if type.endswith("GbModel")
        ]
        logging.info(f"Found AGB models: {AGB_models}")
        df = df.filter(pl.col("Configuration").is_in(AGB_models))
        with pl.StringCache():
            if df["ModelID"].n_unique().collect() == 0:
                raise ValueError("No models found.")

        if last:
            return ADMTrees(
                self.last(df), n_threads=n_threads, verbose=verbose, **kwargs
            )
        else:
            return ADMTrees(df, n_threads=n_threads, verbose=verbose, **kwargs)

    @staticmethod
    def _create_sign_df(
        df: pl.LazyFrame, by, what="ResponseCount", every="1d", mask=True
    ) -> pl.LazyFrame:
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

        df = (
            df.with_column(pl.col("SnapshotTime").cast(pl.Date))
            .sort("SnapshotTime")
            .with_column(
                pl.col(what)
                .cast(pl.UInt64)
                .diff()
                .alias("Daily_increase")
                .over("ModelID")
            )
            .groupby_dynamic("SnapshotTime", every=every, by=by)
            .agg(pl.sum("Daily_increase").alias("Increase"))
            .collect()
            .pivot(index="SnapshotTime", columns=by, values="Increase")
            .lazy()
        )
        if mask:
            df = df.with_columns((pl.all().exclude("SnapshotTime").sign()))
        return df

    def _create_heatmap_df(
        self,
        df: pd.DataFrame,
        lookback: int = 5,
        query: Union[str, dict] = None,
        fill_null_days: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        required_columns = {"ModelID", "SnapshotTime", "ResponseCount"}
        assert required_columns.issubset(df.columns)

        df = (
            df[["ModelID", "SnapshotTime", "ResponseCount"]]
            .sort_values("SnapshotTime")
            .reset_index(drop=True)
        )
        df["Date"] = pd.Series([i.date() for i in df["SnapshotTime"]])
        df = df[df["Date"] > (df["Date"].max() - timedelta(lookback))]
        if df.shape[0] < 1:
            raise ValueError(("No data within lookback range"))

        idx = (
            df.groupby(["ModelID", "Date"])["SnapshotTime"].transform(max)
            == df["SnapshotTime"]
        )
        df = df[idx]
        if fill_null_days:
            idx_date = pd.date_range(df["Date"].min(), df["Date"].max())
            df = (
                df.set_index("Date")
                .groupby("ModelID")
                .apply(lambda d: d.reindex(idx_date))
                # .drop("ModelID", axis=1)
                # .reset_index("ModelID")
                .reset_index()
                .rename(columns={"index": "Date"})
            )
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df_annot = df.pivot(columns="Date", values="ResponseCount", index="ModelID")
        df_sign = self._create_sign_df(df_annot)
        return (df_annot, df_sign)

    @staticmethod
    def _calculate_impact_influence(
        df: pd.DataFrame, context_keys: list = None, ModelID: str = None
    ) -> pd.DataFrame:
        def _ImpactInfluence(X):
            d = {}
            X["BinResponseCountPercentage"] = (
                X["BinResponseCount"] / X["BinResponseCount"].sum()
            )
            X["BinPositivesPercentage"] = X["BinPositives"] / X["BinPositives"].sum()
            X["BinNegativesPercentage"] = X["BinNegatives"] / X["BinNegatives"].sum()
            X["absIc"] = np.abs(
                X["BinPositivesPercentage"] - X["BinNegativesPercentage"]
            )

            d["Impact(%)"] = X["absIc"].max()
            d["Influence(%)"] = (
                X["BinResponseCountPercentage"] * X["absIc"] / 100
            ).sum()
            return pd.Series(d)

        df = df.query("PredictorName != 'Classifier'").reset_index(drop=True)
        if ModelID is not None:
            df = df.query("ModelID == @ModelID")

        mergelist = ["ModelID", "ModelName"] + context_keys
        df = (
            df.groupby(["ModelID", "PredictorName"])
            .apply(_ImpactInfluence)
            .reset_index()
            .merge(
                df[mergelist].drop_duplicates(),
                on="ModelID",
            )
        )
        return df.sort_values(["PredictorName", "Impact(%)"], ascending=[False, False])

    def model_summary(
        self, by: str = "ModelID", query: Union[str, dict] = None, **kwargs
    ):
        """Convenience method to automatically generate a summary over models
        By default, it summarizes ResponseCount, Performance, SuccessRate & Positives by model ID.
        It also adds weighted means for Performance and SuccessRate,
        And adds the count of models without responses and the percentage.

        Parameters
        ----------
        by: str, default = ModelID
            By what column to summarize the models
        query : Union[str, dict]
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is column name in the dataframe
            and the corresponding value is a list of values to keep in the dataframe

        Returns
        -------
        pd.DataFrame:
            Groupby dataframe over all models
        """
        df = self._apply_query(self.modelData, query)
        data = self.last(df, strategy="lazy")

        aggcols = ["ResponseCount", "Performance", "SuccessRate", "Positives"]
        required_columns = set(aggcols).union({by})

        context_keys = kwargs.get("context_keys", self.context_keys)
        assert required_columns.issubset(set(data.columns) | set(context_keys))

        return (
            data.groupby(context_keys)
            .agg(
                [
                    pl.count(by).suffix("_count"),
                    pl.col([aggcols[0], aggcols[3]]).sum().suffix("_sum"),
                    pl.col(aggcols).max().suffix("_max"),
                    pl.col(aggcols).mean().suffix("_mean"),
                    (pl.col("ResponseCount") == 0)
                    .sum()
                    .alias("Count_without_responses"),
                    (
                        cdh_utils.weighed_performance_polars().alias(
                            "Performance_weighted"
                        )
                    ),
                    cdh_utils.weighed_average_polars(
                        "SuccessRate", "ResponseCount"
                    ).alias("SuccessRate_weighted"),
                ],
            )
            .with_column(
                (pl.col("Count_without_responses") / pl.col(f"{by}_count")).alias(
                    "Percentage_without_responses"
                )
            )
        )

    def pivot_df(
        self, df: pl.LazyFrame, by="ModelName", allow_collect=True
    ) -> pl.DataFrame:
        """Simple function to extract pivoted information"""
        if self.import_strategy == "lazy" and not allow_collect:
            raise ValueError("Only supported in eager mode.")
        df = df.filter(pl.col("PredictorName") != "Classifier")
        if by not in ["ModelID", "ModelName"]:
            df = df.groupby([by, "PredictorName"]).agg(
                cdh_utils.weighed_average_polars("PerformanceBin", "ResponseCount")
            )
        with pl.StringCache():
            df = (
                df.collect()
                .pivot(index=by, columns="PredictorName", values="PerformanceBin")
                .fill_null(0.5)
            )
        mod_order = (
            df.select(
                pl.concat_list(pl.col(pl.Float64))
                .arr.eval(pl.element().mean())
                .arr.get(0)
            )
            .select(pl.all().arg_sort(reverse=True))
            .to_series()
        )
        pred_order = [by] + [
            df.columns[i + 1]
            for i in 0
            + df.select(pl.col(pl.Float64).mean())
            .transpose()
            .select(pl.all().arg_sort(reverse=True))
            .to_series()
        ]
        return df[mod_order].select(pred_order)

    @staticmethod
    def response_gain_df(df: pd.DataFrame, by: str = "Channel") -> pd.DataFrame:
        """Simple function to extract the response gain per model"""
        return (
            df.groupby([by, "ModelID"])
            .agg(pl.max("ResponseCount"))
            .sort([by, "ResponseCount"], reverse=True)
            .with_columns(
                [
                    (pl.cumsum("ResponseCount") / pl.sum("ResponseCount"))
                    .over(by)
                    .alias("TotalResponseFraction"),
                    ((pl.col(by).cumcount() + 1) / pl.count("ResponseCount"))
                    .over(by)
                    .alias("TotalModelsFraction"),
                ]
            )
        )

    def models_by_positives_df(
        self, df: pd.DataFrame, by: str = "Channel", allow_collect=True
    ) -> pd.DataFrame:
        if self.import_strategy == "lazy" and not allow_collect:
            raise ValueError("Only supported in eager mode.")

        def orderedCut(
            s, label="PositivesBin", bins=list(range(0, 210, 10))
        ) -> pl.Series:
            _arg_sort = pl.Series(name="_sort", values=s.argsort())
            result = pl.cut(
                s, bins=[bins + [float("inf")]][0], category_label="PositivesBin"
            )
            return (
                result.select(
                    [
                        pl.col(label),
                        _arg_sort,
                    ]
                )
                .sort("_sort")
                .drop("_sort")
                .to_series()
            )

        with pl.StringCache():
            modelsByPositives = df.select([by, "Positives", "ModelID"]).collect()
        return (
            modelsByPositives.hstack(
                [
                    orderedCut(
                        modelsByPositives["Positives"],
                    )
                ]
            )
            .lazy()
            .groupby([by, "PositivesBin"])
            .agg([pl.min("Positives"), pl.n_unique("ModelID").alias("ModelCount")])
            .with_column(
                (pl.col("ModelCount") / (pl.sum("ModelCount").over(by))).alias(
                    "cumModels"
                )
            )
            .sort("PositivesBin")
        )

    def get_model_stats(self, last: bool = True) -> dict:
        if self.modelData is None:
            raise ValueError("No model data to analyze.")

        data = self.last(self.modelData) if last else self.modelData

        ret = dict()

        ret["models_n_snapshots"] = data.SnapshotTime.nunique()
        ret["models_total"] = len(data)
        ret["models_empty"] = data.query("ResponseCount == 0")
        ret["models_nopositives"] = data.query("ResponseCount > 0 & Positives == 0")
        ret["models_isimmature"] = data.query("Positives > 0 & Positives < 200")
        ret["models_noperformance"] = data.query(
            "Positives >= 200 & Performance == 0.5"
        )
        ret["models_n_nonperforming"] = (
            len(ret["models_empty"])
            + len(ret["models_nopositives"])
            + len(ret["models_isimmature"])
            + len(ret["models_noperformance"])
        )
        for key in self.context_keys:
            ret[f"models_missing_{key}"] = data.query(f'{key}=="NULL"')
        ret["models_bottom_left"] = data.query(
            "Performance == 0.5 & (SuccessRate.isnull() | SuccessRate == 0)"
        )
        ret["responses_per_model"] = pd.concat(
            [
                data.ResponseCount.value_counts().sort_index(),
                data.ResponseCount.value_counts(normalize=True)
                .mul(100)
                .round(1)
                .astype(str)
                .sort_index()
                + "%",
            ],
            axis=1,
        )

        self.model_stats = ret
        return ret

    def describe_models(self, **kwargs) -> NoReturn:
        """Convenience method to quickly summarize the models"""
        if not hasattr(self, "model_stats"):
            self.get_model_stats(last=True)
        assert {"models_empty"}.issubset(self.model_stats.keys())
        show_all_missing = kwargs.get("show_all_missing", True)
        nmodels = self.model_stats.get("models_total")

        ret = ""
        ret += f"""From all {nmodels} models:
{len(self.model_stats['models_empty'])} ({round(len(self.model_stats['models_empty'])/nmodels*100,2)}%) models have never recieved a response.
{len(self.model_stats['models_nopositives'])} ({round(len(self.model_stats['models_nopositives'])/nmodels*100,2)}%) models have been used but never recieved a 'positive' response.
{len(self.model_stats['models_isimmature'])} ({round(len(self.model_stats['models_isimmature'])/nmodels*100,2)}%) models are still in an 'immature' phase of learning (Positives between 1 and 200).
{len(self.model_stats['models_noperformance'])} ({round(len(self.model_stats['models_noperformance'])/nmodels*100,2)}%) models have recieved over 200 responses but still show minimum performance.
Meaning in total, {self.model_stats['models_n_nonperforming']} ({round(self.model_stats['models_n_nonperforming']/nmodels*100)}%) models do not perform as well as they could be.\n\n"""

        for key in self.model_stats.keys():
            if not isinstance(self.model_stats[key], int):
                if len(self.model_stats[key]) > 0 or show_all_missing:
                    ret += f"{len(self.model_stats[key])} ({round(len(self.model_stats[key])/nmodels*100,2)}%) {' '.join(key.split('_'))} attribute.\n"

        return ret