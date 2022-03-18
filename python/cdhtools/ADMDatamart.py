from . import cdh_utils

from .plots_plotly import ADMVisualisations as plotly_plot
from .plots_mpl import ADMVisualisations as mpl_plot
from .plot_base import Plots
from datetime import timedelta
import numpy as np
import pandas as pd
import re
import copy
from typing import Optional, Tuple, Union
import json


class ADMDatamart(Plots):
    """Main class for importing, preprocessing and structuring Pega ADM Datamart snapshot data."""

    def __init__(
        self,
        path: str = ".",
        overwrite_mapping: Optional[dict] = None,
        query=None,
        plotting_engine="plotly",
        **kwargs,
    ):
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
        self.modelData, self.predictorData = self.import_data(
            path, overwrite_mapping=overwrite_mapping, query=query, **kwargs
        )
        if self.modelData is not None and self.predictorData is not None:
            self.combinedData = self.get_combined_data()
        else:
            if kwargs.get("verbose", True):
                print(
                    "Could not be combined. Do you have both model data and predictor data?"
                )

        self.context_keys = kwargs.get(
            "context_keys", ["Channel", "Direction", "Issue", "Group"]
        )

        self.plotting_engine = plotting_engine

    @staticmethod
    def get_engine(plotting_engine):
        if plotting_engine == "plotly":
            return plotly_plot
        else:
            return mpl_plot

    def import_data(
        self,
        path: Optional[str] = ".",
        overwrite_mapping: Optional[dict] = None,
        subset: bool = True,
        model_df: Optional[pd.DataFrame] = None,
        predictor_df: Optional[pd.DataFrame] = None,
        query: Union[str, dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
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
            df1["SuccessRate"] = (
                df1["Positives"] / df1["ResponseCount"] if df1 is not None else None
            )
            df1["SuccessRate"] = df1["SuccessRate"].fillna(0)

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
                df2["BinResponseCount"] = df2["BinPositives"] + df2["BinNegatives"]
            df2["BinPropensity"] = (
                df2["BinPositives"] / df2["BinResponseCount"]
                if df2 is not None
                else None
            )
            df2["BinAdjustedPropensity"] = (0.5 + df2["BinPositives"]) / (
                1 + df2["BinResponseCount"]
            )

        if df1 is not None and df2 is not None:
            total_missing = set(self.missing_model) & set(self.missing_preds) - set(
                df1.columns
            ) - set(df2.columns)
            if len(total_missing) > 0 and verbose:
                print(
                    f"""Missing required field values. 
                Please check if they are available in the data, 
                and supply a custom mapping if the naming is different from default. 
                Missing values: {total_missing}"""
                )

        return df1, df2

    def _import_utils(
        self,
        name,
        path=None,
        overwrite_mapping=None,
        subset=True,
        query=None,
        verbose=True,
        **kwargs,
    ):
        extract_col = kwargs.pop("extract_treatment", None)

        if isinstance(name, str):
            df = cdh_utils.readDSExport(
                filename=name, path=path, verbose=verbose, **kwargs
            )
        else:
            df = name
        if not isinstance(df, pd.DataFrame):
            return None, None, None

        df = self.fix_pdc(df)

        if not isinstance(overwrite_mapping, dict):
            overwrite_mapping = {}
        self.model_snapshots = True

        if kwargs.get("prequery", None) is not None:
            try:
                df = df.query(kwargs.get("prequery"))
            except:
                if verbose:
                    print("Error with prequery")
                pass

        if kwargs.get("drop_cols", None) is not None:
            for i in kwargs.get("drop_cols"):
                try:
                    df.drop(i, axis=1, inplace=True)
                except:
                    if verbose:
                        print("Error dropping column", i)
                    pass

        if extract_col is not None and extract_col in df.columns:

            if verbose:
                print("Extracting treatments...")
            self.extracted = self.extract_treatments(df, extract_col)
            self.extracted.columns = ["ModelName", "Treatment"]
            df.loc[:, extract_col] = self.extracted["ModelName"]
            for column in self.extracted.columns:
                if column != "ModelName":
                    df.insert(0, column, self.extracted[column])
                    capitalized = self._capitalize([column, ""])[0]
                    overwrite_mapping[capitalized] = capitalized

        df.columns = self._capitalize(list(df.columns))
        df, renamed, missing = self._available_columns(df, overwrite_mapping)
        if subset:
            df = df[renamed.values()]
        df = self._set_types(df, verbose)
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
        self, df: pd.DataFrame, overwrite_mapping: Optional[dict] = None
    ) -> Tuple[pd.DataFrame, dict, list]:
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
            "ModelID": ["ModelID"],
            "Issue": ["Issue"],
            "Group": ["Group"],
            "Channel": ["Channel"],
            "Direction": ["Direction"],
            "ModelName": ["Name", "ModelName"],
            "Positives": ["Positives"],
            "Configuration": ["ConfigurationName"],
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
            "BinPositivesPercentage": ["BinPositivesPercentage"],
            "BinNegativesPercentage": ["BinNegativesPercentage"],
            "BinResponseCountPercentage": ["BinResponseCountPercentage"],
        }  # NOTE: these default names are already capitalized properly, with py/px/pz removed.

        if overwrite_mapping is not None and len(overwrite_mapping) > 0:
            old_keys = list(overwrite_mapping.keys())
            new_keys = self._capitalize(list(old_keys))
            for i, _ in enumerate(new_keys):
                overwrite_mapping[new_keys[i]] = overwrite_mapping.pop(old_keys[i])

            for key, name in overwrite_mapping.items():
                name = self._capitalize([name])[0]
                if key not in default_names.keys():
                    default_names[key] = [name]
                else:
                    default_names[key].insert(0, name)
        variables = copy.deepcopy(default_names)
        for key, values in default_names.items():
            variables[key] = [name for name in values if name in df.columns]
        missing = [x for x, y in variables.items() if len(y) == 0]
        variables = {y[0]: x for x, y in variables.items() if len(y) > 0}
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
        capitalizeEndWords = [
            "ID",
            "Key",
            "Name",
            "Treatment",
            "Count",
            "Category",
            "Class",
            "Time",
            "DateTime",
            "UpdateTime",
            "ToClass",
            "Version",
            "Predictor",
            "Predictors",
            "Rate",
            "Ratio",
            "Negatives",
            "Positives",
            "Threshold",
            "Error",
            "Importance",
            "Type",
            "Percentage",
            "Index",
            "Symbol",
            "LowerBound",
            "UpperBound",
            "Bins",
            "GroupIndex",
            "ResponseCount",
            "NegativesPercentage",
            "PositivesPercentage",
            "BinPositives",
            "BinNegatives",
            "BinResponseCount",
            "BinSymbol",
            "ResponseCountPercentage",
            "ConfigurationName",
        ]
        fields = [re.sub("^p(x|y|z)", "", field.lower()) for field in fields]
        for word in capitalizeEndWords:
            fields = [re.sub(word, word, field, flags=re.I) for field in fields]
            fields = [field[:1].upper() + field[1:] for field in fields]
        return fields

    @staticmethod
    def _set_types(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
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
        flag = False
        for col in {"Issue", "Group", "Channel", "Direction", "ModelName"} & set(
            df.columns
        ):
            df[col] = df[col].astype(str)

        for col in {"Positives", "Negatives", "ResponseCount"} & set(df.columns):
            try:
                df[col] = df[col].astype(float).astype(int)
            except:
                flag = True
                pass
        if flag and verbose:
            print(
                """Warning: there were some issues casting the Positives/Negatives values to int.
        Please make sure to check missing values or otherwise incorrect values for those columns.
        If any issues arise you may use the 'query' argument to filter those out."""
            )

        for col in {"Performance"} & set(df.columns):
            df[col] = df[col].astype(float)

        try:
            df["SnapshotTime"] = pd.to_datetime(df["SnapshotTime"])
        except Exception:
            if verbose:
                print("Warning: Unable to format timestamps.")

        return df

    def last(self, table="modelData") -> pd.DataFrame:

        if isinstance(table, pd.DataFrame):
            return self._last(table)

        if isinstance(table, str):
            assert table in {"modelData", "predictorData", "combinedData"}
            return self._last(getattr(self, table))

    @staticmethod
    def _last(df: pd.DataFrame) -> pd.DataFrame:
        """Property to retrieve only the last values for a given dataframe."""
        # NOTE Maybe we don't need to groupby predictorname
        if "PredictorName" in df.columns:
            return (
                df.sort_values("SnapshotTime")
                .groupby(["ModelID", "PredictorName", "BinIndex"])
                .last()
                .reset_index()
            )
        if "PredictorName" not in df.columns:
            return (
                df.sort_values("SnapshotTime").groupby(["ModelID"]).last().reset_index()
            )

    def get_combined_data(
        self,
        modelData: pd.DataFrame = None,
        predictorData: pd.DataFrame = None,
        last=True,
    ) -> pd.DataFrame:
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
        # TODO: actives only as parameter
        models = (
            self.last(self.modelData)
            if last
            else self.modelData
            if modelData is None
            else modelData
        )
        preds = (
            self.last(self.predictorData)
            if last
            else self.predictorData
            if predictorData is None
            else predictorData
        )
        combined = models.merge(preds, on="ModelID", how="right", suffixes=("", "Bin"))
        return combined

    @staticmethod
    def fix_pdc(df):
        if not list(df.columns) == ["pxObjClass", "pxResults"]:
            return df
        df = pd.json_normalize(df["pxResults"]).dropna()
        df = df.rename(columns={"ModelName": "Configuration"})
        return df.reset_index(drop=True)

    @staticmethod
    def defaultPredictorCategorization(name: str) -> str:
        raise NotImplementedError("Will be added soon.")

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
                raise TypeError("query must be a dict where values are lists")
            for key, val in query.items():
                if not type(val) == list:
                    raise ValueError("query values must be list")

            for col, val in query.items():
                df = df[df[col].isin(val)]
        return df

    @staticmethod
    def load_if_json(extracted, extract_col="pyname"):
        try:
            ret = json.loads(extracted)
            return {k.lower(): v for k, v in ret.items()}
        except:
            return {extract_col.lower(): extracted}

    def extract_treatments(self, df, extract_col):
        try:
            df = pd.json_normalize(df[extract_col].apply(json.loads), max_level=1)
            df.columns = [col.lower() for col in df.columns]
            return df
        except:
            loaded = df[extract_col].apply(self.load_if_json, extract_col)
            return pd.json_normalize(loaded)

    class NotApplicableError(ValueError):
        pass

    @staticmethod
    def _create_sign_df(df: pd.DataFrame) -> pd.DataFrame:
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
        _df = pd.DataFrame(
            np.hstack(
                (
                    np.array([[vals[i, 0]] for i in range(vals.shape[0])]),
                    np.array(
                        [vals[i, 2:] - vals[i, 1:-1] for i in range(vals.shape[0])]
                    ),
                )
            )
        )
        _df.columns = cols
        _df.rename(columns={cols[0]: "ModelID"}, inplace=True)
        df_sign = _df.set_index("ModelID").mask(_df.set_index("ModelID") > 0, 1)
        df_sign = df_sign.mask(df_sign < 0, -1)
        df_sign[cols[0]] = 1
        return df_sign[cols].fillna(1)

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
        df: pd.DataFrame, context_keys=None, ModelID: str = None
    ):
        def _ImpactInfluence(X):
            d = {}
            d["Impact(%)"] = X["absIc"].max()
            d["Influence(%)"] = (
                X["BinResponseCountPercentage"] * X["absIc"] / 100
            ).sum()
            return pd.Series(d)

        df = df.query("PredictorName != 'Classifier'").reset_index(drop=True)
        if ModelID is not None:
            df = df.query("ModelID == @ModelID")
        df["absIc"] = np.abs(
            df["BinPositivesPercentage"] - df["BinNegativesPercentage"]
        )
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

    def model_summary(self, by="ModelID", query=None, **kwargs):
        """Convenience method to automatically generate a summary over models
        By default, it summarizes ResponseCount, Performance, SuccessRate & Positives by model ID.
        It also adds weighted means for Performance and SuccessRate,
        And adds the count of models without responses and the percentage.

        Parameters
        ----------
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
        data = self.last(df).reset_index()

        required_columns = {
            by,
            "ResponseCount",
            "Performance",
            "SuccessRate",
            "Positives",
        }

        context_keys = kwargs.get("context_keys", self.context_keys)
        assert required_columns.issubset(set(data.columns) | set(context_keys))

        def weighed_average(grp, weights="ResponseCount"):
            return grp.multiply(grp[weights], axis=0).sum() / grp[weights].sum()

        summary = data.groupby(context_keys).agg(
            {
                by: ["count"],
                "ResponseCount": ["sum", "mean", "max"],
                "Performance": ["max", "mean"],
                "SuccessRate": ["max", "mean"],
                "Positives": ["max", "mean", "sum"],
            }
        )

        noresponses = (
            data.query("ResponseCount==0")
            .groupby(context_keys)[by]
            .count()
            .rename("noresponse")
            .sort_values(ascending=False)
        )
        noresponses = pd.DataFrame(noresponses)
        noresponses.columns = [[by], ["count_without_responses"]]

        summary = pd.concat([summary, noresponses], axis=1, join="outer")
        temp = summary.pop((by, "count_without_responses"))

        weighted = data.groupby(context_keys)[
            ["Performance", "SuccessRate", "ResponseCount"]
        ].apply(weighed_average)[["Performance", "SuccessRate"]]
        weighted_perf = weighted["Performance"]
        weighted_perf.columns = [["Performance"], ["Weighted mean"]]
        weighted_succ = weighted["SuccessRate"]
        weighted_succ.columns = [["SuccessRate"], ["Weighted mean"]]

        summary.insert(1, (by, "count_without_responses"), temp)
        summary.insert(
            2,
            (by, "percentage_without_responses"),
            summary[(by, "count_without_responses")] / summary[(by, "count")],
        )
        summary.insert(8, ("Performance", "weighted_mean"), weighted_perf)
        summary.insert(11, ("SuccessRate", "weighted_mean"), weighted_succ)

        summary = summary.fillna(0)
        summary[("Performance", "weighted_mean")] = summary[
            ("Performance", "weighted_mean")
        ].replace(0, 0.5)

        return summary

    @staticmethod
    def pivot_df(df):
        df1 = df.query('PredictorName != "Classifier"')
        pivot_df = df1.pivot_table(
            index="ModelName", columns="PredictorName", values="PerformanceBin"
        )
        dedup = df1[["ModelName", "PredictorName", "PerformanceBin"]].drop_duplicates()
        pred_order = list(
            dedup.groupby("PredictorName")
            .agg({"PerformanceBin": "mean"})
            .fillna(0)
            .sort_values("PerformanceBin", ascending=False)
            .index
        )
        mod_order = list(
            dedup.groupby("ModelName")
            .agg({"PerformanceBin": "mean"})
            .fillna(0)
            .sort_values("PerformanceBin", ascending=False)
            .index
        )
        pivot_df = (pivot_df[pred_order]).reindex(mod_order)
        return pivot_df

    @staticmethod
    def response_gain_df(df, by="Channel"):
        responseGainData = (
            df.groupby([by, "ModelID"])
            .agg({"ResponseCount": "max"})
            .sort_values([by, "ResponseCount"], ascending=False)
        )
        responseGainData.loc[:, "TotalResponseFraction"] = responseGainData.groupby(by)[
            "ResponseCount"
        ].transform(pd.Series.cumsum) / responseGainData.groupby(by)[
            "ResponseCount"
        ].transform(
            pd.Series.sum
        )
        responseGainData = responseGainData.fillna(0)
        responseGainData.loc[:, "TotalModelsFraction"] = (
            responseGainData.groupby(by).cumcount() + 1
        ) / responseGainData.groupby(by).count()["ResponseCount"]
        responseGainData = responseGainData.reset_index()
        return responseGainData

    @staticmethod
    def models_by_positives_df(df, by="Channel"):
        modelsByPositives = df[[by, "Positives", "ModelID"]]
        modelsByPositives.loc[:, "PositivesBin"] = pd.cut(
            modelsByPositives["Positives"],
            bins=[list(range(0, 210, 10)) + [np.inf]][0],
            right=False,
        )
        order = modelsByPositives["PositivesBin"].sort_values()
        order = order.unique().astype(str)
        modelsByPositives.loc[:, "PositivesBin"] = (
            modelsByPositives["PositivesBin"].astype(str).astype("category")
        )
        modelsByPositives.loc[:, "PositivesBin"] = modelsByPositives[
            "PositivesBin"
        ].cat.set_categories(order)

        modelsByPositives = (
            modelsByPositives.groupby([by, "PositivesBin"])
            .agg(Positives=("Positives", "min"), ModelCount=("ModelID", "nunique"))
            .fillna(0)
        )
        modelsByPositives["cumModels"] = modelsByPositives[
            "ModelCount"
        ] / modelsByPositives.groupby(by)["ModelCount"].transform(pd.Series.sum)
        modelsByPositives = modelsByPositives.reset_index()
        modelsByPositives = modelsByPositives.sort_values("PositivesBin")
        return modelsByPositives

    def get_model_stats(self, last=True):
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

    def describe_models(self, **kwargs):
        if not hasattr(self, "model_stats"):
            self.get_model_stats(last=True)
        assert {"models_total", "models_empty"}.issubset(self.model_stats.keys())
        show_all_missing = kwargs.pop("show_all_missing", True)
        ret = ""
        nmodels = self.model_stats.pop("models_total")
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
