from typing import Union
import pandas as pd
from .plots_mpl import ADMVisualisations as mpl
from .plots_plotly import ADMVisualisations as plotly


class Plots:
    def _subset_data(
        self,
        table: str,
        required_columns: set,
        query: Union[str, dict] = None,
        multi_snapshot: bool = False,
        last: bool = False,
        active_only: bool = False,
    ) -> pd.DataFrame:
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
        assert hasattr(
            self, table
        ), f"This visualisation requires {table}, but that table isn't in this dataset."

        df = getattr(self, table)

        assert required_columns.issubset(
            df.columns
        ), f"The following columns are missing in the data: {required_columns - set(df.columns)}"

        df = self._apply_query(df, query)

        if multi_snapshot and not last:
            assert (
                df["SnapshotTime"].nunique() > 1
            ), "There is only one snapshot, so this visualisation doesn't make sense."

        if last:
            df = self.last(df)

        if active_only and "PredictorName" in df.columns:
            df = self._apply_query(df, "EntryType == 'Active'")

        return df[list(required_columns)]

    def plotPerformanceSuccessRateBubbleChart(
        self,
        last=True,
        add_bottom_left_text=True,
        to_html=False,
        file_title: str = None,
        file_path: str = None,
        query: Union[str, dict] = None,
        show_each=False,
        facets=None,
        **kwargs,
    ):
        plotting_engine = self.get_engine(
            kwargs.pop("plotting_engine", self.plotting_engine)
        )

        table = "modelData"
        required_columns = {
            "ModelID",
            "Performance",
            "SuccessRate",
            "ResponseCount",
            "ModelName",
        }.union(set(self.context_keys))

        df = self._subset_data(
            table=table, required_columns=required_columns, query=query, last=last
        )
        df[["Performance", "SuccessRate"]] = df[["Performance", "SuccessRate"]].apply(
            lambda x: round(x * 100, kwargs.pop("round", 5))
        )  # fix to use .loc

        if kwargs.pop("return_df", False):
            return df

        return plotting_engine.PerformanceSuccessRateBubbleChart(
            df=df,
            add_bottom_left_text=add_bottom_left_text,
            to_html=to_html,
            file_title=to_html,
            file_path=file_path,
            query=query,
            show_each=show_each,
            facets=facets,
            context_keys=self.context_keys,
            **kwargs,
        )

    def plotPerformanceAndSuccessRateOverTime(
        self, day_inverval=7, query=None, **kwargs
    ):
        if kwargs.get("plotting_engine") != "matplotlib":
            print("Plot is only available in matplotlib.")

        table = "modelData"
        multi_snapshot = True
        required_columns = {
            "ModelID",
            "ModelName",
            "SnapshotTime",
            "ResponseCount",
            "Performance",
            "SuccessRate",
        }.union(set(self.context_keys))
        df = self._subset_data(
            table, required_columns, query, multi_snapshot=multi_snapshot
        )
        return mpl.PerformanceAndSuccessRateOverTime(df, day_interval=7, **kwargs)

    def plotOverTime(
        self,
        metric="Performance",
        by="ModelID",
        to_html=False,
        file_title=None,
        file_path=None,
        query: Union[str, dict] = None,
        show_each=False,
        facets=None,
        **kwargs,
    ):
        plotting_engine = self.get_engine(
            kwargs.pop("plotting_engine", self.plotting_engine)
        )

        table = "modelData"
        multi_snapshot = True
        required_columns = {
            "ModelID",
            "ModelName",
            "SnapshotTime",
            "ResponseCount",
            "Performance",
            "SuccessRate",
            "Positives",
        }.union(set(self.context_keys))
        df = self._subset_data(
            table, required_columns, query, multi_snapshot=multi_snapshot
        )

        if kwargs.pop("return_df", False):
            return df

        return plotting_engine.OverTime(
            df=df,
            metric=metric,
            by=by,
            to_html=to_html,
            file_title=file_title,
            file_path=file_path,
            query=query,
            show_each=show_each,
            facets=facets,
            **kwargs,
        )

    def plotResponseCountMatrix(
        self, lookback=15, fill_null_days=False, query=None, **kwargs
    ):
        table = "modelData"
        multi_snapshot = True
        required_columns = {"ModelID", "ModelName", "SnapshotTime", "ResponseCount"}
        df = self._subset_data(
            table, required_columns, query=query, multi_snapshot=multi_snapshot
        )
        assert (
            lookback < df["SnapshotTime"].nunique()
        ), f"Lookback ({lookback}) cannot be larger than the number of snapshots {df['SnapshotTime'].nunique()}"

        annot_df, heatmap_df = self._create_heatmap_df(
            df, lookback, query=None, fill_null_days=fill_null_days
        )
        heatmap_df = (
            heatmap_df.reset_index()
            .merge(
                df[["ModelID", "ModelName"]].drop_duplicates(), on="ModelID", how="left"
            )
            .drop("ModelID", axis=1)
            .set_index("ModelName")
        )
        annot_df = (
            annot_df.reset_index()
            .merge(
                df[["ModelID", "ModelName"]].drop_duplicates(), on="ModelID", how="left"
            )
            .drop("ModelID", axis=1)
            .set_index("ModelName")
        )
        if kwargs.pop("return_df", False):
            return df
        return mpl.ResponseCountMatrix(
            annot_df=annot_df, heatmap_df=heatmap_df, query=query, **kwargs
        )

    def plotPropositionSuccessRates(
        self,
        metric="SuccessRate",
        by="ModelName",
        show_error=True,
        to_html=False,
        file_title=None,
        file_path=None,
        query: Union[str, dict] = None,
        show_each=False,
        facets=None,
        **kwargs,
    ):
        plotting_engine = self.get_engine(
            kwargs.get("plotting_engine", self.plotting_engine)
        )

        table = "modelData"
        last = True
        required_columns = (
            {"ModelID", "ModelName", "SuccessRate"}
            .union(self.context_keys)
            .union({metric})
        )
        df = self._subset_data(table, required_columns, query, last=last)

        if kwargs.pop("return_df", False):
            return df

        return plotting_engine.PropositionSuccessRates(
            df=df,
            metric=metric,
            by=by,
            show_error=show_error,
            to_html=to_html,
            file_title=file_title,
            file_path=file_path,
            query=query,
            show_each=show_each,
            facets=facets,
            **kwargs,
        )

    def plotScoreDistribution(
        self, by="ModelID", show_zero_responses=False, query=None, **kwargs
    ):
        plotting_engine = self.get_engine(
            kwargs.get("plotting_engine", self.plotting_engine)
        )()
        table = "combinedData"
        required_columns = {
            "PredictorName",
            "ModelName",
            "BinIndex",
            "BinSymbol",
            "BinResponseCount",
            "BinPropensity",
            "ModelID",
        }
        df = self._subset_data(table, required_columns, query)

        df = df[df["PredictorName"] == "Classifier"]
        df = df.groupby("ModelID")
        if df.ngroups > 10:
            if (
                input(
                    f"""WARNING: you are about to create {df.ngroups} plots because there are that many models. 
            This will take a while, and will probably slow down your system. Are you sure? Type 'Yes' to proceed."""
                )
                != "Yes"
            ):
                print(
                    "Cancelling. Set your 'query' parameter more strictly to generate fewer images"
                )
                return None

        if kwargs.pop("return_df", False):
            return df

        return plotting_engine.ScoreDistribution(
            df=df, show_zero_responses=show_zero_responses, query=query
        )

    def plotPredictorBinning(
        self,
        predictors: list = None,
        modelid: str = None,
        to_html=False,
        file_title=None,
        file_path=None,
        show_each=False,
        query=None,
        facets=None,
        **kwargs,
    ):
        plotting_engine = self.get_engine(
            kwargs.get("plotting_engine", self.plotting_engine)
        )()
        table = "combinedData"
        last = True
        required_columns = {
            "PredictorName",
            "ModelName",
            "BinIndex",
            "BinSymbol",
            "BinResponseCount",
            "BinPropensity",
            "ModelID",
        }
        df = self._subset_data(table, required_columns, query, last=last)
        if modelid is not None:
            df = df.query(f"ModelID == '{modelid}'")
        if predictors:
            df = df.query(f"PredictorName in {predictors}")

        model = df["ModelID"].unique()
        assert len(model) == 1, "Please only supply one model ID"

        modelName = model[0]

        if kwargs.pop("return_df", False):
            return df
        return plotting_engine.PredictorBinning(
            df=df,
            modelName=modelName,
            predictors=predictors,
            modelid=modelid,
            show_each=show_each,
        )

    def plotPredictorPerformance(
        self,
        top_n=0,
        to_html=False,
        file_title=None,
        file_path=None,
        show_each=False,
        query=None,
        facets=None,
        **kwargs,
    ):
        plotting_engine = self.get_engine(
            kwargs.get("plotting_engine", self.plotting_engine)
        )
        table = "combinedData"
        last = True
        required_columns = {"Channel", "PredictorName", "PerformanceBin", "Type"}
        df = self._subset_data(table, required_columns, query, last=last)
        df = df.query("PredictorName != 'Classifier'")

        if top_n > 0:
            topn = (
                df.groupby(["Channel", "PredictorName"])["PerformanceBin"]
                .mean()
                .sort_values(ascending=False)
                .head(top_n)
                .index.get_level_values(1)
                .tolist()
            )

            df = df.query(f"PredictorName == {topn}")
        asc = plotting_engine.__module__.split(".")[1] == "plots_mpl"
        order = (
            df.groupby("PredictorName")["PerformanceBin"]
            .mean()
            .fillna(0)
            .sort_values(ascending=asc)[::-1]
            .index
        )

        df.loc[:, "Legend"] = [
            i.split(".")[0] if len(i.split(".")) > 0 else "Primary"
            for i in df["PredictorName"]
        ]  # TODO: fix with separate function

        if kwargs.pop("return_df", False):
            return df

        return plotting_engine.PredictorPerformance(
            df=df,
            order=order,
            to_html=to_html,
            file_title=file_title,
            file_path=file_path,
            show_each=show_each,
            query=query,
            facets=facets,
            **kwargs,
        )

    def plotPredictorPerformanceHeatmap(
        self,
        top_n=0,
        to_html=False,
        file_title=None,
        file_path=None,
        show_each=False,
        query=None,
        facets=None,
        **kwargs,
    ):
        plotting_engine = self.get_engine(
            kwargs.get("plotting_engine", self.plotting_engine)
        )
        table = "combinedData"
        required_columns = {"PredictorName", "ModelName", "PerformanceBin"}
        df = self._subset_data(table, required_columns, query, last=True)
        df = df[df["PredictorName"] != "Classifier"].reset_index(drop=True)

        df = self.pivot_df(df)
        if top_n > 0:
            df = df.iloc[:, :top_n]

        if kwargs.pop("return_df", False):
            return df

        return plotting_engine.PredictorPerformanceHeatmap(
            df,
            to_html=to_html,
            file_title=file_title,
            file_path=file_path,
            show_each=show_each,
            query=query,
            facets=facets,
            **kwargs,
        )

    def plotImpactInfluence(self, ModelID=None, query=None, **kwargs):
        if kwargs.get("plotting_engine") != "mpl":
            print("Plot is only available in matplotlib.")

        table = "combinedData"
        last = True
        required_columns = {
            "ModelID",
            "PredictorName",
            "ModelName",
            "PerformanceBin",
            "BinPositivesPercentage",
            "BinNegativesPercentage",
            "BinResponseCountPercentage",
            "Issue",
            "Group",
            "Channel",
            "Direction",
        }
        df = self._subset_data(table, required_columns, query, last=last).reset_index()
        df = (
            self._calculate_impact_influence(df, ModelID=ModelID)[
                ["ModelID", "PredictorName", "Impact(%)", "Influence(%)"]
            ]
            .set_index(["ModelID", "PredictorName"])
            .stack()
            .reset_index()
            .rename(columns={"level_2": "metric", 0: "value"})
        )

        if kwargs.pop("return_df", False):
            return df

        return mpl.ImpactInfluence(df=df, ModelID=ModelID, **kwargs)

    def plotResponseGain(
        self,
        by="Channel",
        to_html=False,
        file_title=None,
        file_path=None,
        show=False,
        query=None,
        **kwargs,
    ):
        if kwargs.get("plotting_engine") != "plotly":
            print("Plot is only available in Plotly.")

        table = "modelData"
        last = True
        required_columns = {by, "ResponseCount", "ModelID"}
        df = self._subset_data(table, required_columns, query, last=last)
        df = self.response_gain_df(df, by=by)

        if kwargs.pop("return_df", False):
            return df

        return plotly.ResponseGain(
            df,
            by,
            to_html=to_html,
            file_title=file_title,
            file_path=file_path,
            show=show,
            query=query,
            **kwargs,
        )

    def plotModelsByPositives(
        self,
        by="Channel",
        to_html=False,
        file_title=None,
        file_path=None,
        show=False,
        query=None,
        **kwargs,
    ):
        if kwargs.get("plotting_engine") != "plotly":
            print("Plot is only available in Plotly.")
        table = "modelData"
        last = True
        required_columns = {by, "Positives", "ModelID"}
        df = self._subset_data(table, required_columns, query, last=last)
        df = self.models_by_positives_df(df, by=by)
        if kwargs.pop("return_df", False):
            return df
        return plotly.ModelsByPositives(
            df,
            by="Channel",
            to_html=to_html,
            file_title=file_title,
            file_path=file_path,
            show=show,
            query=query,
            **kwargs,
        )

    def plotTreeMap(
        self,
        color_var="performance_weighted",
        by="ModelID",
        value_in_text=True,
        midpoint=None,
        to_html=False,
        file_title=None,
        file_path=None,
        show=False,
        query=None,
        **kwargs,
    ):
        if kwargs.get("plotting_engine") != "plotly":
            print("Plot is only available in Plotly.")
        df = self.model_summary(by=by, query=query)
        df = df[
            [
                (by, "count"),
                (by, "percentage_without_responses"),
                ("ResponseCount", "sum"),
                ("SuccessRate", "mean"),
                ("Performance", "weighted_mean"),
                ("Positives", "sum"),
            ]
        ]
        df = df.reset_index()
        df.columns = self.context_keys + [
            "Model count",
            "Percentage without responses",
            "Response Count sum",
            "Success Rate mean",
            "Performance weighted mean",
            "Positives sum",
        ]
        if "OmniChannel" in df["Issue"].unique():
            print(
                "WARNING: This plot does not work for OmniChannel models. For that reason, we filter those out by default."
            )
            df = df.query('Issue != "OmniChannel"')

        defaults = {
            "responsecount": [
                "Response Count sum",
                "Model count",
                "Responses per model, per context key combination",
                False,
                False,
                None,
            ],
            "responsecount_log": [
                "Response Count sum",
                "Model count",
                "Log responses per model, per context key combination",
                False,
                True,
                None,
            ],
            "positives": [
                "Positives sum",
                "Model count",
                "Positives per model, per context key combination",
                False,
                False,
                None,
            ],
            "positives_log": [
                "Positives sum",
                "Model count",
                "Log Positives per model, per context key combination",
                False,
                True,
                None,
            ],
            "percentage_without_responses": [
                "Percentage without responses",
                "Model count",
                "Percentage without responses, per context key combination",
                True,
                False,
                None,
            ],
            "performance_weighted": [
                "Performance weighted mean",
                "Model count",
                "Weighted mean performance, per context key combination",
                False,
                False,
                None,
            ],
            "successrate": [
                "Success Rate mean",
                "Model count",
                "Success rate, per context key combination",
                False,
                False,
                0.5,
            ],
        }

        if isinstance(color_var, int):
            color_var = list(defaults.keys())[color_var]
        color = kwargs.get("color_col", defaults[color_var][0])
        values = kwargs.get("groupby_col", defaults[color_var][1])
        title = kwargs.get("title", defaults[color_var][2])
        reverse_scale = kwargs.get("reverse_scale", defaults[color_var][3])
        log = kwargs.get("log", defaults[color_var][4])
        if midpoint is not None:
            midpoint = defaults[color_var][5]

        format = "%" if color in list(defaults.keys())[4:] else ""

        return plotly.TreeMap(
            df=df,
            color=color,
            values=values,
            title=title,
            reverse_scale=reverse_scale,
            log=log,
            midpoint=midpoint,
            format=format,
            context_keys=self.context_keys,
            value_in_text=value_in_text,
            to_html=to_html,
            file_title=file_title,
            file_path=file_path,
            show=show,
            query=query,
            **kwargs,
        )