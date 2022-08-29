from typing import Optional, Union, Dict
import pandas as pd
from .plots_mpl import ADMVisualisations as mpl
from .plots_plotly import ADMVisualisations as plotly
import matplotlib.pyplot as plt
import plotly.graph_objs as go


class Plots:
    def __init__(self):
        self.hasModels = self.modelData is not None
        self.hasPredictorBinning = self.predictorData is not None
        self.hasCombined = hasattr(self, "combinedData")
        if self.hasModels:
            self.hasMultipleSnapshots = self.modelData["SnapshotTime"].nunique() > 1

    @property
    def AvailableVisualisations(self):
        return pd.DataFrame.from_dict(
            {
                "plotPerformanceSuccessRateBubbleChart": [1, 0, 0],
                "plotPerformanceAndSuccessRateOverTime": [1, 0, 1],
                "plotOverTime": [1, 0, 1],
                "plotResponseCountMatrix": [1, 0, 1],
                "plotPropositionSuccessRates": [1, 0, 0],
                "plotScoreDistribution": [1, 1, 0],
                "plotPredictorBinning": [1, 1, 0],
                "plotPredictorPerformance": [1, 1, 0],
                "plotPredictorPerformanceHeatmap": [1, 1, 0],
                "plotImpactInfluence": [1, 1, 0],
                "plotResponseGain": [1, 0, 0],
                "plotModelsByPositives": [1, 0, 0],
                "plotTreeMap": [1, 0, 0],
            },
            orient="index",
            columns=["modelData", "predictorData", "Multiple snapshots"],
        )

    @property
    def ApplicableVisualisations(self):
        df = self.AvailableVisualisations
        if not self.hasModels:
            df = df.query("modelData == 0")
        if not self.hasPredictorBinning:
            df = df.query("predictorData == 0")
        if not self.hasMultipleSnapshots:
            df = df.query("`Multiple snapshots` == 0")
        return list(df.index)

    def plotApplicable(self):
        allplots = []
        for plot in self.ApplicableVisualisations:
            allplots.append(eval(str("self." + plot))())
        return allplots

    @staticmethod
    def top_n(df, top_n, to_plot='PerformanceBin'):
        if top_n > 0:
            topn = (
                df.sort_values(to_plot, ascending=False)
                .groupby("PredictorName")
                .mean()
                .nlargest(top_n, to_plot)
                .index.tolist()
            )
            df = df.query(f"PredictorName == {topn}").reset_index(drop=True)
        return df

    def _subset_data(
        self,
        table: str,
        required_columns: set,
        query: Union[str, Dict[str, list]] = None,
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
            By default, the context keys are added as required columns.
        query : Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        multi_snapshot : bool, default = None
            Whether to make sure there are multiple snapshots present in the data
            Sometimes required for visualisations over time
        last : bool, default = False
            Whether to subset on just the last known value for each model ID/predictor/bin
        active_only : bool, default = False
            Whether to subset on just the active predictors

        Returns
        -------
        pd.DataFrame
            The subsetted dataframe
        """
        if not hasattr(self, table) or getattr(self, table) is None:
            raise self.NotApplicableError(
                f"This visualisation requires {table}, but that table isn't in this dataset."
            )

        df = getattr(self, table)
        if table != "predictorData":
            required_columns = required_columns.union(self.context_keys)

        assert required_columns.issubset(
            df.columns
        ), f"The following columns are missing in the data: {required_columns - set(df.columns)}"

        df = self._apply_query(df, query)

        if multi_snapshot and not last:
            if not df["SnapshotTime"].nunique() > 1:
                raise self.NotApplicableError(
                    "There is only one snapshot, so this visualisation doesn't make sense."
                )

        if last:
            df = self.last(df)

        if active_only and "PredictorName" in df.columns:
            df = self._apply_query(df, "EntryType == 'Active'")

        return df[list(required_columns)]

    def plotPerformanceSuccessRateBubbleChart(
        self,
        last: bool = True,
        add_bottom_left_text: bool = True,
        query: Union[str, dict] = None,
        facets: Optional[list] = None,
        **kwargs,
    ) -> Union[plt.Axes, go.FigureWidget]:
        """Creates bubble chart similar to ADM OOTB.

        Parameters
        ----------
        last: bool, default = True
            Whether to only look at the last snapshot (recommended)
        add_bottom_left_text: bool, default = True
            Whether to display how many models are in the bottom left of the chart
            In other words, who have no performance and no success rate
        query: Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        facets: list, default = None
            Whether to add facets to the plot, should be a list of columns

        Keyword arguments
        -----------------
        round : int, default = 5
            To how many digits to round the hover data
        plotting_engine: str
            One of {plotly, mpl}, see ADMDatamart class
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for either the matplotlib plots (plots_mpl.py) or the
        plotly plots (plots_plotly.py). Some visualisations have parameters
        that differ slightly between the two, and plotly has an additional
        post_plot function defining some more actions, such as writing to
        html automatically or displaying figures while facetting.

        Returns
        -------
        (plt.Axes, go.FigureWidget)
        """
        plotting_engine = self.get_engine(
            kwargs.pop("plotting_engine", self.plotting_engine)
        )()

        table = "modelData"
        required_columns = {
            "ModelID",
            "Performance",
            "SuccessRate",
            "ResponseCount",
            "ModelName",
        }

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
            facets=facets,
            context_keys=self.context_keys,
            query=query,
            **kwargs,
        )

    def plotPerformanceAndSuccessRateOverTime(
        self, day_interval: int = 7, query: Union[str, dict] = None, **kwargs
    ) -> plt.Axes:
        """Plots both performance and success rate over time

        Currently only supported for matplotlib.

        Parameters
        ----------
        day_interval: int, default = 7
            The interval of tick labels along the x axis
        query: Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe

        Keyword arguments
        -----------------
        figsize: tuple
            The size of the graph
        plotting_engine: str
            This chart is only supported in matplotlib (mpl)
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        ----------------------------
        See the docs for the matplotlib plots (plots_mpl.py).

        Returns
        -------
        plt.Axes
        """
        if kwargs.get("plotting_engine", self.plotting_engine) != "mpl":
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
        }
        df = self._subset_data(
            table, required_columns, query, multi_snapshot=multi_snapshot
        )
        if kwargs.pop("return_df", False):
            return df

        return mpl().PerformanceAndSuccessRateOverTime(
            df, day_interval=day_interval, query=query, **kwargs
        )

    def plotOverTime(
        self,
        metric: str = "Performance",
        by: str = "ModelID",
        query: Union[str, dict] = None,
        facets: Optional[list] = None,
        **kwargs,
    ) -> Union[plt.Axes, go.FigureWidget]:
        """Plots a given metric over time

        Parameters
        ----------
        metric: str, default = Performance
            The metric to plot over time. One of the following:
            {ResponseCount, Performance, SuccessRate, Positives}
        by: str, default = ModelID
            What variable to group the data by
            One of {ModelID, ModelName}
        query: Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        facets: list, default = None
            Whether to add facets to the plot, should be a list of columns

        Keyword arguments
        -----------------
        plotting_engine: str
            One of {plotly, mpl}, see ADMDatamart class
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for either the matplotlib plots (plots_mpl.py) or the
        plotly plots (plots_plotly.py). Some visualisations have parameters
        that differ slightly between the two, and plotly has an additional
        post_plot function defining some more actions, such as writing to
        html automatically or displaying figures while facetting.

        Returns
        -------
        (plt.Axes, go.FigureWidget)
        """
        plotting_engine = self.get_engine(
            kwargs.pop("plotting_engine", self.plotting_engine)
        )()

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
        }
        df = self._subset_data(
            table, required_columns, query, multi_snapshot=multi_snapshot
        )

        if kwargs.pop("return_df", False):
            return df

        return plotting_engine.OverTime(
            df=df,
            metric=metric,
            by=by,
            query=query,
            facets=facets,
            **kwargs,
        )

    def plotResponseCountMatrix(
        self,
        lookback: int = 15,
        fill_null_days: bool = False,
        query: Union[str, dict] = None,
        **kwargs,
    ) -> plt.Axes:
        """Plots the response count over time in a matrix

        Parameters
        ----------
        lookback: int, default = 15
            How many days to look back from the last snapshot
        fill_null_days: bool, default = False
            If True, null values will be generated for days without a snapshot
        query: Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe

        Keyword arguments
        -----------------
        plotting_engine: str
            This chart is only supported in matplotlib (mpl)
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for the matplotlib plots (plots_mpl.py).

        Returns
        -------
        plt.Axes
        """
        if kwargs.get("plotting_engine", self.plotting_engine) != "mpl":
            print("Plot is only available in matplotlib.")

        table = "modelData"
        multi_snapshot = True
        required_columns = {"ModelID", "ModelName", "SnapshotTime", "ResponseCount"}
        df = self._subset_data(
            table, required_columns, query=query, multi_snapshot=multi_snapshot
        )
        assert (
            lookback <= df["SnapshotTime"].nunique()
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
        return mpl().ResponseCountMatrix(
            annot_df=annot_df, heatmap_df=heatmap_df, query=query, **kwargs
        )

    def plotPropositionSuccessRates(
        self,
        metric: str = "SuccessRate",
        by: str = "ModelName",
        show_error: bool = True,
        query: Union[str, dict] = None,
        facets: Optional[list] = None,
        **kwargs,
    ) -> Union[plt.Axes, go.FigureWidget]:
        """Plots all latest proposition success rates

        Parameters
        ----------
        metric: str, default = SuccessRate
            Can be changed to plot a different metric
        by: str, default = ModelName
            What variable to group the data by
            One of {ModelID, ModelName}
        show_error: bool, default = True
            Whether to show error bars in the bar plots
        query: Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        facets: list, default = None
            Whether to add facets to the plot, should be a list of columns

        Keyword arguments
        -----------------
        plotting_engine: str
            One of {plotly, mpl}, see ADMDatamart class
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for either the matplotlib plots (plots_mpl.py) or the
        plotly plots (plots_plotly.py). Some visualisations have parameters
        that differ slightly between the two, and plotly has an additional
        post_plot function defining some more actions, such as writing to
        html automatically or displaying figures while facetting.

        Returns
        -------
        (plt.Axes, go.FigureWidget)
        """
        plotting_engine = self.get_engine(
            kwargs.get("plotting_engine", self.plotting_engine)
        )()

        table = "modelData"
        last = True
        required_columns = {"ModelID", "ModelName", "SuccessRate"}.union({metric})
        df = self._subset_data(table, required_columns, query, last=last)

        if kwargs.pop("return_df", False):
            return df

        return plotting_engine.PropositionSuccessRates(
            df=df,
            metric=metric,
            by=by,
            show_error=show_error,
            query=query,
            facets=facets,
            **kwargs,
        )

    def plotScoreDistribution(
        self,
        by: str = "ModelID",
        show_zero_responses: bool = False,
        query: Union[str, dict] = None,
        **kwargs,
    ) -> Union[plt.Axes, go.FigureWidget]:
        """Plots the score distribution, similar to OOTB

        Parameters
        ----------
        by: str, default = ModelName
            What variable to group the data by
            One of {ModelID, ModelName}
        show_zero_responses: bool, default = False
            Whether to include bins with no responses at all
        query: Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe

        Keyword arguments
        -----------------
        plotting_engine: str
            One of {plotly, mpl}, see ADMDatamart class
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for either the matplotlib plots (plots_mpl.py) or the
        plotly plots (plots_plotly.py). Some visualisations have parameters
        that differ slightly between the two, and plotly has an additional
        post_plot function defining some more actions, such as writing to
        html automatically or displaying figures while facetting.

        Returns
        -------
        (plt.Axes, go.FigureWidget)
        """
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
        df = df.groupby(by)
        if df.ngroups > 10:  # pragma: no cover
            print(
                f"""WARNING: you are about to create {df.ngroups} plots because there are that many models. For convenience we've set the 'show_each' parameter to False - so the plots will be returned in a list. Iterate through that list to show the plots, or override 'show_each' to True if that's desired."""
            )
            show_each = kwargs.pop("show_each", False)
        else:  # pragma: no cover
            show_each = kwargs.pop("show_each", True)
        if kwargs.pop("return_df", False):
            return df

        return plotting_engine.ScoreDistribution(
            df=df,
            show_zero_responses=show_zero_responses,
            query=query,
            show_each=show_each,
            **kwargs,
        )

    def plotPredictorBinning(
        self,
        predictors: list = None,
        modelids: list = None,
        query: Union[str, dict] = None,
        **kwargs,
    ) -> Union[plt.Axes, go.FigureWidget]:
        """Plots the binning of given predictors

        Parameters
        ----------
        predictors: list, default = None
            An optional list of predictors to plot the bins for
            Useful for plotting one or more variables over multiple models
        modelids: list, default = None
            An optional list of model ids to plot the predictors for
        query: Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe

        Keyword arguments
        -----------------
        plotting_engine: str
            One of {plotly, mpl}, see ADMDatamart class
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Additional keyword arguments
        ----------------------------
        See the docs for either the matplotlib plots (plots_mpl.py) or the
        plotly plots (plots_plotly.py). Some visualisations have parameters
        that differ slightly between the two, and plotly has an additional
        post_plot function defining some more actions, such as writing to
        html automatically or displaying figures while facetting.

        Returns
        -------
        (plt.Axes, go.FigureWidget)
        """
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
        df = df.query("PredictorName != 'Classifier'")
        if modelids is not None:
            df = df.query(f"ModelID in {modelids}")
        if predictors:
            df = df.query(f"PredictorName in {predictors}")

        if df.ModelID.nunique() == 0:
            raise ValueError(
                "No model found. Please check if model ID is also in the combined data set."
            )
        num_plots = df.ModelID.nunique() * df.PredictorName.nunique()
        if num_plots > 10:  # pragma: no cover
            print(
                f"WARNING: you are about to create {num_plots} plots because there are {df.ModelID.nunique()} models and {df.PredictorName.nunique()} unique predictors.",
                "For convenience we've set the 'show_each' parameter to False - so the plots will be returned in a list.",
                "Iterate through that list to show the plots, or override 'show_each' to True if that's desired.",
            )
            show_each = kwargs.pop("show_each", False)
        else:  # pragma: no cover
            show_each = kwargs.pop("show_each", True)
        if kwargs.pop("return_df", False):
            return df
        return plotting_engine.PredictorBinning(
            df=df,
            query=query,
            show_each=show_each,
            **kwargs,
        )

    def plotPredictorPerformance(
        self,
        top_n: int = 0,
        active_only: bool = False,
        to_plot = 'Performance',
        query: Union[str, dict] = None,
        facets: Optional[list] = None,
        **kwargs,
    ) -> Union[plt.Axes, go.FigureWidget]:
        """Plots a bar chart of the performance of the predictors

        By default, this plot shows the performance over all models
        Use the querying functionality to drill down into a more specific subset

        Parameters
        ----------
        top_n: int, default = 0
            How many of the top predictors to show in the plot
        active_only: bool, default = False
            Whether to only plot active predictors
        query: Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        facets: list, default = None
            Whether to add facets to the plot, should be a list of columns

        Keyword arguments
        -----------------
        categorization: method
            Optional argument to supply your own predictor categorization method
            Useful if you want to be more specific in the legend of the plot
            Function should return a string from a string
        plotting_engine: str
            One of {plotly, mpl}, see ADMDatamart class
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Additional keyword arguments
        ----------------------------
        See the docs for either the matplotlib plots (plots_mpl.py) or the
        plotly plots (plots_plotly.py). Some visualisations have parameters
        that differ slightly between the two, and plotly has an additional
        post_plot function defining some more actions, such as writing to
        html automatically or displaying figures while facetting.

        Returns
        -------
        (plt.Axes, go.FigureWidget)
        """
        plotting_engine = self.get_engine(
            kwargs.get("plotting_engine", self.plotting_engine)
        )()
        if to_plot == 'Performance':
            var_to_plot = 'PerformanceBin'
        else: var_to_plot = to_plot
        
        table = "combinedData"
        last = True
        required_columns = {"Channel", "PredictorName", var_to_plot, "Type"}
        df = self._subset_data(
            table, required_columns, query, last=last, active_only=active_only
        )
        df = df.query("PredictorName != 'Classifier'").reset_index(drop=True)

        df = self.top_n(df, top_n, var_to_plot)
        asc = plotting_engine.__module__.split(".")[1] == "plots_mpl"
        order = (
            df.groupby("PredictorName")[var_to_plot]
            .mean()
            .fillna(0)
            .sort_values(ascending=asc)[::-1]
            .index
        )
        categorization = kwargs.pop(
            "categorization", self.defaultPredictorCategorization
        )
        df.loc[:, "Legend"] = df.PredictorName.apply(categorization)

        if kwargs.pop("return_df", False):
            return df

        return plotting_engine.PredictorPerformance(
            df=df,
            order=order,
            facets=facets,
            query=query,
            to_plot = to_plot,
            **kwargs,
        )

    def plotPredictorPerformanceHeatmap(
        self,
        top_n: int = 0,
        active_only: bool = False,
        query: Union[str, dict] = None,
        facets: list = None,
        **kwargs,
    ) -> Union[plt.Axes, go.FigureWidget]:
        """Plots heatmap of the performance of the predictors

        By default, this plot shows the performance over all models
        Use the querying functionality to drill down into a more specific subset

        Parameters
        ----------
        top_n: int, default = 0
            How many of the top predictors to show in the plot
        active_only: bool, default = False
            Whether to only plot active predictors
        query: Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        facets: list, default = None
            Whether to add facets to the plot, should be a list of columns

        Keyword arguments
        -----------------
        plotting_engine: str
            One of {plotly, mpl}, see ADMDatamart class
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for either the matplotlib plots (plots_mpl.py) or the
        plotly plots (plots_plotly.py). Some visualisations have parameters
        that differ slightly between the two, and plotly has an additional
        post_plot function defining some more actions, such as writing to
        html automatically or displaying figures while facetting.

        Returns
        -------
        (plt.Axes, go.FigureWidget)
        """
        plotting_engine = self.get_engine(
            kwargs.get("plotting_engine", self.plotting_engine)
        )()
        table = "combinedData"
        required_columns = {"PredictorName", "ModelName", "PerformanceBin"}
        df = self._subset_data(
            table, required_columns, query, active_only=active_only, last=True
        )
        df = df[df["PredictorName"] != "Classifier"].reset_index(drop=True)

        df = self.pivot_df(df)
        if top_n > 0:
            df = df.iloc[:, :top_n]

        if kwargs.pop("return_df", False):
            return df

        return plotting_engine.PredictorPerformanceHeatmap(
            df,
            facets=facets,
            query=query,
            **kwargs,
        )

    def plotImpactInfluence(
        self, ModelID: str = None, query: Union[str, dict] = None, **kwargs
    ) -> plt.Axes:
        """Plots the impact and the influence of a given model's predictors

        Parameters
        ----------
        ModelID: str, default = None
            The selected model ID
        query: Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        facets: list, default = None
            Whether to add facets to the plot, should be a list of columns

        Keyword arguments
        -----------------
        plotting_engine: str
            This chart is only supported in matplotlib (mpl)
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for the matplotlib plots (plots_mpl.py).

        Returns
        -------
        plt.Axes
        """
        if kwargs.get("plotting_engine", self.plotting_engine) != "mpl":
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
        }
        df = self._subset_data(table, required_columns, query, last=last).reset_index()
        df = (
            self._calculate_impact_influence(
                df, context_keys=self.context_keys, ModelID=ModelID
            )[["ModelID", "PredictorName", "Impact(%)", "Influence(%)"]]
            .set_index(["ModelID", "PredictorName"])
            .stack()
            .reset_index()
            .rename(columns={"level_2": "metric", 0: "value"})
        )

        if kwargs.pop("return_df", False):
            return df

        return mpl().ImpactInfluence(df=df, ModelID=ModelID, query=query, **kwargs)

    def plotResponseGain(
        self,
        by: str = "Channel",
        query: Union[str, dict] = None,
        **kwargs,
    ) -> go.FigureWidget:
        """Plots the cumulative response per model

        Parameters
        ----------
        by: str, default = Channel
            The column by which to calculate response gain
            Default is Channel, to see the response/gain chart per channel
        query: Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        facets: list, default = None
            Whether to add facets to the plot, should be a list of columns

        Keyword arguments
        -----------------
        plotting_engine: str
            This chart is only supported in plotly
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for the plotly plots (plots_plotly.py).
        Plotly has an additional post_plot function defining some more actions,
        such as writing to html automatically or displaying figures while facetting.

        Returns
        -------
        go.FigureWidget
        """
        if kwargs.get("plotting_engine", self.plotting_engine) != "plotly":
            print("Plot is only available in Plotly.")

        table = "modelData"
        last = True
        required_columns = {by, "ResponseCount", "ModelID"}
        df = self._subset_data(table, required_columns, query, last=last)
        df = self.response_gain_df(df, by=by)

        if kwargs.pop("return_df", False):
            return df

        return plotly().ResponseGain(df, by, **kwargs)

    def plotModelsByPositives(
        self,
        by: str = "Channel",
        query: Union[str, dict] = None,
        **kwargs,
    ) -> go.FigureWidget:
        """Plots the percentage of models vs the number of positive responses

        Parameters
        ----------
        by: str, default = Channel
            The column to calculate the model percentage by
        query: Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        facets: list, default = None
            Whether to add facets to the plot, should be a list of columns

        Keyword arguments
        -----------------
        plotting_engine: str
            This chart is only supported in plotly
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for the plotly plots (plots_plotly.py).
        Plotly has an additional post_plot function defining some more actions,
        such as writing to html automatically or displaying figures while facetting.

        Returns
        -------
        go.FigureWidget
        """
        if kwargs.get("plotting_engine", self.plotting_engine) != "plotly":
            print("Plot is only available in Plotly.")
        table = "modelData"
        last = True
        required_columns = {by, "Positives", "ModelID"}
        df = self._subset_data(table, required_columns, query, last=last)
        df = self.models_by_positives_df(df, by=by)
        if kwargs.pop("return_df", False):
            return df
        return plotly().ModelsByPositives(
            df,
            by="Channel",
            query=query,
            **kwargs,
        )

    def plotTreeMap(
        self,
        color_var: str = "performance_weighted",
        by: str = "ModelID",
        value_in_text: bool = True,
        midpoint: Optional[float] = None,
        query: Union[str, dict] = None,
        **kwargs,
    ) -> go.FigureWidget:
        """Plots a treemap to view performance over multiple context keys

        Parameters
        ----------
        color : str, default = performance_weighted
            The column to set as the color of the squares
            One out of:
            {responsecount, responsecount_log, positives,
            positives_log, percentage_without_responses,
            performance_weighted, successrate}
        by: str, default = Channel
            The column to use as the size of the squares
        value_in_text: bool, default = True
            Whether to print the values of the swuares in the squares
        midpoint : Optional[float]
            A parameter to assert more control over the color distribution
            Set near 0 to give lower values a 'higher' color
            Set near 1 to give higher values a 'lower' color
            Necessary for, for example, Success Rate, where rates lie very far apart
            If not supplied in such cases, there is no difference in the color
            between low values such as 0.001 and 0.1, so midpoint should be set low
        query: Union[str, dict], default = None
            The query to supply to _apply_query
            If a string, uses the default Pandas query function
            Else, a dict of lists where the key is the column name of the dataframe
            and the corresponding value is a list of values to keep in the dataframe
        facets: list, default = None
            Whether to add facets to the plot, should be a list of columns

        Keyword arguments
        -----------------
        colorscale: list
            Give a list of hex values to override the default colors
            Should consist of three colors: 'low', 'neutral' and 'high'
        plotting_engine: str
            This chart is only supported in plotly
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for the plotly plots (plots_plotly.py).
        Plotly has an additional post_plot function defining some more actions,
        such as writing to html automatically or displaying figures while facetting.

        Returns
        -------
        go.FigureWidget
        """
        if kwargs.get("plotting_engine", self.plotting_engine) != "plotly":
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
        if "issue" in df.columns and "OmniChannel" in df["Issue"].unique():
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
        else:
            color_var = color_var.lower()
        color = kwargs.pop("color_col", defaults[color_var][0])
        values = kwargs.pop("groupby_col", defaults[color_var][1])
        title = kwargs.pop("title", defaults[color_var][2])
        reverse_scale = kwargs.pop("reverse_scale", defaults[color_var][3])
        log = kwargs.pop("log", defaults[color_var][4])
        if midpoint is not None:
            midpoint = defaults[color_var][5]

        format = "%" if color_var in list(defaults.keys())[4:] else ""
        if kwargs.pop("return_df", False):
            return df
        return plotly().TreeMap(
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
            query=query,
            **kwargs,
        )
