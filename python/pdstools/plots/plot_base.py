from typing import Optional, Union, Dict, List
import pandas as pd
import polars as pl
from .plots_plotly import ADMVisualisations as plotly
from ..utils.cdh_utils import defaultPredictorCategorization, weighed_performance_polars
from ..utils.errors import NotApplicableError
import plotly.graph_objs as go


class Plots:
    def __init__(self):
        self.hasModels = self.modelData is not None
        self.hasPredictorBinning = self.predictorData is not None
        self.hasCombined = hasattr(self, "combinedData")
        if self.import_strategy == "eager":
            if self.hasModels:
                self.hasMultipleSnapshots = (
                    self.modelData.select(pl.col("SnapshotTime").n_unique() > 1)
                    .collect()
                    .item()
                )

    @property
    def AvailableVisualisations(self):
        df = pl.DataFrame(
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
        )
        df = df.transpose().with_column(pl.Series(df.columns))
        df.columns = ["modelData", "predictorData", "Multiple snapshots", "Type"]
        return df.select(["Type", "modelData", "predictorData", "Multiple snapshots"])

    @property
    def ApplicableVisualisations(self):
        if self.import_strategy != "eager":
            raise ValueError("Function only supported in eager mode.")
        df = self.AvailableVisualisations
        if not self.hasModels:
            df = df.filter(pl.col("modelData") == 0)
        if not self.hasPredictorBinning:
            df = df.filter(pl.col("predictorData") == 0)
        if not self.hasMultipleSnapshots:
            df = df.filter(pl.col("Multiple snapshots") == 0)
        return df.get_column("Type").to_list()

    def plotApplicable(self):
        allplots = []
        for plot in self.ApplicableVisualisations:
            allplots.append(eval(str("self." + plot))())
        return allplots

    @staticmethod
    def top_n(
        df,
        top_n,
        to_plot="PerformanceBin",
    ):
        if top_n > 0:
            df = df.join(
                df.filter(pl.col("PredictorName").cast(pl.Utf8) != "Classifier")
                .groupby("PredictorName")
                .agg(pl.mean(to_plot))
                .sort("PerformanceBin")
                .tail(top_n)
                .select("PredictorName"),
                on="PredictorName",
            )
        return df

    def _subset_data(
        self,
        table: str,
        required_columns: set,
        query: Union[str, Dict[str, list]] = None,
        multi_snapshot: bool = False,
        last: bool = False,
        facets=None,
        active_only: bool = False,
        include_cols: Optional[list] = None,
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
        include_cols: Optional[list]
            Extra columns to include in the subsetting

        Returns
        -------
        pd.DataFrame
            The subsetted dataframe
        """
        if not hasattr(self, table) or getattr(self, table) is None:
            raise NotApplicableError(
                f"This visualisation requires {table}, but that table isn't in this dataset."
            )

        df = getattr(self, table)
        if table != "predictorData":
            required_columns = required_columns.union(self.context_keys)

        assert required_columns.issubset(
            df.columns
        ), f"The following columns are missing in the data: {required_columns - set(df.columns)}"

        df = self._apply_query(df, query)

        # if multi_snapshot and not last:
        #     if not df["SnapshotTime"].nunique() > 1:
        #         raise self.NotApplicableError(
        #             "There is only one snapshot, so this visualisation doesn't make sense."
        #         ) #Move check to plots directly

        if last:
            df = self.last(df, "lazy")

        if active_only and "PredictorName" in df.columns:
            df = df.filter(pl.col("EntryType") == "Active")

        df, facets = self._generateFacets(df, facets)

        if include_cols is not None:
            required_columns = set(list(required_columns) + include_cols)
        if facets is not [None] and facets is not None:
            required_columns = set(list(required_columns) + facets)
        required_columns = {x for x in required_columns if x is not None}

        return (
            df.select(list(required_columns)).with_columns(
                pl.col(pl.Categorical).cast(pl.Utf8)
            ),
            facets,
        )

    def _generateFacets(self, df, facets: Union[str, list, set] = None) -> list:
        if facets is None:
            return df, [None]
        if not isinstance(facets, list):
            facets = [facets]
        for facet in facets:
            if "/" in facet:
                df = df.with_column(pl.concat_str(facet.split("/"), "/").alias(facet))
        df = df.with_column(pl.col(facet).cast(pl.Utf8).fill_null("MISSING"))

        return df, facets

    def facettedPlot(self, facets, plotFunc, partition=None, *args, **kwargs):
        print(partition)
        if len(facets) > 0 and facets[0] is not None:
            figlist = []
            if partition is None:
                for facet in facets:
                    print(facet)
                    figlist.append(plotFunc(facet=facet, *args, **kwargs))
            else:
                if partition == "by":
                    for name, groupdf in (
                        kwargs.pop("df")
                        .partition_by(kwargs.pop("by"), as_dict=True)
                        .items()
                    ):
                        figlist.append(
                            plotFunc(
                                facet=facet, name=name, df=groupdf, *args, **kwargs
                            )
                        )
                elif partition == "facet":
                    for facet, groupdf in (
                        kwargs.pop("df").partition_by(facets, as_dict=True).items()
                    ):
                        figlist.append(
                            plotFunc(facet=facet, df=groupdf, *args, **kwargs)
                        )
            return figlist if len(figlist) > 1 else figlist[0]
        else:
            return plotFunc(*args, **kwargs)

    def plotPerformanceSuccessRateBubbleChart(
        self,
        last: bool = True,
        add_bottom_left_text: bool = True,
        query: Union[str, dict] = None,
        facets: Optional[list] = None,
        **kwargs,
    ) -> go.FigureWidget:
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
            'plotly' or a custom plot class
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for the plotly plots (plots_plotly.py)
        to see further parameters for this plot.

        Returns
        -------
        go.FigureWidget
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
            "Name",
        }

        df, facets = self._subset_data(
            table=table,
            required_columns=required_columns,
            query=query,
            facets=facets,
            last=last,
        )
        df.with_columns(
            (pl.col(["Performance", "SuccessRate"]) * pl.lit(100)).round(
                kwargs.pop("round", 5)
            )
        )
        with pl.StringCache():
            df = df.collect()
        if kwargs.pop("return_df", False):
            return df

        return self.facettedPlot(
            facets,
            plotting_engine.PerformanceSuccessRateBubbleChart,
            df=df,
            add_bottom_left_text=add_bottom_left_text,
            context_keys=self.context_keys,
            query=query,
            **kwargs,
        )

    def plotOverTime(
        self,
        metric: str = "Performance",
        by: str = "ModelID",
        every: int = "1d",
        query: Union[str, dict] = None,
        facets: Optional[list] = None,
        **kwargs,
    ) -> go.FigureWidget:
        """Plots a given metric over time

        Parameters
        ----------
        metric: str, default = Performance
            The metric to plot over time. One of the following:
            {ResponseCount, Performance, SuccessRate, Positives}
        by: str, default = ModelID
            What variable to group the data by
            One of {ModelID, Name}
        every: int, default = 1d
            How often to consider the metrics
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
            'plotly' or a custom plot class
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for the plotly plots (plots_plotly.py)
        to see further parameters for this plot.

        Returns
        -------
        go.FigureWidget
        """
        plotting_engine = self.get_engine(
            kwargs.pop("plotting_engine", self.plotting_engine)
        )()

        table = "modelData"
        multi_snapshot = True
        required_columns = {
            "ModelID",
            "Name",
            "SnapshotTime",
            "ResponseCount",
            "Performance",
            "SuccessRate",
            "Positives",
        }
        df, facets = self._subset_data(
            table,
            required_columns,
            query,
            facets=facets,
            multi_snapshot=multi_snapshot,
            include_cols=[by],
        )
        df = df.sort(by="SnapshotTime")

        groupby = [by]
        if len(facets) > 0 and facets[0] is not None:
            groupby = groupby + facets
        if metric in ["Performance", "weighted_performance", "SuccessRate"]:
            df = (
                df.groupby_dynamic("SnapshotTime", every=every, by=groupby).agg(
                    [
                        (pl.sum("Positives") / pl.sum("ResponseCount")).alias(
                            "SuccessRate"
                        ),
                        weighed_performance_polars().alias("weighted_performance"),
                    ]
                )
            ).sort(["SnapshotTime", by])
        else:
            df = self._create_sign_df(
                df,
                by=groupby,
                what=metric,
                every=every,
                mask=False,
                pivot=False,
            )

        if metric == "Performance":
            metric = "weighted_performance"

        with pl.StringCache():
            df = df.collect()
        if kwargs.pop("return_df", False):
            return df

        return self.facettedPlot(
            facets,
            plotting_engine.OverTime,
            df=df,
            metric=metric,
            by=by,
            query=query,
            **kwargs,
        )

    def plotPropositionSuccessRates(
        self,
        metric: str = "SuccessRate",
        by: str = "Name",
        show_error: bool = True,
        top_n=0,
        subsetted_top_n=False,
        query: Union[str, dict] = None,
        facets: Optional[list] = None,
        **kwargs,
    ) -> go.FigureWidget:
        """Plots all latest proposition success rates

        Parameters
        ----------
        metric: str, default = SuccessRate
            Can be changed to plot a different metric
        by: str, default = Name
            What variable to group the data by
            One of {ModelID, Name}
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
            'plotly' or a custom plot class
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for the plotly plots (plots_plotly.py)
        to see further parameters for this plot.

        Returns
        -------
        go.FigureWidget
        """
        plotting_engine = self.get_engine(
            kwargs.get("plotting_engine", self.plotting_engine)
        )()

        table = "modelData"
        last = True
        required_columns = {"ModelID", "Name", "SuccessRate"}
        df, facets = self._subset_data(
            table,
            required_columns,
            query,
            facets=facets,
            last=last,
            include_cols=[metric, by],
        )
        top_n_by = by if facets == [None] else facets + [by]
        if top_n > 0:  # TODO: fix.
            df = df.join(
                df.groupby(facets)
                .agg(pl.mean(metric))
                .sort(metric)
                .tail(top_n)
                .select(facets),
                on=facets,
            )
        with pl.StringCache():
            df = df.collect()

        if kwargs.pop("return_df", False):
            return df

        return self.facettedPlot(
            facets,
            plotting_engine.PropositionSuccessRates,
            partition="facet",
            df=df,
            metric=metric,
            by=by,
            show_error=show_error,
            query=query,
            **kwargs,
        )

    def plotScoreDistribution(
        self,
        by: str = "ModelID",
        show_zero_responses: bool = False,
        modelids: Optional[List] = None,
        query: Union[str, dict] = None,
        show_each=False,
        **kwargs,
    ) -> go.FigureWidget:
        """Plots the score distribution, similar to OOTB

        Parameters
        ----------
        by: str, default = Name
            What variable to group the data by
            One of {ModelID, Name}
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
            'plotly' or a custom plot class
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for the plotly plots (plots_plotly.py)
        to see further parameters for this plot.

        Returns
        -------
        go.FigureWidget
        """
        plotting_engine = self.get_engine(
            kwargs.get("plotting_engine", self.plotting_engine)
        )()
        table = "combinedData"
        required_columns = {
            "PredictorName",
            "Name",
            "BinIndex",
            "BinSymbol",
            "BinResponseCount",
            "BinPropensity",
            "ModelID",
        }
        df, _ = self._subset_data(table, required_columns, query)
        if modelids is not None:
            df = df.filter(pl.col("ModelID").is_in(modelids))
        with pl.StringCache():
            df = df.filter(pl.col("PredictorName") == "Classifier").collect()

        if kwargs.pop("return_df", False):
            return df

        return self.facettedPlot(
            ["ModelID"],
            plotting_engine.ScoreDistribution,
            partition="facet",
            df=df,
            by=by,
            show_zero_responses=show_zero_responses,
            show_each=show_each,
            query=query,
            **kwargs,
        )

    def plotPredictorBinning(
        self,
        predictors: list = None,
        modelids: list = None,
        show_each=False,
        query: Union[str, dict] = None,
        **kwargs,
    ) -> go.FigureWidget:
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
            'plotly' or a custom plot class
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for the plotly plots (plots_plotly.py)
        to see further parameters for this plot.

        Returns
        -------
        go.FigureWidget
        """
        plotting_engine = self.get_engine(
            kwargs.get("plotting_engine", self.plotting_engine)
        )()
        table = "combinedData"
        last = True
        required_columns = {
            "PredictorName",
            "Name",
            "BinIndex",
            "BinSymbol",
            "BinResponseCount",
            "BinPropensity",
            "ModelID",
        }
        df, _ = self._subset_data(table, required_columns, query, last=last)
        df = df.filter(pl.col("PredictorName") != "Classifier")
        if modelids is not None:
            df = df.filter(pl.col("ModelID").is_in(modelids))
        if predictors is not None:
            df = df.filter(pl.col("PredictorName").is_in(predictors))
        with pl.StringCache():
            df = df.collect()

        if df["ModelID"].n_unique() == 0:
            raise ValueError(
                "No model found. Please check if model ID is also in the combined data set."
            )

        if kwargs.pop("return_df", False):
            return df

        return self.facettedPlot(
            ["ModelID", "PredictorName"],
            plotting_engine.PredictorBinning,
            partition="facet",
            df=df,
            query=query,
            show_each=show_each,
            **kwargs,
        )

    def plotPredictorPerformance(
        self,
        top_n: int = 0,
        active_only: bool = False,
        to_plot="Performance",
        query: Union[str, dict] = None,
        facets: Optional[list] = None,
        **kwargs,
    ) -> go.FigureWidget:
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
            'plotly' or a custom plot class
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for the plotly plots (plots_plotly.py)
        to see further parameters for this plot.

        Returns
        -------
        go.FigureWidget
        """
        plotting_engine = self.get_engine(
            kwargs.get("plotting_engine", self.plotting_engine)
        )()
        if to_plot == "Performance":
            to_plot = "PerformanceBin"

        table = "combinedData"
        last = True
        required_columns = {"Channel", "PredictorName", to_plot, "Type"}

        df, facets = self._subset_data(
            table,
            required_columns,
            query,
            last=last,
            facets=facets,
            active_only=active_only,
        )
        df = df.filter(pl.col("PredictorName") != "Classifier")

        df = self.top_n(df, top_n, to_plot)  # TODO: add groupby

        categorization = kwargs.pop("categorization", defaultPredictorCategorization)
        with pl.StringCache():
            df = df.collect().with_column(
                pl.col("PredictorName").apply(categorization).alias("Legend")
            )

        order = (
            df.groupby("PredictorName")
            .agg(pl.mean(to_plot))
            .fill_nan(0)
            .sort(to_plot, reverse=False)
            .get_column("PredictorName")
            .to_list()
        )

        if kwargs.pop("separate", False):
            partition = "facet"
        else:
            partition = None

        if kwargs.pop("return_df", False):
            return df, order

        return self.facettedPlot(
            facets,
            plotting_engine.PredictorPerformance,
            partition=partition,
            df=df,
            order=order,
            query=query,
            to_plot=to_plot,
            **kwargs,
        )

    def plotPredictorPerformanceHeatmap(
        self,
        top_n: int = 0,
        by="Name",
        active_only: bool = False,
        query: Union[str, dict] = None,
        facets: list = None,
        **kwargs,
    ) -> go.FigureWidget:
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
            'plotly' or a custom plot class
        return_df : bool
            If set to True, returns the dataframe instead of the plot
            Can be useful for debugging or replicating the plots

        Notes
        -----
        See the docs for the plotly plots (plots_plotly.py)
        to see further parameters for this plot.

        Returns
        -------
        go.FigureWidget
        """
        plotting_engine = self.get_engine(
            kwargs.get("plotting_engine", self.plotting_engine)
        )()
        table = "combinedData"
        required_columns = {"PredictorName", "Name", "PerformanceBin"}
        if by is not None:
            required_columns = required_columns.union({by, "ResponseCount"})
        df, facets = self._subset_data(
            table,
            required_columns,
            query,
            active_only=active_only,
            facets=facets,
            last=True,
        )
        df = df.filter(pl.col("PredictorName") != "Classifier")

        # TODO: implement facets.
        df = self.pivot_df(df, by=by)
        if top_n > 0:
            df = df[0:top_n]

        if kwargs.pop("return_df", False):
            return df

        return self.facettedPlot(
            facets,
            plotting_engine.PredictorPerformanceHeatmap,
            partition="facet",
            df=df,
            query=query,
            **kwargs,
        )

    def plotResponseGain(
        self,
        by: str = "Channel",
        query: Union[str, dict] = None,
        facets=None,
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
        df, facets = self._subset_data(
            table, required_columns, query, facets=facets, last=last
        )
        with pl.StringCache():
            df = self.response_gain_df(df, by=by).collect()

        if kwargs.pop("return_df", False):
            return df

        return self.facettedPlot(facets, plotly().ResponseGain, df=df, by=by, **kwargs)

    def plotModelsByPositives(
        self,
        by: str = "Channel",
        query: Union[str, dict] = None,
        facets=None,
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
        df, facets = self._subset_data(
            table, required_columns, query, facets=facets, last=last
        )
        with pl.StringCache():
            df = self.models_by_positives_df(df, by=by).collect()
        if kwargs.pop("return_df", False):
            return df

        return self.facettedPlot(
            facets,
            plotly().ModelsByPositives,
            df=df,
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

        mapping = {
            f"{by}_count": "Model count",
            "Percentage_without_responses": "Percentage without responses",
            "ResponseCount_sum": "Response Count sum",
            "SuccessRate_mean": "Success Rate mean",
            "Performance_weighted": "Performance weighted mean",
            "Positives_sum": "Positives sum",
        }
        with pl.StringCache():
            df = (
                self.model_summary(by=by, query=query)
                .select(
                    [
                        pl.col(self.context_keys).cast(pl.Utf8),
                        pl.col(list(mapping.keys())),
                    ]
                )
                .rename(mapping)
                .sort(self.context_keys)
                .fill_null("Missing")
                .collect()
            )

        if "Issue" in df.columns and "OmniChannel" in df["Issue"].unique():
            print(
                "WARNING: This plot does not work for OmniChannel models. For that reason, we filter those out by default."
            )
            df = df.filter(pl.col("Issue") != "OmniChannel")

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
