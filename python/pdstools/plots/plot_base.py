from typing import Optional, Union, Dict, List, Any
import polars as pl
from .plots_plotly import ADMVisualisations as plotly
from ..utils.cdh_utils import (
    weighed_performance_polars,
    weighed_average_polars,
)
from ..utils.errors import NotApplicableError
from ..utils.types import any_frame
import plotly.graph_objs as go


class Plots:
    """
    Base plotting class

    Attributes
    ----------
    hasModels : bool
        A flag indicating whether the object has model data.
    hasPredictorBinning : bool
        A flag indicating whether the object has predictor data.
    hasCombined : bool
        A flag indicating whether the object has combined data.
    AvailableVisualisations : pl.DataFrame
        A dataframe with available visualizations and whether they require model data, predictor data, or multiple snapshots.
    import_strategy : str
        Whether to import the file fully to memory, or scan the file
        When data fits into memory, 'eager' is typically more efficient
        However, when data does not fit, the lazy methods typically allow
        you to still use the data.
    """

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
            }
        )
        df = df.transpose().with_columns(pl.Series(df.columns))
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
        df: pl.DataFrame,
        top_n: int,
        to_plot: str = "PerformanceBin",
        facets: Optional[list] = None,
    ):
        """Subsets DataFrame to contain only top_n predictors.

        Parameters
        ----------
        df : pl.DataFrame
            Table to subset
        top_n : int
            Number of top predictors
        to_plot: str
            Metric to use for comparing predictors
        facets : list
            Subsets top_n predictors over facets. Seperate top predictors for each facet

        Returns
        -------
        pl.DataFrame
            Subsetted dataframe
        """

        if top_n < 1:
            return df

        if facets:
            df = df.join(
                df.groupby(*facets)
                .agg(
                    pl.col("PredictorName")
                    .sort_by("PerformanceBin", "PredictorName", descending=True)
                    .head(20)
                )
                .explode("PredictorName"),
                on=(*facets, "PredictorName"),
            )

        else:
            df = df.join(
                df.filter(pl.col("PredictorName").cast(pl.Utf8) != "Classifier")
                .groupby("PredictorName")
                .agg(pl.median(to_plot))
                .sort(to_plot)
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
        facets: Union[str, list] = None,
        active_only: bool = False,
        include_cols: Optional[list] = None,
    ) -> Union[pl.DataFrame, List[str]]:
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
        query : Union[pl.Expr, str, Dict[str, list]], default = None
            Please refer to :meth:`pdstools.adm.ADMDatamart._apply_query`
        last : bool, default = False
            Whether to subset on just the last known value for each model ID/predictor/bin
        facets : Union[str, list], deafult = None
            Please refer to :meth:`._generateFacets`
        active_only : bool, default = False
            Whether to subset on just the active predictors
        include_cols: Optional[list]
            Extra columns to include in the subsetting

        Returns
        -------
        Union[pl.DataFrame, List[str]]
            The subsetted dataframe
            Generated facet column name
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

        if last:
            df = self.last(df, "lazy")

        if active_only and "PredictorName" in df.columns:
            df = df.filter(pl.col("EntryType") == "Active")

        df, facets = self._generateFacets(df, facets)

        if include_cols is not None:
            required_columns = set(list(required_columns) + include_cols)
            df, include_cols = self._generateFacets(df, include_cols)

        if facets is not [None] and facets is not None:
            required_columns = set(list(required_columns) + facets)
        required_columns = {x for x in required_columns if x is not None}

        return (
            df.select(list(required_columns)).with_columns(
                pl.col(pl.Categorical).cast(pl.Utf8),
                pl.col(pl.Datetime).dt.replace_time_zone(None),
            ),
            facets,
        )

    def _generateFacets(
        self, df: any_frame, facets: Union[str, List[str]] = None
    ) -> list:
        """Generates a list of facets based on the given dataframe and facet columns.

        Given a string with column names combined with backslash, the function generates that column,
        adds it to the dataframe and return the new dataframe together with the generated column's name

        Parameters
        ----------
        df : pl.DataFrame | pl.LazyFrame
            The input dataframe for which the facets are to be generated.
        facets: Union[str, list], default = None
            By which columns to facet the plots.
            If string, facets it by just that one column.
            If list, facets it by every element of the list.
            If a string contains a `/`, it will combine those columns as one facet.

        Returns
        -------
        DataFrame
            The input dataframe with additional facet columns added.
        Union[str, list], deafult = None
            The generated facets

        Examples
        --------
        >>> df, facets = _generateFacets(df, "Configuration")
            Creates a plot for each Configuration
        >>> df, facets = _generateFacets(df, ["Channel", "Direction"])
            Creates a plot for each Channel and for each Direction as separate facets
        >>> df, facets = _generateFacets(df, "Channel/Configuration")
            Creates a plot for each combination of Channel and Configuration
        """
        if facets is None:
            return df, [None]
        if not isinstance(facets, list):
            facets = [facets]
        for facet in facets:
            if "/" in facet:
                df = df.with_columns(
                    pl.concat_str(
                        pl.col(facet.split("/")).cast(pl.Utf8).fill_null("NA"),
                        separator="/",
                    ).alias(facet)
                )
        return df, facets

    @staticmethod
    def facettedPlot(
        facets: Optional[list], plotFunc: Any, partition: bool = False, *args, **kwargs
    ):
        """Takes care of facetting the plots.

        If `partition` is True, generates a new dataframe for each plot
        If `partition` is False, simply gives the facet as the facet argument

        In effect, this means that `facet = False` give a 'plotly-native' facet,
        while `facet = True` gives a distinct plot for every facet.

        Parameters
        ----------
        facets : Optional[list]
            If there's no facet supplied, we just return the plot
            Else, we loop through each facet and create the plot
        plotFunc : Any
            The original function to create the plot
            The plot is simply passed through to this function
            Along with all arguments
        partition: bool, default=False
            If True, generates a new dataframe for each plot
            If False, simply gives the facet as the facet argument
        *args:
            Any additional arguments, depending on the plotFunc

        Keyword arguments
        -----------------
        order: dict
            The order of categories, for each facet
        **kwargs:
            Any additional keyword arguments, depending on the plotFunc

        """
        if kwargs.pop("verbose", False):
            print(partition)
        if len(facets) > 0 and facets[0] is not None:
            figlist = []
            if not partition:
                for facet in facets:
                    figlist.append(plotFunc(facet=facet, *args, **kwargs))
            else:
                order = kwargs.pop("order", None)
                for facet_val, groupdf in kwargs.pop("df").groupby(*facets):
                    figlist.append(
                        plotFunc(
                            df=groupdf,
                            facet=None,
                            facet_val=facet_val,
                            order=order[facet_val] if order is not None else None,
                            *args,
                            **kwargs,
                        )
                    )
            return figlist if len(figlist) > 1 else figlist[0]
        else:
            return plotFunc(*args, **kwargs)

    def plotPerformanceSuccessRateBubbleChart(
        self,
        last: bool = True,
        add_bottom_left_text: bool = True,
        query: Optional[Union[pl.Expr, str, Dict[str, list]]] = None,
        facets: Union[str, list] = None,
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
        query: Optional[Union[pl.Expr, str, Dict[str, list]]]
            Please refer to :meth:`pdstools.adm.ADMDatamart._apply_query`
        facets : Union[str, list], deafult = None
            Please refer to :meth:`._generateFacets`

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
        df = df.with_columns(
            (pl.col(["Performance", "SuccessRate"]) * pl.lit(100)).round(
                kwargs.pop("round", 5)
            )
        )
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
        query: Optional[Union[pl.Expr, str, Dict[str, list]]] = None,
        facets: Union[str, list] = None,
        **kwargs,
    ) -> go.FigureWidget:
        """Plots a given metric over time

        Parameters
        ----------
        metric: str, default = Performance
            The metric to plot over time. One of the following:
            {ResponseCount, Performance, SuccessRate, Positives, weighted_performance}
        by: str, default = ModelID
            What variable to group the data by
            One of {ModelID, Name}
        every: int, default = 1d
            How often to consider the metrics
        query: Optional[Union[pl.Expr, str, Dict[str, list]]]
            Please refer to :meth:`pdstools.adm.ADMDatamart._apply_query`
        facets : Union[str, list], deafult = None
            Please refer to :meth:`._generateFacets`

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
                df.groupby_dynamic("SnapshotTime", every=every, by=groupby)
                .agg(
                    [
                        weighed_average_polars("SuccessRate", "ResponseCount").alias(
                            "SuccessRate"
                        ),
                        weighed_performance_polars().alias("weighted_performance"),
                    ]
                )
                .with_columns(pl.col("weighted_performance") * 100)
            ).sort(["SnapshotTime", by])
        else:
            df = self._create_sign_df(
                df, by=groupby, what=metric, every=every, mask=False, pivot=False
            )
        if metric == "Performance":
            metric = "weighted_performance"

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
        query: Optional[Union[pl.Expr, str, Dict[str, list]]] = None,
        facets: Union[str, list] = None,
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
        top_n: int, default = 0
            The number of rows to include in the pivoted DataFrame. If set to 0, all rows are included.
        query: Optional[Union[pl.Expr, str, Dict[str, list]]]
            Please refer to :meth:`pdstools.adm.ADMDatamart._apply_query`
        facets : Union[str, list], deafult = None
            Please refer to :meth:`._generateFacets`

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
        df = df.collect()

        if kwargs.pop("return_df", False):
            return df

        return self.facettedPlot(
            facets,
            plotting_engine.PropositionSuccessRates,
            partition=True,
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
        *,
        show_zero_responses: bool = False,
        modelids: Optional[List] = None,
        query: Optional[Union[pl.Expr, str, Dict[str, list]]] = None,
        show_each=False,
        **kwargs,
    ) -> go.FigureWidget:
        """Plots the score distribution, similar to OOTB

        Parameters
        ----------
        by: str, default = Name
            What variable to group the data by
            One of {ModelID, Name}

        Keyword arguments
        -----------------
        show_zero_responses: bool, default = False
            Whether to include bins with no responses at all
        modelids: Optional[List], default = None
            Models to plot for. If multiple ids are given,
            returns a list of Plots for each model
        query: Optional[Union[pl.Expr, str, Dict[str, list]]]
            Please refer to :meth:`pdstools.adm.ADMDatamart._apply_query`
        show_each : bool
            Whether to show each file when multiple facets are used

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
        df = df.filter(pl.col("PredictorName") == "Classifier").collect()

        if kwargs.pop("return_df", False):
            return df

        return self.facettedPlot(
            ["ModelID"],
            plotting_engine.ScoreDistribution,
            partition=True,
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
        query: Optional[Union[pl.Expr, str, Dict[str, list]]] = None,
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
        query: Optional[Union[pl.Expr, str, Dict[str, list]]]
            Please refer to :meth:`pdstools.adm.ADMDatamart._apply_query`

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
            partition=True,
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
        query: Optional[Union[pl.Expr, str, Dict[str, list]]] = None,
        facets: Union[str, list] = None,
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
        to_plot: str, default = Performance
            Metric to compare predictors
        query: Optional[Union[pl.Expr, str, Dict[str, list]]]
            Please refer to :meth:`pdstools.adm.ADMDatamart._apply_query`
        facets : Union[str, list], deafult = None
            Please refer to :meth:`._generateFacets`
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
        if to_plot == "Performance":
            to_plot = "PerformanceBin"

        table = "combinedData"
        last = True
        required_columns = {
            "Channel",
            "PredictorName",
            "ModelID",
            "Name",
            "ResponseCountBin",
            to_plot,
            "Type",
            "PredictorCategory",
        }

        df, facets = self._subset_data(
            table,
            required_columns,
            query,
            last=last,
            facets=facets,
            active_only=active_only,
        )

        df = (
            df.filter(pl.col("PredictorName") != "Classifier")
            .groupby("ModelID", "PredictorName")
            .agg(pl.all().first())
            .with_columns(
                pl.col("PredictorName"), pl.col("PredictorCategory").alias("Legend")
            )
        )

        separate = kwargs.pop("separate", False)
        df = df.collect()

        if separate:
            partition = "facet"
            df = self.top_n(df, top_n, to_plot, facets=facets)

            order = {}
            for facet, group_df in (
                df.groupby(*facets, "PredictorName")
                .agg(pl.median(to_plot))
                .sort(to_plot)
                .partition_by(*facets, as_dict=True)
                .items()
            ):
                order[facet] = group_df.get_column("PredictorName").to_list()

        else:
            partition = None
            df = self.top_n(df, top_n, to_plot)  # TODO: add groupby
            order = (
                df.groupby("PredictorName")
                .agg(pl.median(to_plot))
                .fill_nan(0)
                .sort(to_plot, descending=False)
                .get_column("PredictorName")
                .to_list()
            )
        if to_plot == "PerformanceBin":
            df = df.with_columns(pl.col(to_plot) * 100)
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

    def plotPredictorCategoryPerformance(
        self,
        active_only: bool = False,
        to_plot="Performance",
        query: Optional[Union[pl.Expr, str, Dict[str, list]]] = None,
        facets: Union[str, list] = None,
        **kwargs,
    ) -> go.FigureWidget:
        """Plots a bar chart of the performance of the predictor categories

        By default, this plot shows the performance over all models
        Use the querying functionality to drill down into a more specific subset

        Parameters
        ----------
        active_only: bool, default = False
            Whether to only plot active predictors
        to_plot: str, default = Performance
            Metric to compare predictor categories
        query: Optional[Union[pl.Expr, str, Dict[str, list]]]
            Please refer to :meth:`pdstools.adm.ADMDatamart._apply_query`
        facets : Union[str, list], deafult = None
            Please refer to :meth:`._generateFacets`

        Keyword arguments
        -----------------
        plotting_engine: str
            'plotly' or a custom plot class
        separate: bool
            If set to true, dataset is subsetted using the facet column, creating seperate
            plots
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
        required_columns = {
            "Configuration",
            "Channel",
            "Direction",
            "PredictorName",
            "Name",
            "BinResponseCount",
            to_plot,
            "Type",
            "PredictorCategory",
        }

        df, facets = self._subset_data(
            table,
            required_columns,
            query,
            last=last,
            facets=facets,
            active_only=active_only,
        )
        df = df.filter(pl.col("PredictorName").cast(pl.Utf8) != "Classifier")

        df = df.collect().with_columns(
            pl.col("PredictorName"),
            pl.col("PredictorCategory").alias("Predictor Category"),
        )

        df = (
            df.groupby(facets + ["Name", "Predictor Category"])
            .agg(
                weighed_average_polars("PerformanceBin", "BinResponseCount").alias(
                    "PerformanceBin"
                )
            )
            .with_columns(
                [
                    (pl.col("Predictor Category").alias("Legend")),
                    (pl.col("PerformanceBin") * 100),
                ]
            )
        )

        if kwargs.pop("separate", False):
            partition = "facet"
        else:
            partition = None

        if kwargs.pop("return_df", False):
            return df

        return self.facettedPlot(
            facets,
            plotting_engine.PredictorPerformance,
            partition=partition,
            df=df,
            y="Predictor Category",
            query=query,
            to_plot=to_plot,
            **kwargs,
        )

    def plotPredictorContribution(
        self,
        by: str = "Configuration",
        query: Optional[Union[pl.Expr, str, Dict[str, list]]] = None,
        **kwargs,
    ) -> go.FigureWidget:
        """Plots the contribution of each predictor across a group

        Parameters
        ----------
        by: str, default = Configuration
            The column to group the bars with
        query: Optional[Union[pl.Expr, str, Dict[str, list]]]
            Please refer to :meth:`pdstools.adm.ADMDatamart._apply_query`

        Keyword arguments
        -----------------
        predictorCategorization : pl.Expr
            An optional override for the predictor categorization function
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

        plotting_engine = self.get_engine(
            kwargs.get("plotting_engine", self.plotting_engine)
        )()

        table = "combinedData"
        last = True
        required_columns = {
            "PredictorName",
            "PerformanceBin",
            "BinResponseCount",
            "PredictorCategory",
            by,
        }

        df, facets = self._subset_data(
            table,
            required_columns,
            query,
            last=last,
            facets=None,
        )

        if "predictorCategorization" in kwargs:
            df = df.with_columns(
                PredictorCategory=kwargs.get("predictorCategorization")
            )

        df = (
            df.filter(pl.col("PredictorName") != "Classifier")
            .with_columns((pl.col("PerformanceBin") - 0.5) * 2)
            .groupby(by, "PredictorCategory")
            .agg(
                Performance=weighed_average_polars("PerformanceBin", "BinResponseCount")
            )
            .with_columns(
                Contribution=(
                    (pl.col("Performance") / (pl.sum("Performance").over(by))) * 100
                )
            )
            .collect()
        )
        if kwargs.pop("return_df", False):
            return df

        return self.facettedPlot(
            facets,
            plotting_engine.PredictorContribution,
            df=df,
            by=by,
            **kwargs,
        )

    def plotPredictorPerformanceHeatmap(
        self,
        top_n: int = 0,
        by="Name",
        active_only: bool = False,
        query: Optional[Union[pl.Expr, str, Dict[str, list]]] = None,
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
        by: str, default = Name
            The column to use at the x axis of the heatmap
        active_only: bool, default = False
            Whether to only plot active predictors
        query: Optional[Union[pl.Expr, str, Dict[str, list]]]
            Please refer to :meth:`pdstools.adm.ADMDatamart._apply_query`
        facets : Union[str, list], deafult = None
            Please refer to :meth:`._generateFacets`

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
            required_columns = required_columns.union(
                set([col for col in by.split("/")] + ["ResponseCount"])
            )
        df, facets = self._subset_data(
            table,
            required_columns,
            query,
            active_only=active_only,
            facets=facets,
            last=True,
        )

        df, facet_col = self._generateFacets(df, by)
        df = self.pivot_df(df, by=facet_col, top_n=top_n)
        df = df.with_columns(pl.all().exclude(facet_col) * 100)

        if kwargs.pop("return_df", False):
            return df

        if kwargs.get("separate", False):
            partition = "facet"
        else:
            partition = None

        return self.facettedPlot(
            facets,
            plotting_engine.PredictorPerformanceHeatmap,
            partition=partition,
            df=df,
            query=query,
            by=by,
            **kwargs,
        )

    def plotResponseGain(
        self,
        by: str = "Channel",
        query: Optional[Union[pl.Expr, str, Dict[str, list]]] = None,
        facets=None,
        **kwargs,
    ) -> go.FigureWidget:
        """Plots the cumulative response per model

        Parameters
        ----------
        by: str, default = Channel
            The column by which to calculate response gain
            Default is Channel, to see the response/gain chart per channel
        query: Optional[Union[pl.Expr, str, Dict[str, list]]]
            Please refer to :meth:`pdstools.adm.ADMDatamart._apply_query`
        facets : Union[str, list], deafult = None
            Please refer to :meth:`._generateFacets`

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
        required_columns = {"ResponseCount", "ModelID"}

        if by is not None:
            required_columns = required_columns.union(
                set([col for col in by.split("/")] + ["ResponseCount"])
            )
        df, facets = self._subset_data(
            table, required_columns, query, facets=facets, last=last
        )
        df, by = self._generateFacets(df, by)
        df = (
            self.response_gain_df(df, by=by)
            .sort(by + ["TotalModelsFraction"])
            .collect()
        )

        if kwargs.pop("return_df", False):
            return df

        return self.facettedPlot(facets, plotly().ResponseGain, df=df, by=by, **kwargs)

    def plotModelsByPositives(
        self,
        by: str = "Channel",
        query: Optional[Union[pl.Expr, str, Dict[str, list]]] = None,
        facets=None,
        **kwargs,
    ) -> go.FigureWidget:
        """Plots the percentage of models vs the number of positive responses

        Parameters
        ----------
        by: str, default = Channel
            The column to calculate the model percentage by
        query: Optional[Union[pl.Expr, str, Dict[str, list]]]
            Please refer to :meth:`pdstools.adm.ADMDatamart._apply_query`
        facets : Union[str, list], deafult = None
            Please refer to :meth:`._generateFacets`

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
        query: Optional[Union[pl.Expr, str, Dict[str, list]]] = None,
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
        query: Optional[Union[pl.Expr, str, Dict[str, list]]]
            Please refer to :meth:`pdstools.adm.ADMDatamart._apply_query`
        facets : Union[str, list], deafult = None
            Please refer to :meth:`._generateFacets`

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

        levels = kwargs.pop("levels", self.context_keys)
        mapping = {
            f"{by}_count": "Model count",
            "Percentage_without_responses": "Percentage without responses",
            "ResponseCount_sum": "Response Count sum",
            "SuccessRate_mean": "(%) Success Rate mean",
            "Performance_weighted": "Performance weighted mean",
            "Positives_sum": "Positives sum",
        }
        df = (
            self.model_summary(by=by, query=query, context_keys=levels)
            .select(pl.col(levels).cast(pl.Utf8), pl.col(list(mapping.keys())))
            .rename(mapping)
            .sort(levels)
            .fill_null("Missing")
            .with_columns(pl.col("Performance weighted mean") * 100)
            .with_columns(pl.col("(%) Success Rate mean") * 100)
            .fill_nan(pl.lit(50))
            .fill_nan(0)
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
                "(%) Success Rate mean",
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
            context_keys=levels,
            value_in_text=value_in_text,
            query=query,
            **kwargs,
        )

    def plotPredictorCount(
        self,
        facets: Union[str, list],
        query: Optional[Union[pl.Expr, str, Dict[str, list]]] = None,
        by: str = "Type",
        **kwargs,
    ):
        plotting_engine = self.get_engine(
            kwargs.get("plotting_engine", self.plotting_engine)
        )()

        required_columns = {"Name", "EntryType", "PredictorName"}
        table = "combinedData"
        last = True
        df, facets = self._subset_data(
            table, required_columns, query, facets=facets, last=last, include_cols=[by]
        )

        df = (
            df.filter(pl.col("PredictorName") != "Classifier")
            .groupby(pl.all().exclude("PredictorName"))
            .agg(pl.n_unique("PredictorName").alias("Predictor Count"))
        )

        overall = (
            df.groupby(pl.all().exclude(["PredictorName", "Type", "Predictor Count"]))
            .agg(pl.sum("Predictor Count"))
            .with_columns(pl.lit("Overall").alias("Type"))
        )

        df = (
            pl.concat([df, overall.select(df.columns)])
            .with_columns(pl.col("Predictor Count").cast(pl.Int64))
            .sort(["EntryType", "Type"])
            .collect()
        )
        if kwargs.pop("return_df", False):
            return df

        return self.facettedPlot(
            facets, plotting_engine.PredictorCount, df=df, **kwargs
        )
