import functools
import operator
from os import PathLike
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from .. import pega_io
from ..utils import cdh_utils


class ValueFinder:
    """Class to analyze Value Finder datasets.

    Relies heavily on polars for faster reading and transformations.
    See https://pola-rs.github.io/polars/py-polars/html/index.html

    Requires either df or a path to be supplied,
    If a path is supplied, the 'filename' argument is optional.
    If path is given and no filename is, it will look for the most recent.

    Parameters
    ----------
    path : Optional[str]
        Path to the ValueFinder data files
    df : Optional[DataFrame]
        Override to supply a dataframe instead of a file.
        Supports pandas or polars dataframes
    import_strategy: Literal['eager', 'lazy'], default = 'eager'
        Whether to import the file fully to memory, or scan the file
        When data fits into memory, 'eager' is typically more efficient
        However, when data does not fit, the lazy methods typically allow
        you to still use the data.
    verbose : bool
        Whether to print out information during importing

    Keyword arguments
    -----------------
    th: float
        An optional keyword argument to override the propensity threshold
    filename : Optional[str]
        The name, or extended filepath, towards the file
    subset : bool
        Whether to select only a subset of columns.
        Will speed up analysis and reduce unused information
    """

    def __init__(
        self,
        path: Optional[str] = None,
        df: Optional[Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]] = None,
        verbose: bool = True,
        import_strategy: Literal["eager", "lazy"] = "eager",
        **kwargs,
    ):
        if path is None and df is None:
            raise ValueError(
                "Please supply a path to the data files or the df directly."
            )

        self.import_strategy = import_strategy

        keep_cols = [
            "pyStage",
            "pyIssue",
            "pyGroup",
            "pyChannel",
            "pyDirection",
            "CustomerID",
            "pyName",
            "pyModelPropensity",
            "pyPropensity",
            "FinalPropensity",
        ]

        if df is not None:
            self.df = pega_io.readDSExport(df, verbose=verbose)
        else:
            filename = kwargs.pop("filename", "ValueFinder")
            self.df = pega_io.readDSExport(filename, path, verbose=verbose)
        if kwargs.get("subset", True):
            self.df = self.df.select(keep_cols)
        self.df = cdh_utils.set_types(self.df, "pyValueFinder")

        if "th" not in kwargs:
            self.th = self.df.filter(pl.col("pyStage") == "Eligibility").select(
                pl.quantile("pyModelPropensity", 0.05).alias("th")
            )
        else:
            self.th = pl.LazyFrame({"th": kwargs.pop("th")})

        self.ncust = self.df.select(
            (pl.lit(10) ** pl.col("CustomerID").n_unique().log10().ceil())
            .cast(pl.UInt32)
            .alias("ncust")
        )

        self.NBADStages = ["Eligibility", "Applicability", "Suitability", "Arbitration"]
        self.StageOrder = (
            pl.DataFrame(
                {"pyStage": self.NBADStages}, schema={"pyStage": pl.Categorical}
            )
            .select(pl.col("pyStage").cat.set_ordering("physical"))
            .lazy()
        )  # This pre-fills the stringcache to make the ordering of stages correct
        self.maxPropPerCustomer = self.df.groupby(["CustomerID", "pyStage"]).agg(
            pl.max("pyModelPropensity").alias("MaxModelPropensity")
        )

        if import_strategy == "eager":
            self.df = self.df.collect().lazy()
            self.th = self.th.collect().lazy()
            self.ncust = self.ncust.collect().lazy()
            self.maxPropPerCustomer = self.maxPropPerCustomer.collect().lazy()

        self.customersummary = self.getCustomerSummary(self.th)
        self.countsPerStage = self.getCountsPerStage(self.customersummary)

        self._thMap = dict()
        self.countsPerThreshold = dict()

    def save_data(self, path: str = ".") -> PathLike:
        """Cache the ValueFinder dataset to a file

        Parameters
        ----------
        path : str
            Where to place the file

        Returns
        -------
        PathLike:
            The paths to the file
        """
        from datetime import datetime

        time = cdh_utils.toPRPCDateTime(datetime.now())
        out = pega_io.cache_to_file(self.df, path, name=f"cached_ValueFinder_{time}")
        return out

    def getCustomerSummary(
        self,
        th: Optional[float] = None,
    ) -> pl.DataFrame:
        """Computes the summary of propensities for all customers

        Parameters
        ----------
        th: Optional[float]
            The threshold to consider an action 'good'.
            If a customer has actions with propensity above this,
            the customer has at least one relevant action.
            If not given, will default to 5th quantile.
        """
        if th is None:
            th = self.th
        if isinstance(th, float):
            th = pl.LazyFrame({"th": th})

        df = (
            self.df.with_context(th)
            .groupby(["CustomerID", "pyStage"])
            .agg(
                [
                    pl.max("pyPropensity").alias("MaxPropensity"),
                    pl.max("FinalPropensity").alias("MaxFinalPropensity"),
                    pl.max("pyModelPropensity").alias("MaxModelPropensity"),
                    pl.count("pyPropensity").alias("NOffers"),
                ]
            )
            .with_columns(
                [
                    (pl.col("MaxModelPropensity") >= pl.col("th")).alias(
                        "relevantActions"
                    ),
                    (pl.col("MaxModelPropensity") < pl.col("th")).alias(
                        "irrelevantActions"
                    ),
                ]
            )
        )

        if self.import_strategy == "eager":
            return df.collect().lazy()
        else:
            return df

    def getCountsPerStage(
        self,
        customersummary: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """Generates an aggregated view per stage.

        Parameters
        ----------
        customersummary : Optional[pl.DataFrame]
            Optional override of the customer summary,
            which can be generated by getCustomerSummary().
        """
        if customersummary is None:
            customersummary = self.customersummary

        df = (
            customersummary.groupby("pyStage")
            .agg(
                [
                    pl.sum("relevantActions"),
                    pl.sum("irrelevantActions"),
                    pl.count("relevantActions").alias("NoActions"),
                ]
            )
            .with_context(self.ncust)
            .with_columns((pl.col("ncust") - pl.col("NoActions")).alias("NoActions"))
        )
        if self.import_strategy == "eager":
            return df.collect().lazy()
        else:
            return df

    def getThFromQuantile(self, quantile: float) -> float:
        """Return the propensity threshold corresponding to a given quantile

        If the threshold is already in `self._thMap`, simply gets it from there
        Otherwise, computes the threshold and then adds it to the map.

        Parameters
        ----------
        quantile: float
            The quantile to get the threshold for

        """
        import functools
        import operator

        if quantile not in functools.reduce(operator.iconcat, self._thMap.values(), []):
            th = (
                self.df.filter(pl.col("pyStage") == "Eligibility")
                .select(pl.col("pyModelPropensity").quantile(quantile))
                .collect()
                .item()
            )
            if th in self._thMap.keys():
                self._thMap[th].append(quantile)
            else:
                self._thMap[th] = [quantile]
        return [th for th, quantiles in self._thMap.items() if quantile in quantiles][0]

    def getCountsPerThreshold(self, th, return_df=False) -> Optional[pl.LazyFrame]:
        if th not in self.countsPerThreshold.keys():
            df = (
                (
                    self.maxPropPerCustomer.with_columns(
                        [
                            (pl.col("MaxModelPropensity") >= th).alias(
                                "relevantActions"
                            ),
                            (pl.col("MaxModelPropensity") < th).alias(
                                "irrelevantActions"
                            ),
                        ]
                    )
                )
                .groupby("pyStage")
                .agg(
                    [
                        pl.sum("relevantActions"),
                        pl.sum("irrelevantActions"),
                        pl.count("relevantActions").alias("NoActions"),
                    ]
                )
                .with_context(self.ncust)
                .with_columns(
                    (pl.col("ncust") - pl.col("NoActions")).alias("NoActions")
                )
            )
            if self.import_strategy == "eager":
                self.countsPerThreshold[th] = df.collect().lazy()
            else:
                self.countsPerThreshold[th] = df
        if return_df:
            return self.countsPerThreshold[th]

    def addCountsForThresholdRange(
        self, start, stop, step, method=Literal["threshold, quantile"]
    ) -> None:
        """Adds the counts per stage for a range of quantiles or thresholds.

        Once computed, the values are added to `.countsPerThreshold` so we
        only need to compute each value once.

        Parameters
        ----------
        start : float
            The starting of the range
        stop : float
            The end of the range
        step : float
            The steps to compute between start and stop
        method: Literal["threshold", "quantile"]:
            Whether to get a range of thresholds directly or compute
            the thresholds from their quantiles
        """
        import numpy as np

        to_add = np.arange(start, stop, step)
        if method == "quantile":
            to_add = list(map(self.getThFromQuantile, to_add))
        list(map(self.getCountsPerThreshold, to_add))

    def plotPropensityDistribution(self, sampledN: int = 10_000) -> go.Figure:
        """Plots the distribution of the different propensities.

        For optimization reasons (storage for all points in a boxplot and
        time complexity for computing the distribution plot),
        we have to sample to a reasonable amount of data points.

        Parameters
        ----------
        sampledN : int, default = 10_000
            The number of datapoints to sample

        """
        i = 0
        figs = make_subplots(
            rows=len(self.NBADStages), cols=1, shared_xaxes=True, vertical_spacing=0.005
        )
        for stage in self.NBADStages:
            data = self.df.filter(pl.col("pyStage") == stage)
            data = data.pdstools.sample(sampledN).drop_nulls().collect()
            temp = ff.create_distplot(
                [data["pyModelPropensity"].to_list()],
                ["pyModelPropensity"],
                show_rug=False,
                show_hist=False,
            )
            tempdf = pd.DataFrame(
                {"x": temp["data"][0]["x"], "y": temp["data"][0]["y"]}
            )
            fig = go.Scatter(
                x=tempdf["x"],
                y=tempdf["y"],
                name=stage,
                showlegend=False,
                line_color="rgba(0,0,0)",
            )
            boxy = go.Box(
                x=data["pyModelPropensity"], name=stage, y0=0, boxpoints="outliers"
            )
            figs.add_trace(fig, row=i + 1, col=1)
            figs.add_trace(boxy, row=i + 1, col=1)
            figs.update_yaxes(title_text="Density", row=i + 1, col=1)
            i += 1

        figs.update_xaxes(title_text="pyModelPropensity", row=4, col=1)
        figs.update_layout(
            title_text="Propensity Distribution<br><sup>per NBAD stage</sup>",
            legend_title_text="pyStage",
            height=800,
        )
        return figs

    def plotPropensityThreshold(self, sampledN=10000, stage="Eligibility") -> go.Figure:
        """Plots the propensity threshold vs the different propensities.

        Parameters
        ----------
        sampledN : int, default = 10_000
            The number of datapoints to sample

        """
        colors = ["#001F5F", "#10A5AC", "#F76923"]
        propensities = ["FinalPropensity", "pyPropensity", "pyModelPropensity"]

        i = 0
        figs = make_subplots(rows=len(propensities), cols=1, shared_xaxes=True)
        yrange = [0, 15]
        th = self.th.pdstools.item()
        data = (
            self.df.filter(pl.col("pyStage") == stage)
            .select(propensities)
            .pdstools.sample(sampledN)
            .collect()
        )
        for ptype in propensities:
            plotdf = data[ptype].to_list()
            temp = ff.create_distplot(
                [plotdf], ["value"], show_rug=False, show_hist=False
            )
            tempdf = pd.DataFrame(
                {"x": temp["data"][0]["x"], "y": temp["data"][0]["y"]}
            )
            figs.add_trace(
                go.Scatter(
                    x=tempdf["x"],
                    y=tempdf["y"],
                    name=ptype,
                    showlegend=False,
                    line_color=colors[i],
                    hoverinfo="skip",
                ),
                row=i + 1,
                col=1,
            )
            figs.add_trace(
                go.Histogram(
                    x=plotdf,
                    name=ptype,
                    histnorm="probability density",
                    marker_color=colors[i],
                ),
                row=i + 1,
                col=1,
            )
            figs.update_yaxes(range=yrange, col=1, row=i + 1)
            figs.add_shape(
                type="line",
                x0=th,
                x1=th,
                y0=yrange[0],
                y1=yrange[1],
                line=dict(dash="dot"),
                col=1,
                row=i + 1,
            )
            i += 1
        figs.update_xaxes(
            title_text="Propensity", ticks="outside", showticklabels=True, row=3, col=1
        )
        figs.update_yaxes(title_text="density", row=2, col=1)
        figs.update_layout(
            title_text=f"Propensity Distribution <br><sup>{stage} stage</sup>",
            title_x=0.1,
            legend_title="Type",
            height=800,
        )
        return figs

    def plotPieCharts(
        self,
        start: float = None,
        stop: float = None,
        step: float = None,
        *,
        method: Literal["threshold", "quantile"] = "threshold",
        rounding: int = 3,
        th: Optional[float] = None,
    ) -> go.FigureWidget:
        """Plots pie charts showing the distribution of customers

        The pie charts each represent the fraction of customers with
        the color indicating whether they have sufficient relevant actions
        in that stage of the NBAD arbitration.

        If no values are provided for start, stop or step, the pie charts are
        shown using the default propensity threshold, as part of the Value Finder
        class.

        Parameters
        ----------
        start : float
            The starting of the range
        stop : float
            The end of the range
        step : float
            The steps to compute between start and stop

        Keyword arguments
        -----------------
        method : Literal['threshold', 'quantile'], default='threshold'
            Whether the range is computed based on the threshold directly
            or based on the quantile of the propensity
        rounding : int
            The number of digits to round the values by
        th : Optional[float]
            Choose a specific propensity threshold to plot
        """
        default = th or self.th.pdstools.item()
        if None in (start, stop, step):
            if start is not None:
                start, stop = start, start + 1
            else:
                start, stop = default, default + 1
            step = 10000
        iter = np.arange(start, stop, step)
        fig = make_subplots(
            rows=1,
            cols=len(self.NBADStages),
            specs=[[{"type": "domain"}] * len(self.NBADStages)],
            subplot_titles=self.NBADStages,
        )

        found_default = None
        default_n = 0
        steps = []
        for n, iter_val in enumerate(iter):
            if method == "quantile":
                threshold = self.getThFromQuantile(iter_val)
            else:
                threshold = iter_val
            if threshold == default:
                visible, found_default, default_n = True, True, n
            else:
                visible = False
            df = (
                self.getCountsPerThreshold(threshold, True)
                .collect()
                .partition_by("pyStage", as_dict=True)
            )
            for i, stage in enumerate(self.NBADStages):
                plotdf = df[stage].drop("pyStage")
                fig.add_trace(
                    go.Pie(
                        values=list(plotdf.to_numpy())[0],
                        labels=list(
                            plotdf.rename(
                                {
                                    "relevantActions": "At least one relevant action",
                                    "irrelevantActions": "Only irrelevant actions",
                                    "NoActions": "Without actions",
                                }
                            ).columns
                        ),
                        name=stage,
                        visible=visible,
                        sort=False,
                    ),
                    1,
                    i + 1,
                )

            visibleStages = [False] * len(iter) * 4
            for x in range(n * 4, n * 4 + 4):
                visibleStages[x] = True
            title = "Distribution of customers per stage "
            if method == "threshold":
                title += f"at propensity threshold: {round(threshold,rounding):.1%}"
            else:
                title += f"at quantile: {round(iter_val,rounding)} (threshold: {round(threshold,rounding):.1%})"
            step = dict(
                method="update",
                label=str(round(iter_val, rounding)),
                args=[
                    {"visible": visibleStages},
                    {"title": title},
                ],
            )
            steps.append(step)
        sliders = [
            dict(
                active=default_n,
                currentvalue={"prefix": f"Propensity {method}: "},
                pad={"t": 50},
                steps=steps,
            )
        ]
        fig.update_layout(
            sliders=sliders,
            title_text=f"Distribution of customers per stage at propensity threshold {round(float(default), rounding):.1%}",
        )
        fig.update_traces(marker=dict(colors=["#219e3f", "#fca52e", "#cd001f"]))

        if not found_default:
            for i in range(0, 4):
                fig.data[i].visible = True
        for i in range(len(fig.layout.sliders[0].steps)):
            fig.layout.sliders[0].steps[
                i
            ].label = f"{float(fig.layout.sliders[0].steps[i].label):.1%}"
        return fig

    def plotDistributionPerThreshold(
        self,
        start: float = None,
        stop: float = None,
        step: float = None,
        *,
        method: Literal["threshold", "quantile"] = "threshold",
        rounding=3,
    ) -> go.FigureWidget:
        """Plots the distribution of customers per threshold, per stage.

        Based on the precomputed data in self.countsPerThreshold,
        this function will plot the distribution per stage.

        To add more data points between a given range,
        simply pass all three arguments to this function:
        start, stop and step.

        Parameters
        ----------
        start : float
            The starting of the range
        stop : float
            The end of the range
        step : float
            The steps to compute between start and stop

        Keyword arguments
        -----------------
        method : Literal['threshold', 'quantile'], default='threshold'
            Whether the range is computed based on the threshold directly
            or based on the quantile of the propensity
        rounding : int
            The number of digits to round the values by
        """

        if None not in (start, stop, step):
            self.addCountsForThresholdRange(start, stop, step, method)
        self.addCountsForThresholdRange(0.01, 1, 0.1, "quantile")
        if method == "threshold":
            df = [
                df.with_columns(threshold=pl.lit(th).round(rounding))
                for th, df in self.countsPerThreshold.items()
            ]
        else:
            df = []
            for quantile in functools.reduce(
                operator.iconcat, self._thMap.values(), []
            ):
                th = self.getThFromQuantile(quantile)
                df.append(
                    self.getCountsPerThreshold(th, True).with_columns(
                        quantile=pl.lit(quantile).round(rounding)
                    )
                )
        df = pl.concat(df).collect().to_pandas().set_index(method)
        fig = px.area(
            df.rename(
                columns={
                    "relevantActions": "At least one relevant action",
                    "irrelevantActions": "Only irrelevant actions",
                    "NoActions": "Without actions",
                }
            ),
            color_discrete_sequence=["#219e3f", "#fca52e", "#cd001f"],
            facet_col="pyStage",
            title="Distribution of offers per stage",
            labels={"value": "Number of people", method: method.title()},
            category_orders={"pyStage": self.NBADStages},
            template="none",
        )
        fig.update_layout(legend_title_text="Status")
        if method == "threshold":
            fig.update_xaxes(tickformat=",.1%")
        return fig

    def plotFunnelChart(self, level: str = "Action", query=None, return_df=False):
        """Plots the funnel of actions or issues per stage.

        Parameters
        ----------
        level : str, default = 'Actions'
            Which element to plot:
            - If 'Actions', plots the distribution of actions.
            - If 'Issues', plots the distribution of issues
        """
        if level.casefold() in {"action", "name", "pyname"}:
            level, cat = "pyName", "Actions"
        elif level.casefold() in {"issue", "pyissue"}:
            level, cat = "pyIssue", "Issues"
        elif level.casefold() in {"group", "pygroup"}:
            level, cat = "pyGroup", "Groups"

        df = self.df if query is None else self.df.filter(query)
        df = (
            df.groupby("pyStage")
            .agg(pl.col(level).cast(pl.Utf8).value_counts(sort=True))
            .explode(level)
            .unnest(level)
            .sort("pyStage")
            .rename({level: "Name", "counts": "Count", "pyStage": "Stage"})
            .collect()
        )
        if return_df:
            return df

        fig = px.funnel(
            df.with_columns(pl.col(pl.Categorical).cast(pl.Utf8)).to_pandas(),
            y="Count",
            x="Stage",
            color="Name",
            text="Name",
            title=f"Distribution of {cat.casefold()} over the stages",
            template="none",
        )
        fig.update_xaxes(categoryorder="array", categoryarray=self.NBADStages)
        fig.update_layout(legend_title_text=cat)
        if return_df == "both":
            return fig, df
        return fig
