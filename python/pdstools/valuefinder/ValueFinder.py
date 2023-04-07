from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Union, Literal
from .. import pega_io
from ..utils import cdh_utils
from os import PathLike


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

        keep_cols = [
            "pyStage",
            "pyIssue",
            "pyGroup",
            "pyChannel",
            "pyDirection",
            "CustomerID",
            "pyName",
            "pyWorkID",
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
        self.maxPropPerCustomer = self.df.groupby(["CustomerID", "pyStage"]).agg(
            pl.max("pyModelPropensity").alias("MaxModelPropensity")
        )
        if import_strategy == "eager":
            self.df = self.df.collect().lazy()
            self.th = self.th.collect().lazy()
            self.ncust = self.ncust.collect().lazy()
            self.maxPropPerCustomer = self.maxPropPerCustomer.collect().lazy()

        self.customersummary = self.getCustomerSummary(self.th, import_strategy)
        self.countsPerStage = self.getCountsPerStage(
            self.customersummary, import_strategy
        )

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
        import_strategy: Literal["eager", "lazy"] = "eager",
    ) -> pl.DataFrame:
        """Computes the summary of propensities for all customers

        Parameters
        ----------
        th: Optional[float]
            The threshold to consider an action 'good'.
            If a customer has actions with propensity above this,
            the customer has at least one relevant action.
            If not given, will default to 5th quantile.
        verbose: bool, default = True
            Whether to print out the execution times
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

        if import_strategy == "eager":
            return df.collect().lazy()
        else:
            return df

    def getCountsPerStage(
        self,
        customersummary: Optional[pl.DataFrame] = None,
        import_strategy: Literal["eager", "lazy"] = "eager",
    ) -> pl.DataFrame:
        """Generates an aggregated view per stage.

        Parameters
        ----------
        customersummary : Optional[pl.DataFrame]
            Optional override of the customer summary,
            which can be generated by getCustomerSummary().
        import_strategy: Literal['eager', 'lazy'], default = 'eager'
            Whether to import the file fully to memory, or scan the file
            When data fits into memory, 'eager' is typically more efficient
            However, when data does not fit, the lazy methods typically allow
            you to still use the data.
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
        if import_strategy == "eager":
            return df.collect().lazy()
        else:
            return df

    def addCountsPerThresholdRange(
        self, start: float, stop: float, step: float, verbose: bool = True
    ):
        """Adds the counts per stage for a range of quantiles.

        In the background, uses numpy's arange function
        to generate the range of quantiles.
        Then, for each quantile the counts per stage are computed,
        and added to the 'countsPerThreshold' dictionary.
        As optimization, if a quantile produces a threshold that is
        already previously computed, it simply adds that computed dataframe.

        Parameters
        ----------
        start : float
            The starting quantile
        stop : float
            The ending quantile
        step : float
            The steps to compute between start and stop
        verbose : bool, default = True
            Whether to print out the progress of computation
        """
        to_add = set(np.arange(start, stop, step)) - self.countsPerThreshold.keys()
        if len(to_add) > 0:
            self.skippedquantiles = []
            for quantile in tqdm(to_add, disable=not verbose):
                th = self.df.filter(pl.col("pyStage") == "Eligibility")[
                    "pyModelPropensity"
                ].quantile(quantile)

                skipped = False
                for quantile2, data in self.countsPerThreshold.items():
                    if data[0] == th:
                        self.countsPerThreshold[quantile] = (th, data[1])
                        skipped = True
                        self.skippedquantiles.append((quantile2, th))
                        break

                if not skipped:
                    self.countsPerThreshold[quantile] = (
                        th,
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
                        .with_columns(
                            (self.ncust - pl.col("NoActions")).alias("NoActions")
                        ),
                    )
        else:
            if verbose:
                print("All thresholds already computed.")

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
            data = data.pdstools.sample(sampledN).collect()
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
        start: Optional[float] = None,
        stop: Optional[float] = None,
        step: Optional[float] = None,
        verbose: bool = True,
    ):
        """Plots the pie chart, split per stage.

        If start, stop and step are supplied,
        it will generate a slider to set the threshold
        dynamically throughout the given range of quantiles.
        If any of the is None, it will simply use the default
        threshold based on the 5th quantile.

        Parameters
        ----------
        start : float
            The starting quantile
        stop : float
            The ending quantile
        step : float
            The steps to compute between start and stop
        verbose : bool, default = True
            Whether to print out the progress of computation
        """
        if None in (start, stop, step):
            if start is not None:
                start, stop = start, start + 1
            else:
                start, stop = self.th, self.th + 1
            step = 10000

        fig = make_subplots(
            rows=1,
            cols=len(self.NBADStages),
            specs=[[{"type": "domain"}] * len(self.NBADStages)],
            subplot_titles=self.NBADStages,
        )
        self.addCountsPerThresholdRange(start, stop, step, verbose=verbose)
        founddefault = False
        rounding = 3
        arange = np.arange(start, stop, step)
        for i, quantile in enumerate(arange):
            if self.countsPerThreshold[quantile][0] == self.th:
                visible, founddefault = True, True
            else:
                visible = False
            for i, stage in enumerate(self.NBADStages):
                data = (
                    self.countsPerThreshold[quantile][1]
                    .filter(pl.col("pyStage") == stage)
                    .drop("pyStage")
                )
                fig.add_trace(
                    go.Pie(
                        values=list(data.to_numpy())[0],
                        labels=list(
                            data.rename(
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
        steps = []
        for i, quantile in enumerate(arange):
            step = dict(
                method="update",
                label=str(round(quantile, rounding)),
                args=[
                    {"visible": [False] * len(fig.data)},
                    {
                        "title": f"Propensity percentile: {str(round(quantile,rounding))}, propensity: {self.countsPerThreshold[quantile][0]}"
                    },
                ],
            )
            for x in range(i * 4, i * 4 + 4):
                step["args"][0]["visible"][x] = True
            steps.append(step)
        sliders = [
            dict(
                active=list(arange)[0],
                currentvalue={"prefix": "Propensity percentile: "},
                pad={"t": 50},
                steps=steps,
            )
        ]

        fig.update_layout(sliders=sliders)
        fig.update_traces(marker=dict(colors=["#219e3f", "#fca52e", "#cd001f"]))
        if not founddefault:
            for i in range(0, 4):
                fig.data[i].visible = True
        return fig

    def plotDistributionPerThreshold(self, target: str = "Quantile", **kwargs):
        """Plots the distribution of customers per threshold, per stage.

        Based on the precomputed data in self.countsPerThreshold,
        this function will plot the distribution per stage.

        To add more data points between a given range,
        simply pass all three arguments to this function:
        start, stop and step. Alternatively, you may
        call the self.addCountsPerThresholdRange() function,
        with the start, stop and step arguments outside of this call.

        Parameters
        ----------
        target : str, default = Quantile
            Determines which threshold to plot:
            based on the quantiles or the raw propensities.
            One of: {'Quantile', 'Propensity'}

        Keyword arguments
        -----------------
        start : float
            The starting quantile
        stop : float
            The ending quantile
        step : float
            The steps to compute between start and stop
        verbose : bool, default = True
            Whether to print out the progress of computation
        """
        if {"start", "stop", "step"}.issubset(kwargs):
            self.addCountsPerThresholdRange(
                kwargs.get("start"), kwargs.get("stop"), kwargs.get("step")
            )
        self.addCountsPerThresholdRange(0.01, 1, 0.1)

        df = list()
        for quantile, data in self.countsPerThreshold.items():
            target2 = data[0] if target.casefold() == "propensity" else quantile
            df.append(data[1].with_columns(pl.lit(target2).alias(target)))
        df = pl.concat(df).to_pandas().set_index(target)
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
            title=f"Distribution of offers per stage",
            labels={"value": "Number of people"},
            category_orders={"pyStage": self.NBADStages},
            template="none",
        )
        fig.update_layout(legend_title_text="Status")
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
            pl.LazyFrame({"pyStage": self.NBADStages})
            .join(
                df.groupby("pyStage")
                .agg(pl.col("pyName").value_counts().sort())
                .explode("pyName")
                .unnest("pyName"),
                on="pyStage",
            )
            .rename({level: "Name", "counts": "Count", "pyStage": "Stage"})
            .collect()
        )
        if return_df:
            return df

        fig = px.funnel(
            df.to_pandas(),
            y="Count",
            x="Stage",
            color="Name",
            text="Name",
            title=f"Distribution of {cat.casefold()} over the stages",
            template="none",
        )
        fig.update_xaxes(categoryorder="array", categoryarray=self.NBADStages)
        fig.update_layout(legend_title_text=cat)
        return fig
