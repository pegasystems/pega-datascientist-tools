import datetime
import itertools
import os
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Tuple,
    Union,
)

import polars as pl
import logging

from ..utils.types import QUERY
from ..utils.namespaces import LazyNamespace

from ..adm.CDH_Guidelines import CDHGuidelines
from ..utils import cdh_utils
from ..pega_io.File import read_ds_export


logger = logging.getLogger(__name__)
try:
    import plotly.express as px
    import plotly.graph_objects as go

    from ..utils import pega_template as pega_template
except ImportError as e:  # pragma: no cover
    logger.debug(f"Failed to import optional dependencies: {e}")

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go

COLORSCALE_TYPES = Union[List[Tuple[float, str]], List[str]]

Figure = Union[Any, "go.Figure"]

# T = TypeVar("T", bound="Plots")
# P = ParamSpec("P")


class PredictionPlots(LazyNamespace):
    """
    Plots for visualizing Prediction Studio data.

    This class provides various plotting methods to visualize prediction performance,
    lift, CTR, and response counts over time.
    """
    dependencies = ["plotly"]

    def __init__(self, prediction):
        self.prediction = prediction
        super().__init__()

    def _prediction_trend(
        self,
        period: str,
        query: Optional[QUERY],
        metric: str,
        title: str,
        **kwargs
    ):
        """Internal method to create trend plots for various metrics.

        Parameters
        ----------
        period : str
            Time period for aggregation (e.g., "1d", "1w", "1mo")
        query : Optional[QUERY]
            Optional query to filter the data
        metric : str
            The metric to plot (e.g., "Performance", "Lift", "CTR")
        title : str
            Plot title
        **kwargs
            Additional keyword arguments passed directly to plotly.express.line
            See plotly.express.line documentation for all available options

        Returns
        -------
        tuple
            (plotly figure, dataframe with plot data)
        """
        plot_df = self.prediction.summary_by_channel(by_period=period).with_columns(
            Prediction=pl.format("{} ({})", pl.col.Channel, pl.col.Prediction),
        )

        plot_df = (
            cdh_utils._apply_query(plot_df, query)
            .with_columns(
                Period=pl.format(
                    "{} days",
                    ((pl.col("Duration") / 3600 / 24).round() + 1).cast(pl.Int32),
                )
            )
            .rename({"DateRange Min": "Date"})
        )

        date_range = (
            cdh_utils._apply_query(self.prediction.predictions, query)
            .select(
                pl.format(
                    "period: {} to {}",
                    pl.col("SnapshotTime").min().dt.to_string("%v"),
                    pl.col("SnapshotTime").max().dt.to_string("%v"),
                )
            )
            .collect()
            .item()
        )

        # plt = px.bar(
        #     plot_df.filter(pl.col("isMultiChannelPrediction").not_())
        #     .filter(pl.col("Channel") != "Unknown")
        #     .sort("Date")
        #     .collect(),
        #     x="Date",
        #     y=metric,
        #     barmode="group",
        #     facet_row=facet_row,
        #     facet_col=facet_col,
        #     color="Prediction",
        #     title=f"{title}<br><sub>{date_range}</sub>",
        #     template="pega",
        #     hover_data=hover_data,
        # )
        plt = px.line(
            plot_df.filter(pl.col("isMultiChannel").not_())
            .filter(pl.col("Channel") != "Unknown")
            .sort("Date")
            .collect(),
            x="Date",
            y=metric,
            color="Prediction",
            title=f"{title}<br><sub>{date_range}</sub>",
            template="pega",
            markers=True,
            **kwargs
        )

        plt.for_each_annotation(lambda a: a.update(text="")).update_layout(
            legend_title_text="Channel"
        )

        # Update axis titles if faceting is used
        if kwargs.get("facet_row") is not None:
            plt.update_yaxes(title="")
        if kwargs.get("facet_col") is not None:
            plt.update_xaxes(title="")

        return plt, plot_df

    def performance_trend(
        self,
        period: str = "1d",
        *,
        query: Optional[QUERY] = None,
        return_df: bool = False,
        **kwargs
    ):
        """Create a performance trend plot showing AUC over time.

        Parameters
        ----------
        period : str, optional
            Time period for aggregation (e.g., "1d", "1w", "1mo"), by default "1d"
        query : Optional[QUERY], optional
            Optional query to filter the data, by default None
        return_df : bool, optional
            If True, returns the dataframe used for plotting instead of the plot, by default False
        **kwargs
            Additional keyword arguments passed directly to plotly.express.line
            See plotly.express.line documentation for all available options

        Returns
        -------
        Union[Figure, pl.DataFrame]
            Either a plotly figure or the dataframe used for plotting if return_df is True
        """
        # Default hover data for performance plots
        hover_data = {
            "Period": True,
            "Positives": True,
            "Negatives": True,
            "Positives_Test": True,
            "Negatives_Test": True,
            "CTR_Test": ":.3%",
            "Positives_Control": True,
            "Negatives_Control": True,
            "CTR_Control": ":.3%",
            "Positives_NBA": True,
            "Negatives_NBA": True,
            "CTR_NBA": ":.3%",
        }
        
        # Merge default hover_data with any provided in kwargs
        if "hover_data" in kwargs:
            hover_data.update(kwargs.pop("hover_data"))
            
        plt, plt_data = self._prediction_trend(
            query=query,
            period=period,
            metric="Performance",
            title="Prediction Performance",
            hover_data=hover_data,
            **kwargs
        )
        if return_df:
            return plt_data

        plt.update_yaxes(range=[50, 100], title="Performance (AUC)")
        return plt

    def lift_trend(
        self,
        period: str = "1d",
        *,
        query: Optional[QUERY] = None,
        return_df: bool = False,
        **kwargs
    ):
        """Create a lift trend plot showing engagement lift over time.

        Parameters
        ----------
        period : str, optional
            Time period for aggregation (e.g., "1d", "1w", "1mo"), by default "1d"
        query : Optional[QUERY], optional
            Optional query to filter the data, by default None
        return_df : bool, optional
            If True, returns the dataframe used for plotting instead of the plot, by default False
        **kwargs
            Additional keyword arguments passed directly to plotly.express.line
            See plotly.express.line documentation for all available options

        Returns
        -------
        Union[Figure, pl.DataFrame]
            Either a plotly figure or the dataframe used for plotting if return_df is True
        """
        # Default hover data for lift plots
        hover_data = {
            "Period": True,
            "Positives": True,
            "Negatives": True,
            "Positives_Test": True,
            "Negatives_Test": True,
            "CTR_Test": ":.3%",
            "Positives_Control": True,
            "Negatives_Control": True,
            "CTR_Control": ":.3%",
            "Positives_NBA": True,
            "Negatives_NBA": True,
            "CTR_NBA": ":.3%",
        }
        
        # Merge default hover_data with any provided in kwargs
        if "hover_data" in kwargs:
            hover_data.update(kwargs.pop("hover_data"))
            
        plt, plt_data = self._prediction_trend(
            period=period,
            query=query,
            metric="Lift",
            title="Prediction Lift",
            hover_data=hover_data,
            **kwargs
        )
        if return_df:
            return plt_data

        data_max = plt_data.select(pl.col("Lift").max()).collect().item()
        plt.update_yaxes(
            range=[-1, max(1, data_max * 1.2)],
            tickformat=",.2%",
            title="Engagement Lift",
        )
        return plt

    def ctr_trend(
        self,
        period: str = "1d",
        facetting=False,
        *,
        query: Optional[QUERY] = None,
        return_df: bool = False,
        **kwargs
    ):
        """Create a CTR (Click-Through Rate) trend plot over time.

        Parameters
        ----------
        period : str, optional
            Time period for aggregation (e.g., "1d", "1w", "1mo"), by default "1d"
        facetting : bool, optional
            Whether to create facets by prediction, by default False
        query : Optional[QUERY], optional
            Optional query to filter the data, by default None
        return_df : bool, optional
            If True, returns the dataframe used for plotting instead of the plot, by default False
        **kwargs
            Additional keyword arguments passed directly to plotly.express.line
            See plotly.express.line documentation for all available options

        Returns
        -------
        Union[Figure, pl.DataFrame]
            Either a plotly figure or the dataframe used for plotting if return_df is True
        """
        # Default hover data for CTR plots
        hover_data = {
            "Period": True,
            "Positives": True,
            "Negatives": True,
            "Positives_Test": True,
            "Negatives_Test": True,
            "CTR_Test": ":.3%",
            "Positives_Control": True,
            "Negatives_Control": True,
            "CTR_Control": ":.3%",
            "Positives_NBA": True,
            "Negatives_NBA": True,
            "CTR_NBA": ":.3%",
        }
        
        # Merge default hover_data with any provided in kwargs
        if "hover_data" in kwargs:
            hover_data.update(kwargs.pop("hover_data"))
            
        # Handle facetting
        facet_kwargs = {}
        if facetting:
            facet_kwargs["facet_row"] = "Prediction"
            
        # Merge facet_kwargs with any provided in kwargs
        kwargs.update(facet_kwargs)
            
        plt, plt_data = self._prediction_trend(
            period=period,
            query=query,
            metric="CTR",
            title="Prediction CTR",
            hover_data=hover_data,
            **kwargs
        )
        if return_df:
            return plt_data

        plt.update_yaxes(tickformat=",.3%", rangemode="tozero")
        return plt

    def responsecount_trend(
        self,
        period: str = "1d",
        facetting=False,
        *,
        query: Optional[QUERY] = None,
        return_df: bool = False,
        **kwargs
    ):
        """Create a response count trend plot showing total responses over time.

        Parameters
        ----------
        period : str, optional
            Time period for aggregation (e.g., "1d", "1w", "1mo"), by default "1d"
        facetting : bool, optional
            Whether to create facets by prediction, by default False
        query : Optional[QUERY], optional
            Optional query to filter the data, by default None
        return_df : bool, optional
            If True, returns the dataframe used for plotting instead of the plot, by default False
        **kwargs
            Additional keyword arguments passed directly to plotly.express.line
            See plotly.express.line documentation for all available options

        Returns
        -------
        Union[Figure, pl.DataFrame]
            Either a plotly figure or the dataframe used for plotting if return_df is True
        """
        # Default hover data for response count plots
        hover_data = {
            "Period": True,
            "Positives": True,
            "Negatives": True,
            "Positives_Test": True,
            "Negatives_Test": True,
            "CTR_Test": ":.3%",
            "Positives_Control": True,
            "Negatives_Control": True,
            "CTR_Control": ":.3%",
            "Positives_NBA": True,
            "Negatives_NBA": True,
            "CTR_NBA": ":.3%",
        }
        
        # Merge default hover_data with any provided in kwargs
        if "hover_data" in kwargs:
            hover_data.update(kwargs.pop("hover_data"))
            
        # Handle facetting
        facet_kwargs = {}
        if facetting:
            facet_kwargs["facet_col"] = "Prediction"
            
        # Merge facet_kwargs with any provided in kwargs
        kwargs.update(facet_kwargs)
            
        plt, plt_data = self._prediction_trend(
            period=period,
            query=query,
            metric="Responses",
            title="Prediction Responses",
            hover_data=hover_data,
            **kwargs
        )
        if return_df:
            return plt_data

        plt.update_layout(yaxis_title="Responses")
        return plt


class Prediction:
    """
    Monitor and analyze Pega Prediction Studio Predictions.

    To initialize this class, either
    1. Initialize directly with the df polars LazyFrame
    2. Use one of the class methods

    This class will read in the data from different sources, properly structure them
    for further analysis, and apply correct typing and useful renaming.

    There is also a "namespace" that you can call from this class:

    - `.plot` contains ready-made plots to analyze the prediction data with

    Parameters
    ----------
    df : pl.LazyFrame
        The Polars LazyFrame representation of the prediction data.
    query : QUERY, optional
        An optional query to apply to the input data.
        For details, see :meth:`pdstools.utils.cdh_utils._apply_query`.

    Examples
    --------
    >>> pred = Prediction.from_ds_export('/my_export_folder/predictions.zip')
    >>> pred = Prediction.from_mock_data(days=70)
    >>> from pdstools import Prediction
    >>> import polars as pl
    >>> pred = Prediction(
             df = pl.scan_parquet('predictions.parquet'),
             query = {"Class":["DATA-DECISION-REQUEST-CUSTOMER-CDH"]}
             )

    See Also
    --------
    pdstools.prediction.PredictionPlots : The out of the box plots on the Prediction data
    pdstools.utils.cdh_utils._apply_query : How to query the Prediction class and methods
    """

    predictions: pl.LazyFrame
    plot: PredictionPlots

    # These are pretty strict conditions - many configurations appear not to satisfy these
    # perhaps the Total = Test + Control is no longer met when Impact Analyzer is around
    prediction_validity_expr = (
        (pl.col("Positives") > 0)
        & (pl.col("Positives_Test") > 0)
        & (pl.col("Positives_Control") > 0)
        & (pl.col("Negatives") > 0)
        & (pl.col("Negatives_Test") > 0)
        & (pl.col("Negatives_Control") > 0)
        # & (
        #     pl.col("Positives")
        #     == (pl.col("Positives_Test") + pl.col("Positives_Control"))
        # )
        # & (
        #     pl.col("Negatives")
        #     == (pl.col("Negatives_Test") + pl.col("Negatives_Control"))
        # )
    )

    def __init__(
        self,
        df: pl.LazyFrame,
        *,
        query: Optional[QUERY] = None,
    ):
        """Initialize the Prediction class

        Parameters
        ----------
        df : pl.LazyFrame
            The read in data as a Polars LazyFrame
        query : QUERY, optional
            An optional query to apply to the input data.
            For details, see :meth:`pdstools.utils.cdh_utils._apply_query`.
        """
        self.cdh_guidelines = CDHGuidelines()
        self.plot = PredictionPlots(prediction=self)

        predictions_raw_data_prepped = (
            (
                df.filter(pl.col.pyModelType == "PREDICTION")
                .with_columns(
                    Performance=pl.col("pyValue").cast(pl.Float32, strict=False),
                )
                .rename(
                    {
                        "pyPositives": "Positives",
                        "pyNegatives": "Negatives",
                        "pyCount": "ResponseCount",
                    }
                )
            )
            # collect/lazy hopefully helps to zoom in into issues
            .collect().lazy()
        )
        schema = predictions_raw_data_prepped.collect_schema()
        if not schema.get("pySnapShotTime").is_temporal():  # pl.Datetime
            predictions_raw_data_prepped = predictions_raw_data_prepped.with_columns(
                SnapshotTime=cdh_utils.parse_pega_date_time_formats(
                    "pySnapShotTime", timestamp_dtype=pl.Date
                )
            )
        else:
            predictions_raw_data_prepped = predictions_raw_data_prepped.with_columns(
                SnapshotTime=pl.col("pySnapShotTime").cast(pl.Date)
            )

        # Below looks like a pivot.. but we want to make sure Control, Test and NBA
        # columns are always there...
        # TODO we may want to assert that this results in exactly one record for
        # every combination of model ID and snapshot time.
        counts_control = predictions_raw_data_prepped.filter(
            pl.col.pyDataUsage == "Control"
        ).select(
            ["pyModelId", "SnapshotTime", "Positives", "Negatives", "ResponseCount"]
        )
        counts_test = predictions_raw_data_prepped.filter(
            pl.col.pyDataUsage == "Test"
        ).select(
            ["pyModelId", "SnapshotTime", "Positives", "Negatives", "ResponseCount"]
        )
        counts_NBA = predictions_raw_data_prepped.filter(
            pl.col.pyDataUsage == "NBA"
        ).select(
            ["pyModelId", "SnapshotTime", "Positives", "Negatives", "ResponseCount"]
        )

        self.predictions = (
            # Performance is taken for the records with a filled in "snapshot type".
            # The numbers of positives, negatives may not make sense but are included
            # anyways.
            predictions_raw_data_prepped.filter(pl.col.pySnapshotType == "Daily")
            .select(
                [
                    "pyModelId",
                    "SnapshotTime",
                    "Positives",
                    "Negatives",
                    "ResponseCount",
                    "Performance",
                ]
            )
            .join(counts_test, on=["pyModelId", "SnapshotTime"], suffix="_Test")
            .join(counts_control, on=["pyModelId", "SnapshotTime"], suffix="_Control")
            .join(
                counts_NBA, on=["pyModelId", "SnapshotTime"], suffix="_NBA", how="left"
            )
            .with_columns(
                Class=pl.col("pyModelId").str.extract(r"(.+)!.+"),
                ModelName=pl.col("pyModelId").str.extract(r".+!(.+)"),
                CTR=pl.col("Positives") / (pl.col("Positives") + pl.col("Negatives")),
                CTR_Test=pl.col("Positives_Test")
                / (pl.col("Positives_Test") + pl.col("Negatives_Test")),
                CTR_Control=pl.col("Positives_Control")
                / (pl.col("Positives_Control") + pl.col("Negatives_Control")),
                CTR_NBA=pl.col("Positives_NBA")
                / (pl.col("Positives_NBA") + pl.col("Negatives_NBA")),
            )
            .with_columns(
                CTR_Lift=(pl.col("CTR_Test") - pl.col("CTR_Control"))
                / pl.col("CTR_Control"),
                isValidPrediction=self.prediction_validity_expr,
            )
            .sort(["pyModelId", "SnapshotTime"])
        )

        self.predictions = cdh_utils._apply_query(self.predictions, query)

    @classmethod
    def from_ds_export(
        cls,
        predictions_filename: Union[os.PathLike, str],
        base_path: Union[os.PathLike, str] = ".",
        *,
        query: Optional[QUERY] = None,
    ):
        """Import from a Pega Dataset Export of the PR_DATA_DM_SNAPSHOTS table.

        Parameters
        ----------
        predictions_filename : Union[os.PathLike, str]
            The full path or name (if base_path is given) to the prediction snapshot files
        base_path : Union[os.PathLike, str], optional
            A base path to provide if predictions_filename is not given as a full path, by default "."
        query : Optional[QUERY], optional
            An optional argument to filter out selected data, by default None

        Returns
        -------
        Prediction
            The properly initialized Prediction class

        Examples
        --------
        >>> from pdstools import Prediction
        >>> pred = Prediction.from_ds_export('predictions.zip', '/my_export_folder')

        Note
        ----
        By default, the dataset export in Infinity returns a zip file per table.
        You do not need to open up this zip file! You can simply point to the zip,
        and this method will be able to read in the underlying data.

        See Also
        --------
        pdstools.pega_io.File.read_ds_export : More information on file compatibility
        pdstools.utils.cdh_utils._apply_query : How to query the Prediction class and methods
        """
        predictions_raw_data = read_ds_export(predictions_filename, base_path)
        return cls(predictions_raw_data, query=query)

    @classmethod
    def from_s3(cls):
        """Not implemented yet. Please let us know if you would like this functionality!

        Returns
        -------
        Prediction
            The properly initialized Prediction class
        """
        ...

    @classmethod
    def from_dataflow_export(cls):
        """Import from a data flow, such as the Prediction Studio export. Not implemented yet. Please let us know if you would like this functionality!

        Returns
        -------
        Prediction
            The properly initialized Prediction class
        """
        ...

    @classmethod
    def from_pdc(
        cls,
        df: pl.LazyFrame,
        *,
        return_df=False,
        query: Optional[QUERY] = None,
    ):
        """Import from (Pega-internal) PDC data, which is a combination of the PR_DATA_DM_SNAPSHOTS and PR_DATA_DM_ADMMART_MDL_FACT tables.

        Parameters
        ----------
        df : pl.LazyFrame
            The Polars LazyFrame containing the PDC data
        return_df : bool, optional
            If True, returns the processed DataFrame instead of initializing the class, by default False
        query : Optional[QUERY], optional
            An optional query to apply to the input data, by default None

        Returns
        -------
        Union[Prediction, pl.LazyFrame]
            Either the initialized Prediction class or the processed DataFrame if return_df is True

        See Also
        --------
        pdstools.utils.cdh_utils._read_pdc : More information on PDC data processing
        pdstools.utils.cdh_utils._apply_query : How to query the Prediction class and methods
        """
        pdc_data = cdh_utils._read_pdc(df)

        snapshotType = "Daily"
        prediction_data = (
            pdc_data.filter(pl.col("ModelType").str.starts_with("Prediction"))
            .filter(pl.col("Name") == "auc")
            .with_columns(
                pyModelId=pl.format("{}!{}", pl.col("ModelClass"), pl.col("ModelName")),
                # pyUnscaledPerformance=(pl.col("Performance").cast(pl.Float64) / 100), # not unscaled, it's not 'flipped' so can be < 50
                pyDataUsage=pl.col("ModelType").str.extract(r".+_(Test|Control|NBA)"),
                pyModelType=pl.lit("PREDICTION"),
                # pysnapshotday=pl.col("SnapshotTime").str.slice(0, 8), # I don't think we need that. If we do, be careful that SnapshotTime can be a parsed datetime already.
                pySnapshotType=pl.lit(snapshotType),
            )
            .rename(
                {
                    "SnapshotTime": "pySnapShotTime",
                    "Positives": "pyPositives",
                    "Negatives": "pyNegatives",
                    "ResponseCount": "pyCount",
                    "Name": "pyName",
                    "Performance": "pyValue",
                }
            )
            .cast(
                {
                    "pyNegatives": pl.Float64,
                    "pyPositives": pl.Float64,
                    "pyCount": pl.Float64,
                }
            )
            .drop(
                [
                    "ModelClass",
                    "ModelID",
                    "ModelName",
                    "ModelType",
                    "ADMModelType",
                    "TotalPositives",
                    "TotalResponses",
                ]
                + [
                    c
                    for c in [
                        "pxObjClass",
                        "pzInsKey",
                        "Channel",
                        "Direction",
                        "Issue",
                        "Group",
                    ]
                    if c in pdc_data.collect_schema().names()
                ]
            )
        )

        if return_df:
            return prediction_data

        return cls(prediction_data, query=query)

    @classmethod
    def from_mock_data(cls, days=70):
        """Create a Prediction instance with mock data for testing and demonstration purposes.

        Parameters
        ----------
        days : int, optional
            Number of days of mock data to generate, by default 70

        Returns
        -------
        Prediction
            The initialized Prediction class with mock data

        Examples
        --------
        >>> from pdstools import Prediction
        >>> pred = Prediction.from_mock_data(days=30)
        >>> pred.plot.performance_trend()
        """
        n_conditions = 4  # can't change this
        n_predictions = 3  # tied to the data below
        now = datetime.datetime.now()

        def _interpolate(min, max, i, n):
            return min + (max - min) * i / (n - 1)

        mock_prediction_data = (
            pl.LazyFrame(
                {
                    "pySnapShotTime": sorted(
                        [
                            cdh_utils.to_prpc_date_time(
                                now - datetime.timedelta(days=i)
                            )[
                                0:15
                            ]  # Polars doesn't like time zones like GMT+0200
                            for i in range(days)
                        ]
                        * n_conditions
                        * n_predictions
                    ),
                    "pyModelId": (
                        [
                            "DATA-DECISION-REQUEST-CUSTOMER!PredictOutboundEmailPropensity"
                        ]
                        * n_conditions
                        + ["DATA-DECISION-REQUEST-CUSTOMER!PREDICTMOBILEPROPENSITY"]
                        * n_conditions
                        + ["DATA-DECISION-REQUEST-CUSTOMER!PREDICTWEBPROPENSITY"]
                        * n_conditions
                    )
                    * days,
                    "pyModelType": "PREDICTION",
                    "pySnapshotType": ["Daily", "Daily", "Daily", None]
                    * n_predictions
                    * days,
                    "pyDataUsage": ["Control", "Test", "NBA", ""]
                    * n_predictions
                    * days,  # Control=Random, Test=Model
                    # "pyPositives": (
                    #     [100, 160, 120, None] + [200, 420, 250, None] + [350, 700, 380, None]
                    # )
                    # * n_days,
                    "pyPositives": list(
                        itertools.chain.from_iterable(
                            [
                                [
                                    _interpolate(100, 100, p, days),
                                    _interpolate(160, 200, p, days),
                                    _interpolate(120, 120, p, days),
                                    None,
                                ]
                                + [
                                    _interpolate(120, 120, p, days),
                                    _interpolate(250, 300, p, days),
                                    _interpolate(150, 150, p, days),
                                    None,
                                ]
                                + [
                                    _interpolate(1400, 1400, p, days),
                                    _interpolate(2800, 4000, p, days),
                                    _interpolate(1520, 1520, p, days),
                                    None,
                                ]
                                for p in range(0, days)
                            ]
                        )
                    ),
                    "pyNegatives": (
                        [10000] * n_conditions
                        + [6000] * n_conditions
                        + [40000] * n_conditions
                    )
                    * days,
                    "pyValue": list(
                        itertools.chain.from_iterable(
                            [
                                [_interpolate(60.0, 65.0, p, days)] * n_conditions
                                + [_interpolate(70.0, 73.0, p, days)] * n_conditions
                                + [_interpolate(66.0, 68.0, p, days)] * n_conditions
                                for p in range(0, days)
                            ]
                        )
                    ),
                }
            ).sort(["pySnapShotTime", "pyModelId", "pySnapshotType"])
            # .with_columns(
            #     pl.col("pyPositives").cum_sum().over(["pyModelId", "pySnapshotType"]),
            #     pl.col("pyNegatives").cum_sum().over(["pyModelId", "pySnapshotType"]),
            # )
            .with_columns(pyCount=pl.col("pyPositives") + pl.col("pyNegatives"))
        )

        return cls(mock_prediction_data)

    @property
    def is_available(self) -> bool:
        """Check if prediction data is available.

        Returns
        -------
        bool
            True if prediction data is available, False otherwise
        """
        return len(self.predictions.head(1).collect()) > 0

    @property
    def is_valid(self) -> bool:
        """Check if prediction data is valid.

        A valid prediction meets the criteria defined in prediction_validity_expr,
        which requires positive and negative responses in both test and control groups.

        Returns
        -------
        bool
            True if prediction data is valid, False otherwise
        """
        return (
            self.is_available
            # or even stronger: pos = pos_test + pos_control
            and self.predictions.select(self.prediction_validity_expr.all())
            .collect()
            .item()
        )

    def summary_by_channel(
        self,
        custom_predictions: Optional[List[List]] = None,
        *,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        window: Optional[Union[int, datetime.timedelta]] = None,
        by_period: Optional[str] = None,
        debug: bool = False,
    ) -> pl.LazyFrame:
        """Summarize prediction per channel

        Parameters
        ----------
        custom_predictions : Optional[List[CDH_Guidelines.NBAD_Prediction]], optional
            Optional list with custom prediction name to channel mappings. Defaults to None.
        start_date : datetime.datetime, optional
            Start date of the summary period. If None (default) uses the end date minus the window, or if both absent, the earliest date in the data
        end_date : datetime.datetime, optional
            End date of the summary period. If None (default) uses the start date plus the window, or if both absent, the latest date in the data
        window : int or datetime.timedelta, optional
            Number of days to use for the summary period or an explicit timedelta. If None (default) uses the whole period. Can't be given if start and end date are also given.
        by_period : str, optional
            Optional additional grouping by time period. Format string as in polars.Expr.dt.truncate (https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.truncate.html), for example "1mo", "1w", "1d" for calendar month, week day. Defaults to None.
        debug : bool, optional
            If True, enables debug mode for additional logging or outputs. Defaults to False.

        Returns
        -------
        pl.LazyFrame
            Summary across all Predictions as a dataframe with the following fields:

            Time and Configuration Fields:
            - DateRange Min - The minimum date in the summary time range
            - DateRange Max - The maximum date in the summary time range
            - Duration - The duration in seconds between the minimum and maximum snapshot times
            - Prediction: The prediction name
            - Channel: The channel name
            - Direction: The direction (e.g., Inbound, Outbound)
            - ChannelDirectionGroup: Combined Channel/Direction identifier
            - isValid: Boolean indicating if the prediction data is valid
            - usesNBAD: Boolean indicating if this is a standard NBAD prediction
            - isMultiChannel: Boolean indicating if this is a multichannel prediction
            - ControlPercentage: Percentage of responses in control group
            - TestPercentage: Percentage of responses in test group

            Performance Metrics:
            - Performance: Weighted model performance (AUC)
            - Positives: Sum of positive responses
            - Negatives: Sum of negative responses
            - Responses: Sum of all responses
            - Positives_Test: Sum of positive responses in test group
            - Positives_Control: Sum of positive responses in control group
            - Positives_NBA: Sum of positive responses in NBA group
            - Negatives_Test: Sum of negative responses in test group
            - Negatives_Control: Sum of negative responses in control group
            - Negatives_NBA: Sum of negative responses in NBA group
            - CTR: Clickthrough rate (Positives over Positives + Negatives)
            - CTR_Test: Clickthrough rate for test group (model propensitities)
            - CTR_Control: Clickthrough rate for control group (random propensities)
            - CTR_NBA: Clickthrough rate for NBA group (available only when Impact Analyzer is used)
            - Lift: Lift in Engagement when testing prioritization with just Adaptive Models vs just Random Propensity

            Technology Usage Indicators:
            - usesImpactAnalyzer: Boolean indicating if Impact Analyzer is used
        """
        if not custom_predictions:
            custom_predictions = []

        start_date, end_date = cdh_utils._get_start_end_date_args(
            self.predictions, start_date, end_date, window
        )

        query = pl.col("SnapshotTime").is_between(start_date, end_date)
        prediction_data = cdh_utils._apply_query(
            self.predictions, query=query, allow_empty=True
        )

        if by_period is not None:
            period_expr = [
                pl.col("SnapshotTime")
                .dt.truncate(by_period)
                .cast(pl.Date)
                .alias("Period")
            ]
        else:
            period_expr = []

        return (
            prediction_data.with_columns(pl.col("ModelName").str.to_uppercase())
            .join(
                self.cdh_guidelines.get_predictions_channel_mapping(
                    custom_predictions
                ).lazy(),
                left_on="ModelName",
                right_on="Prediction",
                how="left",
            )
            .rename({"ModelName": "Prediction"})
            .with_columns(
                [
                    pl.when(pl.col("Channel").is_null())
                    .then(pl.lit("Unknown"))
                    .otherwise(pl.col("Channel"))
                    .alias("Channel"),
                    pl.when(pl.col("Direction").is_null())
                    .then(pl.lit("Unknown"))
                    .otherwise(pl.col("Direction"))
                    .alias("Direction"),
                    self.cdh_guidelines.is_standard_prediction().alias("usesNBAD"),
                    pl.when(pl.col("isMultiChannel").is_null())
                    .then(pl.lit(False))
                    .otherwise(pl.col("isMultiChannel"))
                    .alias("isMultiChannel"),
                ]
                + period_expr
            )
            .group_by(
                [
                    "Prediction",
                    "Channel",
                    "Direction",
                    "usesNBAD",
                    "isMultiChannel",
                ]
                + (["Period"] if by_period is not None else [])
            )
            .agg(
                pl.col("SnapshotTime").min().cast(pl.Date).alias("DateRange Min"),
                pl.col("SnapshotTime").max().cast(pl.Date).alias("DateRange Max"),
                (pl.col("SnapshotTime").max() - pl.col("SnapshotTime").min())
                .dt.total_seconds()
                .alias("Duration"),
                cdh_utils.weighted_performance_polars().alias("Performance"),
                pl.col("Positives").sum(),
                pl.col("Negatives").sum(),
                pl.col("ResponseCount").sum().alias("Responses"),
                pl.col("Positives_Test").sum(),
                pl.col("Positives_Control").sum(),
                pl.col("Positives_NBA").sum(),
                pl.col("Negatives_Test").sum(),
                pl.col("Negatives_Control").sum(),
                pl.col("Negatives_NBA").sum(),
            )
            .with_columns(
                usesImpactAnalyzer=(pl.col("Positives_NBA") > 0)
                & (pl.col("Negatives_NBA") > 0),
                ControlPercentage=100.0
                * (pl.col("Positives_Control") + pl.col("Negatives_Control"))
                / (
                    pl.col("Positives_Test")
                    + pl.col("Negatives_Test")
                    + pl.col("Positives_Control")
                    + pl.col("Negatives_Control")
                    + pl.col("Positives_NBA")
                    + pl.col("Negatives_NBA")
                ),
                TestPercentage=100.0
                * (pl.col("Positives_Test") + pl.col("Negatives_Test"))
                / (
                    pl.col("Positives_Test")
                    + pl.col("Negatives_Test")
                    + pl.col("Positives_Control")
                    + pl.col("Negatives_Control")
                    + pl.col("Positives_NBA")
                    + pl.col("Negatives_NBA")
                ),
                CTR=pl.col("Positives") / (pl.col("Positives") + pl.col("Negatives")),
                CTR_Test=pl.col("Positives_Test")
                / (pl.col("Positives_Test") + pl.col("Negatives_Test")),
                CTR_Control=pl.col("Positives_Control")
                / (pl.col("Positives_Control") + pl.col("Negatives_Control")),
                CTR_NBA=pl.col("Positives_NBA")
                / (pl.col("Positives_NBA") + pl.col("Negatives_NBA")),
                ChannelDirectionGroup=pl.when(
                    pl.col("Channel").is_not_null()
                    & pl.col("Direction").is_not_null()
                    & pl.col("Channel").is_in(["Other", "Unknown", ""]).not_()
                    & pl.col("Direction").is_in(["Other", "Unknown", ""]).not_()
                    & pl.col("isMultiChannel").not_()
                )
                .then(pl.concat_str(["Channel", "Direction"], separator="/"))
                .otherwise(pl.lit("Other")),
                isValid=self.prediction_validity_expr,
            )
            .with_columns(
                Lift=(pl.col("CTR_Test") - pl.col("CTR_Control"))
                / pl.col("CTR_Control"),
            )
            .drop([] if debug else ([] + ([] if by_period is None else ["Period"])))
            .sort("Prediction", "DateRange Min")
        )

    # TODO rethink use of multi-channel. If the only valid predictions are multi-channel predictions
    # then use those. If there are valid non-multi-channel predictions then only use those.
    def overall_summary(
        self,
        custom_predictions: Optional[List[List]] = None,
        *,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        window: Optional[Union[int, datetime.timedelta]] = None,
        by_period: Optional[str] = None,
        debug: bool = False,
    ) -> pl.LazyFrame:
        """Overall prediction summary. Only valid prediction data is included.

        Parameters
        ----------
        custom_predictions : Optional[List[CDH_Guidelines.NBAD_Prediction]], optional
            Optional list with custom prediction name to channel mappings. Defaults to None.
        start_date : datetime.datetime, optional
            Start date of the summary period. If None (default) uses the end date minus the window, or if both absent, the earliest date in the data
        end_date : datetime.datetime, optional
            End date of the summary period. If None (default) uses the start date plus the window, or if both absent, the latest date in the data
        window : int or datetime.timedelta, optional
            Number of days to use for the summary period or an explicit timedelta. If None (default) uses the whole period. Can't be given if start and end date are also given.
        by_period : str, optional
            Optional additional grouping by time period. Format string as in polars.Expr.dt.truncate (https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.truncate.html), for example "1mo", "1w", "1d" for calendar month, week day. Defaults to None.
        debug : bool, optional
            If True, enables debug mode for additional logging or outputs. Defaults to False.

        Returns
        -------
        pl.LazyFrame
            Summary across all Predictions as a dataframe with the following fields:

            Time and Configuration Fields:
            - DateRange Min - The minimum date in the summary time range
            - DateRange Max - The maximum date in the summary time range
            - Duration - The duration in seconds between the minimum and maximum snapshot times
            - ControlPercentage: Weighted average percentage of control group responses
            - TestPercentage: Weighted average percentage of test group responses
            - usesNBAD: Boolean indicating if any of the predictions is a standard NBAD prediction

            Performance Metrics:
            - Performance: Weighted average performance across all valid channels
            - Positives Inbound: Sum of positive responses across all valid inbound channels
            - Positives Outbound: Sum of positive responses across all valid outbound channels
            - Responses Inbound: Sum of all responses across all valid inbound channels
            - Responses Outbound: Sum of all responses across all valid outbound channels
            - Overall Lift: Weighted average lift across all valid channels
            - Minimum Negative Lift: The lowest negative lift value found

            Channel Statistics:
            - Number of Valid Channels: Count of unique valid channel/direction combinations
            - Channel with Minimum Negative Lift: Channel with the lowest negative lift value

            Technology Usage Indicators:
            - usesImpactAnalyzer: Boolean indicating if any channel uses Impact Analyzer
        """

        # start_date, end_date = cdh_utils.get_start_end_date_args(
        #     self.datamart.model_data, start_date, end_date, window
        # )

        channel_summary = self.summary_by_channel(
            custom_predictions=custom_predictions,
            start_date=start_date,
            end_date=end_date,
            window=window,
            by_period=by_period,
            debug=True,  # should give us Period
        )

        if (
            channel_summary.select(
                (pl.col("isMultiChannel").not_() & pl.col("isValid")).any()
            )
            .collect()
            .item()
        ):
            # There are valid non-multi-channel predictions
            validity_filter_expr = pl.col("isMultiChannel").not_() & pl.col("isValid")
        else:
            validity_filter_expr = pl.col("isValid")

        return (
            channel_summary.filter(validity_filter_expr)
            .group_by(["Period"] if by_period is not None else None)
            .agg(
                pl.col("DateRange Min").min(),
                pl.col("DateRange Max").max(),
                pl.col("Duration").max(),
                pl.concat_str(["Channel", "Direction"], separator="/")
                .n_unique()
                .alias("Number of Valid Channels"),
                cdh_utils.weighted_average_polars("Lift", "Responses").alias(
                    "Overall Lift"
                ),
                cdh_utils.weighted_performance_polars("Performance", "Responses").alias(
                    "Performance"
                ),
                pl.col("Positives")
                .filter(Direction="Inbound")
                .sum()
                .alias("Positives Inbound"),
                pl.col("Positives")
                .filter(Direction="Outbound")
                .sum()
                .alias("Positives Outbound"),
                pl.col("Responses")
                .filter(Direction="Inbound")
                .sum()
                .alias("Responses Inbound"),
                pl.col("Responses")
                .filter(Direction="Outbound")
                .sum()
                .alias("Responses Outbound"),
                pl.col("Channel")
                .filter((pl.col("Lift") == pl.col("Lift").min()) & (pl.col("Lift") < 0))
                .first()
                .alias("Channel with Minimum Negative Lift"),
                pl.col("Lift")
                .filter((pl.col("Lift") == pl.col("Lift").min()) & (pl.col("Lift") < 0))
                .first()
                .alias("Minimum Negative Lift"),
                pl.col("usesImpactAnalyzer"),
                cdh_utils.weighted_average_polars(
                    "ControlPercentage", "Responses"
                ).alias("ControlPercentage"),
                cdh_utils.weighted_average_polars("TestPercentage", "Responses").alias(
                    "TestPercentage"
                ),
                pl.col("usesNBAD").any(ignore_nulls=False),
            )
            .drop(["literal"] if by_period is None else [])  # created by null group
            .with_columns(
                # CTR=(pl.col("Positives")) / (pl.col("Responses")),
                usesImpactAnalyzer=pl.col("usesImpactAnalyzer").list.any(),
            )
            .drop([] if debug else ([] + ([] if by_period is None else ["Period"])))
            .sort("DateRange Min")
        )
