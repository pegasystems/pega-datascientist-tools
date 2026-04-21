from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, overload

import polars as pl

from ..utils import cdh_utils
from ..utils.namespaces import LazyNamespace
from ..utils.types import QUERY

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

    from .Prediction import Prediction

logger = logging.getLogger(__name__)

# Importing pega_template at module load registers the "pega" Plotly
# template (and friends) as a side effect. Several plot methods across
# pdstools — including some in adm/Plots.py — pass `template="pega"`
# without importing pega_template themselves and rely on this side effect.
# Tracked in docs/plans/adm/plots-pega-template-registration.md.
try:
    from ..utils import pega_template as _pega_template  # noqa: F401
except ImportError:  # pragma: no cover
    logger.debug("plotly not installed; pega template not registered.")


class PredictionPlots(LazyNamespace):
    """Plots for visualizing Prediction Studio data.

    This class provides various plotting methods to visualize prediction performance,
    lift, CTR, and response counts over time.
    """

    dependencies = ["plotly"]
    dependency_group = "adm"

    def __init__(self, prediction: Prediction):
        self.prediction = prediction
        super().__init__()

    def _prediction_trend(
        self,
        period: str,
        query: QUERY | None,
        metric: str,
        title: str,
        **kwargs,
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
        import plotly.express as px

        from ..utils import pega_template as pega_template  # noqa: F401  (registers template)

        # Calculate date_range FIRST and collect it to avoid Polars lazy query race condition
        # where multiple lazy queries from the same LazyFrame can cause crashes
        queried_data = cdh_utils._apply_query(self.prediction.predictions, query)
        date_range = (
            queried_data.select(
                pl.format(
                    "period: {} to {}",
                    pl.col("SnapshotTime").min().dt.to_string("%v"),
                    pl.col("SnapshotTime").max().dt.to_string("%v"),
                ),
            )
            .collect()
            .item()
        )

        # Collect plot_df immediately to avoid having multiple lazy queries from same source
        plot_df = (
            self.prediction.summary_by_channel(every=period)
            .with_columns(
                Prediction=pl.format("{} ({})", pl.col.Channel, pl.col.Prediction),
            )
            .pipe(lambda df: cdh_utils._apply_query(df, query))
            .with_columns(
                Period=pl.format(
                    "{} days",
                    ((pl.col("Duration") / 3600 / 24).round() + 1).cast(pl.Int32),
                ),
            )
            .rename({"DateRange Min": "Date"})
            .collect()
            .lazy()
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
            **kwargs,
        )

        plt.for_each_annotation(lambda a: a.update(text="")).update_layout(
            legend_title_text="Channel",
        )

        # Update axis titles if faceting is used
        if kwargs.get("facet_row") is not None:
            plt.update_yaxes(title="")
        if kwargs.get("facet_col") is not None:
            plt.update_xaxes(title="")

        return plt, plot_df

    @overload
    def performance_trend(
        self,
        period: str = ...,
        *,
        query: QUERY | None = ...,
        return_df: Literal[False] = ...,
        **kwargs,
    ) -> Figure: ...
    @overload
    def performance_trend(
        self,
        period: str = ...,
        *,
        query: QUERY | None = ...,
        return_df: Literal[True],
        **kwargs,
    ) -> pl.LazyFrame: ...
    def performance_trend(
        self,
        period: str = "1d",
        *,
        query: QUERY | None = None,
        return_df: bool = False,
        **kwargs,
    ):
        """Create a performance trend plot showing AUC over time.

        Displays a line chart showing how prediction performance (AUC) changes over time,
        with configurable time period aggregation and filtering capabilities.

        Parameters
        ----------
        period : str, optional
            Time period for aggregation (e.g., "1d", "1w", "1mo"), by default "1d".
            Uses Polars truncate syntax for time period grouping.
        query : Optional[QUERY], optional
            Query to filter the prediction data. See pdstools.utils.cdh_utils._apply_query for details.
        return_df : bool, optional
            If True, returns the underlying data instead of the plot, by default False.
        **kwargs
            Additional keyword arguments passed directly to plotly.express.line.
            See plotly.express.line documentation for all available options.

        Returns
        -------
        Union[plotly.graph_objects.Figure, polars.LazyFrame]
            A Plotly figure object or the underlying data if return_df is True.

        Examples
        --------
        >>> # Basic performance trend plot
        >>> pred.plot.performance_trend()

        >>> # Weekly aggregated performance trend
        >>> pred.plot.performance_trend(period="1w")

        >>> # Performance trend with faceting by prediction
        >>> pred.plot.performance_trend(facet_row="Prediction")

        >>> # Get underlying data for custom analysis
        >>> data = pred.plot.performance_trend(return_df=True)

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
            **kwargs,
        )
        if return_df:
            return plt_data

        # Scale Performance from 0.5-1.0 internal format to 50-100 for display
        for trace in plt.data:
            if hasattr(trace, "y") and trace.y is not None:
                trace.y = tuple(y * 100 if y is not None else None for y in trace.y)

        plt.update_yaxes(range=[50, 100], title="Performance (AUC)")
        return plt

    @overload
    def lift_trend(
        self,
        period: str = ...,
        *,
        query: QUERY | None = ...,
        return_df: Literal[False] = ...,
        **kwargs,
    ) -> Figure: ...
    @overload
    def lift_trend(
        self,
        period: str = ...,
        *,
        query: QUERY | None = ...,
        return_df: Literal[True],
        **kwargs,
    ) -> pl.LazyFrame: ...
    def lift_trend(
        self,
        period: str = "1d",
        *,
        query: QUERY | None = None,
        return_df: bool = False,
        **kwargs,
    ):
        """Create a lift trend plot showing engagement lift over time.

        Displays a line chart showing how prediction engagement lift changes over time,
        comparing test group performance against control group baseline.

        Parameters
        ----------
        period : str, optional
            Time period for aggregation (e.g., "1d", "1w", "1mo"), by default "1d".
            Uses Polars truncate syntax for time period grouping.
        query : Optional[QUERY], optional
            Query to filter the prediction data. See pdstools.utils.cdh_utils._apply_query for details.
        return_df : bool, optional
            If True, returns the underlying data instead of the plot, by default False.
        **kwargs
            Additional keyword arguments passed directly to plotly.express.line.
            See plotly.express.line documentation for all available options.

        Returns
        -------
        Union[plotly.graph_objects.Figure, polars.LazyFrame]
            A Plotly figure object or the underlying data if return_df is True.

        Examples
        --------
        >>> # Basic lift trend plot
        >>> pred.plot.lift_trend()

        >>> # Monthly aggregated lift trend
        >>> pred.plot.lift_trend(period="1mo")

        >>> # Lift trend with custom query
        >>> pred.plot.lift_trend(query=pl.col("Channel") == "Email")

        >>> # Get underlying data
        >>> data = pred.plot.lift_trend(return_df=True)

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
            **kwargs,
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

    @overload
    def ctr_trend(
        self,
        period: str = ...,
        facetting: bool = ...,
        *,
        query: QUERY | None = ...,
        return_df: Literal[False] = ...,
        **kwargs,
    ) -> Figure: ...
    @overload
    def ctr_trend(
        self,
        period: str = ...,
        facetting: bool = ...,
        *,
        query: QUERY | None = ...,
        return_df: Literal[True],
        **kwargs,
    ) -> pl.LazyFrame: ...
    def ctr_trend(
        self,
        period: str = "1d",
        facetting=False,
        *,
        query: QUERY | None = None,
        return_df: bool = False,
        **kwargs,
    ):
        """Create a CTR (Click-Through Rate) trend plot over time.

        Displays a line chart showing how prediction click-through rates change over time,
        with optional faceting capabilities for comparing multiple predictions.

        Parameters
        ----------
        period : str, optional
            Time period for aggregation (e.g., "1d", "1w", "1mo"), by default "1d".
            Uses Polars truncate syntax for time period grouping.
        facetting : bool, optional
            Whether to create facets by prediction for side-by-side comparison, by default False.
        query : Optional[QUERY], optional
            Query to filter the prediction data. See pdstools.utils.cdh_utils._apply_query for details.
        return_df : bool, optional
            If True, returns the underlying data instead of the plot, by default False.
        **kwargs
            Additional keyword arguments passed directly to plotly.express.line.
            See plotly.express.line documentation for all available options.

        Returns
        -------
        Union[plotly.graph_objects.Figure, polars.LazyFrame]
            A Plotly figure object or the underlying data if return_df is True.

        Examples
        --------
        >>> # Basic CTR trend plot
        >>> pred.plot.ctr_trend()

        >>> # Weekly CTR trend with faceting
        >>> pred.plot.ctr_trend(period="1w", facetting=True)

        >>> # CTR trend with custom query
        >>> pred.plot.ctr_trend(query=pl.col("Prediction").str.contains("Email"))

        >>> # Get underlying data
        >>> data = pred.plot.ctr_trend(return_df=True)

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
            **kwargs,
        )
        if return_df:
            return plt_data

        plt.update_yaxes(tickformat=",.3%", rangemode="tozero")
        return plt

    @overload
    def responsecount_trend(
        self,
        period: str = ...,
        facetting: bool = ...,
        *,
        query: QUERY | None = ...,
        return_df: Literal[False] = ...,
        **kwargs,
    ) -> Figure: ...
    @overload
    def responsecount_trend(
        self,
        period: str = ...,
        facetting: bool = ...,
        *,
        query: QUERY | None = ...,
        return_df: Literal[True],
        **kwargs,
    ) -> pl.LazyFrame: ...
    def responsecount_trend(
        self,
        period: str = "1d",
        facetting=False,
        *,
        query: QUERY | None = None,
        return_df: bool = False,
        **kwargs,
    ):
        """Create a response count trend plot showing total responses over time.

        Displays a line chart showing how total response volumes change over time,
        useful for monitoring prediction usage and data volume trends.

        Parameters
        ----------
        period : str, optional
            Time period for aggregation (e.g., "1d", "1w", "1mo"), by default "1d".
            Uses Polars truncate syntax for time period grouping.
        facetting : bool, optional
            Whether to create facets by prediction for side-by-side comparison, by default False.
        query : Optional[QUERY], optional
            Query to filter the prediction data. See pdstools.utils.cdh_utils._apply_query for details.
        return_df : bool, optional
            If True, returns the underlying data instead of the plot, by default False.
        **kwargs
            Additional keyword arguments passed directly to plotly.express.line.
            See plotly.express.line documentation for all available options.

        Returns
        -------
        Union[plotly.graph_objects.Figure, polars.LazyFrame]
            A Plotly figure object or the underlying data if return_df is True.

        Examples
        --------
        >>> # Basic response count trend plot
        >>> pred.plot.responsecount_trend()

        >>> # Monthly response count trend with faceting
        >>> pred.plot.responsecount_trend(period="1mo", facetting=True)

        >>> # Response count trend for specific predictions
        >>> pred.plot.responsecount_trend(query=pl.col("Channel") == "Web")

        >>> # Get underlying data for analysis
        >>> data = pred.plot.responsecount_trend(return_df=True)

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
            **kwargs,
        )
        if return_df:
            return plt_data

        plt.update_layout(yaxis_title="Responses")
        return plt
