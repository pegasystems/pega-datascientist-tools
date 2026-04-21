"""Score-distribution plots for the classifier."""

from __future__ import annotations

import polars as pl

from ...utils import cdh_utils
from ...utils.plot_utils import Figure
from ...utils.types import QUERY
from ._base import _PlotsBase
from ._helpers import distribution_graph, requires


class _ScorePlotsMixin(_PlotsBase):
    @requires(
        combined_columns={
            "PredictorName",
            "Name",
            "BinIndex",
            "BinSymbol",
            "BinResponseCount",
            "BinPropensity",
            "ModelID",
        },
    )
    def score_distribution(
        self,
        model_id: str,
        *,
        active_range: bool = True,
        return_df: bool = False,
    ):
        """Generate a score distribution plot for a specific model.

        Parameters
        ----------
        model_id : str
            The ID of the model to generate the score distribution for
        active_range : bool, optional
            Whether to filter to active score range only, by default True
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Union[Figure, pl.LazyFrame]
            Plotly figure showing score distribution or DataFrame if return_df=True

        Raises
        ------
        ValueError
            If no data is available for the provided model ID

        Examples
        --------
        >>> # Score distribution for a specific model (active range only)
        >>> fig = dm.plot.score_distribution(model_id="M-1001")

        >>> # Include the full score range rather than just the active range
        >>> fig = dm.plot.score_distribution(model_id="M-1001", active_range=False)

        >>> # Retrieve the underlying binning data
        >>> df = dm.plot.score_distribution(model_id="M-1001", return_df=True)

        """
        df = (
            self.datamart.aggregates.last(table="combined_data")
            .select(
                {
                    "PredictorName",
                    "Name",
                    "BinIndex",
                    "BinSymbol",
                    "BinResponseCount",
                    "BinPropensity",
                    "ModelID",
                    "Configuration",
                }
                | set(
                    self.datamart.context_keys,
                ),
            )
            .filter(PredictorName="Classifier", ModelID=model_id)
        ).sort("BinIndex")

        if active_range:
            active_ranges = self.datamart.active_ranges(model_id).collect()
            if active_ranges.height > 0:
                active_range_info = active_ranges.to_dicts()[0]
                active_range_filter_expr = (pl.col("BinIndex") >= active_range_info["idx_min"]) & (
                    pl.col("BinIndex") <= active_range_info["idx_max"]
                )
                df = df.filter(active_range_filter_expr)

        if df.select(pl.first().len()).collect().item() == 0:
            raise ValueError(f"There is no data for the provided modelid '{model_id}'")

        if return_df:
            return df

        context = "/".join(
            df.select(
                pl.col("Configuration", *self.datamart.context_keys).fill_null(
                    "MISSING",
                ),
            )
            .unique()
            .collect()
            .row(0),
        )
        return distribution_graph(
            df,
            f"""Classifier score distribution<br>
            <sup>{context}</sup>
            """,
        )

    def multiple_score_distributions(
        self,
        query: QUERY | None = None,
        show_all: bool = True,
    ) -> list[Figure]:
        """Generate the score distribution plot for all models in the query

        Parameters
        ----------
        query : Optional[QUERY], optional
            A query to apply to the data, by default None
        show_all : bool, optional
            Whether to 'show' all plots or just get a list of them, by default True

        Returns
        -------
        list[go.Figure]
            A list of Plotly charts, one for each model instance

        Examples
        --------
        >>> # Generate and display score distributions for all models
        >>> figs = dm.plot.multiple_score_distributions()

        >>> # Collect figures without displaying them
        >>> figs = dm.plot.multiple_score_distributions(show_all=False)

        >>> # Limit to a specific channel before generating all distributions
        >>> figs = dm.plot.multiple_score_distributions(
        ...     query={"Channel": "Web"},
        ...     show_all=False,
        ... )

        """
        plots = []
        for model_id in (
            cdh_utils._apply_query(
                self.datamart.aggregates.last(table="combined_data"),
                query,
            )
            .select(pl.col("ModelID").unique())
            .collect()["ModelID"]
        ):
            fig = self.score_distribution(model_id=model_id)
            if show_all:
                fig.show()
            plots.append(fig)
        return plots
