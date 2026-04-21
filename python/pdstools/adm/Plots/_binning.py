"""Predictor binning visualisations and lift charts."""

from __future__ import annotations

import polars as pl

from ...utils import cdh_utils
from ...utils.plot_utils import (
    LIFT_DIRECTION_COLORS,
    Figure,
    abbreviate_label_expr,
    simplify_facet_titles,
)
from ...utils.types import QUERY
from ._base import _PlotsBase
from ._helpers import distribution_graph, requires


class _BinningPlotsMixin(_PlotsBase):
    @requires(
        combined_columns={
            "PredictorName",
            "Name",
            "ModelID",
            "Configuration",
            "BinIndex",
            "BinSymbol",
            "BinResponseCount",
            "BinPropensity",
        },
    )
    def predictor_binning(
        self,
        model_id: str,
        predictor_name: str,
        return_df: bool = False,
    ):
        """Generate a predictor binning plot for a specific model and predictor.

        Parameters
        ----------
        model_id : str
            The ID of the model containing the predictor
        predictor_name : str
            Name of the predictor to analyze
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Union[Figure, pl.LazyFrame]
            Plotly figure showing predictor binning or DataFrame if return_df=True

        Raises
        ------
        ValueError
            If no data is available for the provided model ID and predictor name

        Examples
        --------
        >>> # Predictor binning for a single predictor on a model
        >>> fig = dm.plot.predictor_binning(
        ...     model_id="M-1001",
        ...     predictor_name="Customer.Age",
        ... )

        >>> # Retrieve the underlying binning data
        >>> df = dm.plot.predictor_binning(
        ...     model_id="M-1001",
        ...     predictor_name="Customer.Age",
        ...     return_df=True,
        ... )

        """
        df = (
            self.datamart.aggregates.last(table="combined_data")
            .select(
                {
                    "PredictorName",
                    "Name",
                    "ModelID",
                    "Configuration",
                    "BinIndex",
                    "BinSymbol",
                    "BinResponseCount",
                    "BinPropensity",
                }
                | set(
                    self.datamart.context_keys,
                ),
            )
            .filter(
                pl.col("PredictorName") == predictor_name,
                pl.col("ModelID") == model_id,
            )
        ).sort("BinIndex")

        if df.select(pl.first().len()).collect().item() == 0:
            raise ValueError(
                f"There is no data for the provided modelid {model_id} and predictor {predictor_name}",
            )

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
            title=f"""Predictor binning for {predictor_name}<br><sup>{context}""",
        )

    def multiple_predictor_binning(
        self,
        model_id: str,
        query: QUERY | None = None,
        show_all=True,
    ) -> list[Figure]:
        """Generate predictor binning plots for all predictors in a model.

        Parameters
        ----------
        model_id : str
            The ID of the model to generate predictor binning plots for
        query : Optional[QUERY], optional
            A query to apply to the predictor data, by default None
        show_all : bool, optional
            Whether to display all plots or just return the list, by default True

        Returns
        -------
        list[Figure]
            A list of Plotly figures, one for each predictor in the model

        Examples
        --------
        >>> # Plot binning for every predictor in a model
        >>> figs = dm.plot.multiple_predictor_binning(model_id="M-1001")

        >>> # Collect figures without displaying, filtered to IH predictors
        >>> figs = dm.plot.multiple_predictor_binning(
        ...     model_id="M-1001",
        ...     query={"PredictorCategory": "IH"},
        ...     show_all=False,
        ... )

        """
        plots = []
        for predictor in (
            cdh_utils._apply_query(
                self.datamart.aggregates.last(table="predictor_data"),
                query,
            )
            .filter(pl.col("ModelID") == model_id)
            .select(pl.col("PredictorName").unique())
            .sort("PredictorName")
            .collect()["PredictorName"]
        ):
            fig = self.predictor_binning(model_id=model_id, predictor_name=predictor)
            if show_all:
                fig.show()
            plots.append(fig)
        return plots

    def binning_lift(
        self,
        model_id: str,
        predictor_name: str,
        *,
        query: QUERY | None = None,
        return_df: bool = False,
    ):
        """Generate a binning lift plot for a specific predictor showing propensity lift per bin.

        Parameters
        ----------
        model_id : str
            The ID of the model containing the predictor
        predictor_name : str
            Name of the predictor to analyze for lift
        query : Optional[QUERY], optional
            Optional query to filter the predictor data, by default None
        return_df : bool, optional
            Whether to return a dataframe instead of a plot, by default False

        Returns
        -------
        Union[Figure, pl.LazyFrame]
            Plotly bar chart showing binning lift or DataFrame if return_df=True

        Examples
        --------
        >>> # Lift chart for a single predictor on a model
        >>> fig = dm.plot.binning_lift(
        ...     model_id="M-1001",
        ...     predictor_name="Customer.Age",
        ... )

        >>> # Return the lift data for further analysis
        >>> df = dm.plot.binning_lift(
        ...     model_id="M-1001",
        ...     predictor_name="Customer.Age",
        ...     return_df=True,
        ... )

        """
        df = cdh_utils._apply_query(
            (
                self.datamart.aggregates.last(table="predictor_data")
                .filter(
                    pl.col("PredictorName") == predictor_name,
                    pl.col("ModelID") == model_id,
                )
                .sort("BinIndex")
            ),
            query,
        ).select(
            "PredictorName",
            "BinIndex",
            "BinPositives",
            "BinNegatives",
            "BinSymbol",
        )
        cols = df.collect_schema().names()

        if "Lift" not in cols:
            df = df.with_columns(
                (cdh_utils.lift(pl.col("BinPositives"), pl.col("BinNegatives")) - 1.0).alias("Lift"),
            )

        shading_expr = pl.col("BinPositives") <= 5 if "BinPositives" in cols else pl.lit(False)

        plot_df = df.with_columns(
            pl.when((pl.col("Lift") >= 0.0) & shading_expr.not_())
            .then(pl.lit("pos"))
            .when((pl.col("Lift") >= 0.0) & shading_expr)
            .then(pl.lit("pos_shaded"))
            .when((pl.col("Lift") < 0.0) & shading_expr.not_())
            .then(pl.lit("neg"))
            .otherwise(pl.lit("neg_shaded"))
            .alias("Direction"),
            BinSymbolAbbreviated=abbreviate_label_expr("BinSymbol"),
        ).sort(["PredictorName", "BinIndex"])

        if return_df:
            return plot_df

        import plotly.express as px

        fig = px.bar(
            plot_df.collect(),
            x="Lift",
            y="BinSymbolAbbreviated",
            color="Direction",
            color_discrete_map=LIFT_DIRECTION_COLORS,
            orientation="h",
            title=f"Propensity Lift for {predictor_name}",
            template="pega",
            custom_data=["PredictorName", "BinSymbol"],
            facet_col_wrap=3,
            category_orders=plot_df.collect().to_dict(),
        )
        fig.update_traces(
            hovertemplate="<br>".join(
                [
                    "<b>%{customdata[0]}</b>",
                    "%{customdata[1]}",
                    "<b>Lift: %{x:.2%}</b>",
                ],
            ),
        )
        fig.add_vline(x=0, line_color="black")

        fig.update_layout(
            showlegend=False,
            hovermode="y",
        )
        fig.update_xaxes(title="", tickformat=",.2%")
        fig.update_yaxes(
            type="category",
            categoryorder="array",
            automargin=True,
            title="",
            dtick=1,  # show all bins
            matches=None,  # allow independent y-labels if there are row facets
        )
        simplify_facet_titles(fig)
        return fig
