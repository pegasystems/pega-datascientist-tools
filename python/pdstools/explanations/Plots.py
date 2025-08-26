__all__ = ["Plots"]

import logging
from typing import TYPE_CHECKING, List, Optional

import polars as pl

from ..utils.namespaces import LazyNamespace
from .ExplanationsUtils import _COL, _CONTRIBUTION_TYPE, _DEFAULT, _SPECIAL, ContextInfo

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
except ImportError as e:
    logger.debug("Failed to import optional dependencies: %s", e)

if TYPE_CHECKING:
    from .Explanations import Explanations


class Plots(LazyNamespace):
    dependencies = ["numpy", "plotly"]
    dependency_group = "explanations"

    X_AXIS_TITLE_DEFAULT = "Contribution"
    Y_AXIS_TITLE_DEFAULT = "Predictor"

    def __init__(self, explanations: "Explanations"):
        self.explanations = explanations
        self.aggregate = self.explanations.aggregate
        super().__init__()

    def contributions(
        self,
        top_n: int = _DEFAULT.TOP_N.value,
        top_k: int = _DEFAULT.TOP_K.value,
        descending: bool = _DEFAULT.DESCENDING.value,
        missing: bool = _DEFAULT.MISSING.value,
        remaining: bool = _DEFAULT.REMAINING.value,
        contribution_calculation: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value,
    ):
        """Plots contributions for the overall model or a selected context.
        Args:
            top_n (int):
                Number of top predictors to display.
            top_k (int):
                Number of top unique values for each categorical predictor to display.
            descending (bool):
                Whether to sort the predictors by most or least contribution.
            missing (bool):
                Whether to include missing values in the plot.
            remaining (bool):
                predictors/predictor values not included in the top_n/top_k
                will be grouped into a "remaining" category.
            contribution_calculation (str):
                Type of contribution calculation to use.
        Returns:
            tuple[go.Figure, List[go.Figure]]:
                - left: context header if context is selected, otherwise None
                - right: overall contributions plot and a list of predictor contribution plots.

        """
        contribution_type = _CONTRIBUTION_TYPE.validate_and_get_type(
            contribution_calculation
        )

        if self.explanations.filter.is_context_selected():
            context_plot, overall_plot, predictor_plots = (
                self.plot_contributions_by_context(
                    context=self.explanations.filter.get_selected_context(),
                    top_n=top_n,
                    top_k=top_k,
                    descending=descending,
                    missing=missing,
                    remaining=remaining,
                    contribution_calculation=contribution_type.value,
                )
            )

            plots = [overall_plot] + predictor_plots
            for plot in [context_plot] + plots:
                plot.show()

            return context_plot, plots

        else:
            print(
                "No context selected, plotting overall contributions. Use explanations.filter.interative() to select a context."
            )

            overall_plot, predictor_plots = self.plot_contributions_for_overall(
                top_n=top_n,
                top_k=top_k,
                descending=descending,
                missing=missing,
                remaining=remaining,
                contribution_calculation=contribution_type.value,
            )

            plots = [overall_plot] + predictor_plots
            for plot in plots:
                plot.show()

            return None, plots

    def plot_contributions_for_overall(
        self,
        top_n: int = _DEFAULT.TOP_N.value,
        top_k: int = _DEFAULT.TOP_K.value,
        descending: bool = _DEFAULT.DESCENDING.value,
        missing: bool = _DEFAULT.MISSING.value,
        remaining: bool = _DEFAULT.REMAINING.value,
        contribution_calculation: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value,
    ) -> tuple[go.Figure, List[go.Figure]]:
        contribution_type = _CONTRIBUTION_TYPE.validate_and_get_type(
            contribution_calculation
        )

        df = self.aggregate.get_predictor_contributions(
            top_n=top_n,
            descending=descending,
            missing=missing,
            remaining=remaining,
            contribution_calculation=contribution_calculation,
        )

        predictors = (
            df.filter(pl.col(_COL.PREDICTOR_NAME.value) != _SPECIAL.REMAINING.value)
            .select(_COL.PREDICTOR_NAME.value)
            .unique()
            .to_series()
            .to_list()
        )

        df_predictors = self.aggregate.get_predictor_value_contributions(
            predictors=predictors,
            top_k=top_k,
            descending=descending,
            missing=missing,
            remaining=remaining,
            contribution_calculation=contribution_calculation,
        )

        overall_fig = self._plot_overall_contributions(
            df,
            x_col=contribution_type.value,
            y_col=_COL.PREDICTOR_NAME.value,
            x_title=contribution_type.alt,
        )
        predictors_figs = self._plot_predictor_contributions(
            df_predictors,
            x_col=contribution_type.value,
            y_col=_COL.BIN_CONTENTS.value,
            x_title=contribution_type.alt,
        )

        return overall_fig, predictors_figs

    def plot_contributions_by_context(
        self,
        context: dict[str, str],
        top_n: int = _DEFAULT.TOP_N.value,
        top_k: int = _DEFAULT.TOP_K.value,
        descending: bool = _DEFAULT.DESCENDING.value,
        missing: bool = _DEFAULT.MISSING.value,
        remaining: bool = _DEFAULT.REMAINING.value,
        contribution_calculation: str = _CONTRIBUTION_TYPE.CONTRIBUTION.value,
    ) -> tuple[go.Figure, go.Figure, List[go.Figure]]:
        contribution_type = _CONTRIBUTION_TYPE.validate_and_get_type(
            contribution_calculation
        )

        df_context = self.aggregate.get_predictor_contributions(
            context,
            top_n,
            descending,
            missing,
            remaining,
            contribution_type.value,
        )

        # filter out the context rows for plotting by context
        contexts = list(context.keys())
        df_context = df_context.filter(
            ~pl.col(_COL.PREDICTOR_NAME.value).is_in(contexts)
        )

        predictors = (
            df_context.filter(
                pl.col(_COL.PREDICTOR_NAME.value) != _SPECIAL.REMAINING.value
            )
            .select(_COL.PREDICTOR_NAME.value)
            .unique()
            .to_series()
            .to_list()
        )

        df = self.aggregate.get_predictor_value_contributions(
            predictors,
            context=context,
            top_k=top_k,
            descending=descending,
            missing=missing,
            remaining=remaining,
            contribution_calculation=contribution_type.value,
        )

        header_fig = self._plot_context_table(context)

        overall_fig = self._plot_overall_contributions(
            df_context,
            x_col=contribution_type.value,
            y_col=_COL.PREDICTOR_NAME.value,
            x_title=contribution_type.alt,
            context=context,
        )

        predictors_figs = self._plot_predictor_contributions(
            df,
            x_col=contribution_type.value,
            y_col=_COL.BIN_CONTENTS.value,
            x_title=contribution_type.alt,
        )

        return header_fig, overall_fig, predictors_figs

    @staticmethod
    def _plot_overall_contributions(
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        x_title: str = X_AXIS_TITLE_DEFAULT,
        y_title: str = Y_AXIS_TITLE_DEFAULT,
        context: Optional[ContextInfo] = None,
    ) -> go.Figure:
        title = "Overall average predictor contributions for "
        if context is None:
            title += "the whole model"
        else:
            title += "-".join([f"{v}" for k, v in context.items()])

        fig = go.Figure(
            data=[
                go.Bar(
                    x=df[x_col].to_list(),
                    y=df[y_col].to_list(),
                    orientation="h",
                )
            ]
        )

        fig.update_layout(title=title)

        colors_values = df.select(pl.col(x_col)).to_series().to_list()

        fig.update_traces(
            marker=dict(
                color=colors_values,
                colorscale="RdBu_r",
                cmid=0.0,
            )
        )
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title, height=600)
        return fig

    @staticmethod
    def _plot_predictor_contributions(
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        x_title: str = X_AXIS_TITLE_DEFAULT,
        y_title: str = Y_AXIS_TITLE_DEFAULT,
    ) -> list[go.Figure]:
        predictors = df.select(_COL.PREDICTOR_NAME.value).unique().to_series().to_list()

        plots = []
        for predictor in predictors:
            predictor_df = df.filter(pl.col(_COL.PREDICTOR_NAME.value) == predictor)

            predictor_type = predictor_df.select(_COL.PREDICTOR_TYPE.value).to_series()[
                0
            ]
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=predictor_df[x_col].to_list(),
                        y=predictor_df[y_col].to_list(),
                        orientation="h",
                        customdata=[predictor_type],
                    )
                ]
            )

            colors_values = predictor_df.select(pl.col(x_col)).to_series().to_list()
            fig.update_traces(
                marker=dict(
                    color=colors_values,
                    colorscale="RdBu_r",
                    cmid=0.0,
                ),
                hovertemplate="Value: %{y}<br>PredictorType: %{customdata[0]}<extra></extra>",
            )
            fig.update_layout(
                xaxis_title=x_title,
                yaxis_title=predictor,
                title=predictor,
            )
            plots.append(fig)
        return plots

    @staticmethod
    def _plot_context_table(context_info: ContextInfo) -> go.Figure:
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=["Context key", "Context value"], align="left"),
                    cells=dict(
                        values=[list(context_info.keys()), list(context_info.values())],
                        align="left",
                        height=25,
                    ),
                )
            ]
        )
        fig.update_layout(
            title="Context Information", height=len(context_info) * 30 + 200
        )
        return fig
