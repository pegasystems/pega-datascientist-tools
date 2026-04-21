from __future__ import annotations

__all__ = ["Plots"]

import logging
from typing import TYPE_CHECKING, Literal, cast, overload

import polars as pl

from ..utils.namespaces import LazyNamespace
from .ExplanationsUtils import (
    _COL,
    _SPECIAL,
    ContextInfo,
    DisplayBy,
    SortBy,
    _resolve_contribution_type,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import plotly.graph_objects as go

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
        top_n: int = 20,
        top_k: int = 20,
        *,
        return_df: bool = False,
        sort_by: SortBy = "contribution_abs",
        display_by: DisplayBy = "contribution",
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        include_numeric_single_bin: bool = False,
    ):
        """Plots contributions for the overall model or a selected context.

        Args:
            top_n (int):
                Number of top predictors to display.
            top_k (int):
                Number of top unique values for each categorical predictor to display.
            return_df (bool, keyword-only):
                If True, skip plotting and return the underlying dataframes instead.
                When a context is selected, returns
                ``(predictor_df, predictor_value_df)``; otherwise returns the same
                pair computed against the overall model.
            sort_by (str, keyword-only):
                Column to rank/select top predictors. One of
                ``contribution``, ``contribution_abs``,
                ``contribution_weighted``, ``contribution_weighted_abs``.
                Default: ``"contribution_abs"``.
            display_by (str, keyword-only):
                Column to use for the chart axis values.
                Default: ``"contribution"``.
            descending (bool, keyword-only):
                Sort most- or least-impactful first. Default: ``True``.
            missing (bool, keyword-only):
                Include missing-value bins. Default: ``True``.
            remaining (bool, keyword-only):
                Include an aggregated "remaining" row. Default: ``True``.
            include_numeric_single_bin (bool, keyword-only):
                Include numeric predictors that have only a single bin.
                Default: ``False``.

        Returns:
            tuple[go.Figure, list[go.Figure]]:
                - left: context header if context is selected, otherwise None
                - right: overall contributions plot and a list of predictor contribution plots.

        """
        common_kwargs = {
            "sort_by": sort_by,
            "display_by": display_by,
            "descending": descending,
            "missing": missing,
            "remaining": remaining,
            "include_numeric_single_bin": include_numeric_single_bin,
        }
        if self.explanations.filter.is_context_selected():
            if return_df:
                return self.plot_contributions_by_context(
                    context=self.explanations.filter.get_selected_context(),
                    top_n=top_n,
                    top_k=top_k,
                    return_df=True,
                    **common_kwargs,
                )
            context_plot, overall_plot, predictor_plots = self.plot_contributions_by_context(
                context=self.explanations.filter.get_selected_context(),
                top_n=top_n,
                top_k=top_k,
                **common_kwargs,
            )

            plots = [overall_plot] + predictor_plots
            for plot in [context_plot] + plots:
                plot.show()

            return context_plot, plots

        if return_df:
            return self.plot_contributions_for_overall(
                top_n=top_n,
                top_k=top_k,
                return_df=True,
                **common_kwargs,
            )

        logger.info(
            "No context selected, plotting overall contributions. "
            "Use explanations.filter.interactive() to select a context.",
        )

        overall_plot, predictor_plots = self.plot_contributions_for_overall(
            top_n=top_n,
            top_k=top_k,
            **common_kwargs,
        )

        plots = [overall_plot] + predictor_plots
        for plot in plots:
            plot.show()

        return None, plots

    @overload
    def plot_contributions_for_overall(
        self,
        top_n: int = ...,
        top_k: int = ...,
        *,
        return_df: Literal[False] = ...,
        sort_by: SortBy = ...,
        display_by: DisplayBy = ...,
        descending: bool = ...,
        missing: bool = ...,
        remaining: bool = ...,
        include_numeric_single_bin: bool = ...,
    ) -> tuple[go.Figure, list[go.Figure]]: ...

    @overload
    def plot_contributions_for_overall(
        self,
        top_n: int = ...,
        top_k: int = ...,
        *,
        return_df: Literal[True],
        sort_by: SortBy = ...,
        display_by: DisplayBy = ...,
        descending: bool = ...,
        missing: bool = ...,
        remaining: bool = ...,
        include_numeric_single_bin: bool = ...,
    ) -> tuple[pl.DataFrame, pl.DataFrame]: ...

    def plot_contributions_for_overall(
        self,
        top_n: int = 20,
        top_k: int = 20,
        *,
        return_df: bool = False,
        sort_by: SortBy = "contribution_abs",
        display_by: DisplayBy = "contribution",
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        include_numeric_single_bin: bool = False,
    ) -> tuple[go.Figure, list[go.Figure]] | tuple[pl.DataFrame, pl.DataFrame]:
        display_by_enum = _resolve_contribution_type(display_by)
        agg_kwargs = {
            "sort_by": sort_by,
            "descending": descending,
            "missing": missing,
            "remaining": remaining,
            "include_numeric_single_bin": include_numeric_single_bin,
        }

        df = self.aggregate.get_predictor_contributions(
            top_n=top_n,
            **agg_kwargs,
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
            **agg_kwargs,
        )

        if return_df:
            return df, df_predictors

        overall_fig = self._plot_overall_contributions(
            df,
            x_col=display_by_enum.value,
            y_col=_COL.PREDICTOR_NAME.value,
            x_title=display_by_enum.alt,
        )
        predictors_figs = self._plot_predictor_contributions(
            df_predictors,
            x_col=display_by_enum.value,
            y_col=_COL.BIN_CONTENTS.value,
            x_title=display_by_enum.alt,
        )

        return overall_fig, predictors_figs

    @overload
    def plot_contributions_by_context(
        self,
        context: dict[str, str],
        top_n: int = ...,
        top_k: int = ...,
        *,
        return_df: Literal[False] = ...,
        sort_by: SortBy = ...,
        display_by: DisplayBy = ...,
        descending: bool = ...,
        missing: bool = ...,
        remaining: bool = ...,
        include_numeric_single_bin: bool = ...,
    ) -> tuple[go.Figure, go.Figure, list[go.Figure]]: ...

    @overload
    def plot_contributions_by_context(
        self,
        context: dict[str, str],
        top_n: int = ...,
        top_k: int = ...,
        *,
        return_df: Literal[True],
        sort_by: SortBy = ...,
        display_by: DisplayBy = ...,
        descending: bool = ...,
        missing: bool = ...,
        remaining: bool = ...,
        include_numeric_single_bin: bool = ...,
    ) -> tuple[pl.DataFrame, pl.DataFrame]: ...

    def plot_contributions_by_context(
        self,
        context: dict[str, str],
        top_n: int = 20,
        top_k: int = 20,
        *,
        return_df: bool = False,
        sort_by: SortBy = "contribution_abs",
        display_by: DisplayBy = "contribution",
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        include_numeric_single_bin: bool = False,
    ) -> tuple[go.Figure, go.Figure, list[go.Figure]] | tuple[pl.DataFrame, pl.DataFrame]:
        display_by_enum = _resolve_contribution_type(display_by)
        agg_kwargs = {
            "sort_by": sort_by,
            "descending": descending,
            "missing": missing,
            "remaining": remaining,
            "include_numeric_single_bin": include_numeric_single_bin,
        }

        df_context = self.aggregate.get_predictor_contributions(
            context,
            top_n=top_n,
            **agg_kwargs,
        )

        # filter out the context rows for plotting by context
        contexts = list(context.keys())
        df_context = df_context.filter(
            ~pl.col(_COL.PREDICTOR_NAME.value).is_in(contexts),
        )

        predictors = (
            df_context.filter(
                pl.col(_COL.PREDICTOR_NAME.value) != _SPECIAL.REMAINING.value,
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
            **agg_kwargs,
        )

        if return_df:
            return df_context, df

        header_fig = self._plot_context_table(cast("ContextInfo", context))

        overall_fig = self._plot_overall_contributions(
            df_context,
            x_col=display_by_enum.value,
            y_col=_COL.PREDICTOR_NAME.value,
            x_title=display_by_enum.alt,
            context=cast("ContextInfo", context),
        )

        predictors_figs = self._plot_predictor_contributions(
            df,
            x_col=display_by_enum.value,
            y_col=_COL.BIN_CONTENTS.value,
            x_title=display_by_enum.alt,
        )

        return header_fig, overall_fig, predictors_figs

    @staticmethod
    def _build_hover_customdata(
        df: pl.DataFrame,
        x_col: str,
    ):
        """Build customdata array and hovertemplate for contribution plots.

        Expects df to contain a ``frequency_pct`` column.

        Returns (customdata, hovertemplate) with columns:
        predictor_name, predictor_type, contribution (x_col value), frequency_pct.
        """
        customdata = df.select(
            _COL.PREDICTOR_NAME.value,
            _COL.PREDICTOR_TYPE.value,
            pl.col(x_col).alias("contribution"),
            "frequency_pct",
        ).to_numpy()

        hovertemplate = (
            "predictor_name: %{customdata[0]}<br>"
            "predictor_type: %{customdata[1]}<br>"
            "contribution: %{customdata[2]:.8f}<br>"
            "frequency: %{customdata[3]}%"
            "<extra></extra>"
        )
        return customdata, hovertemplate

    def _plot_overall_contributions(
        self,
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        x_title: str = X_AXIS_TITLE_DEFAULT,
        y_title: str = Y_AXIS_TITLE_DEFAULT,
        context: ContextInfo | None = None,
    ) -> go.Figure:
        import plotly.graph_objects as go

        title = "Overall average predictor contributions for "
        if context is None:
            title += "the whole model"
        else:
            title += "-".join([f"{v}" for k, v in context.items()])

        df_with_pct = self.aggregate.add_frequency_pct_to_df(df, group_by=[_COL.PARTITON.value])
        customdata, hovertemplate = self._build_hover_customdata(df_with_pct, x_col)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=df[x_col].to_list(),
                    y=df[y_col].to_list(),
                    orientation="h",
                    customdata=customdata,
                ),
            ],
        )

        fig.update_layout(title=title)

        colors_values = df.select(pl.col(x_col)).to_series().to_list()

        fig.update_traces(
            marker=dict(
                color=colors_values,
                colorscale="RdBu_r",
                cmid=0.0,
            ),
            hovertemplate=hovertemplate,
        )
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title, height=600)
        return fig

    def _plot_predictor_contributions(
        self,
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        x_title: str = X_AXIS_TITLE_DEFAULT,
        y_title: str = Y_AXIS_TITLE_DEFAULT,
    ) -> list[go.Figure]:
        import plotly.graph_objects as go

        df_with_frequency_pct = self.aggregate.add_frequency_pct_to_df(
            df, group_by=[_COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value]
        )

        predictor_info = df.select([_COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value]).unique()

        plots = []
        for predictor, predictor_type in predictor_info.iter_rows():
            predictor_df = df_with_frequency_pct.filter(pl.col(_COL.PREDICTOR_NAME.value) == predictor)

            customdata, hovertemplate = self._build_hover_customdata(predictor_df, x_col)

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=predictor_df[x_col].to_list(),
                        y=predictor_df[y_col].to_list(),
                        orientation="h",
                        customdata=customdata,
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
                hovertemplate=hovertemplate,
            )
            fig.update_layout(
                xaxis_title=x_title,
                yaxis_title=predictor,
                title=f"{predictor}<br><sup><span style='color:gray'>{predictor_type}</span></sup>",
            )
            plots.append(fig)
        return plots

    @staticmethod
    def _plot_context_table(context_info: ContextInfo) -> go.Figure:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["Model context key", "Model context value"],
                        align="left",
                    ),
                    cells=dict(
                        values=[list(context_info.keys()), list(context_info.values())],
                        align="left",
                        height=25,
                    ),
                ),
            ],
        )
        fig.update_layout(
            title="Model Context Information",
            height=len(context_info) * 30 + 200,
        )
        return fig
