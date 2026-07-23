from __future__ import annotations

__all__ = ["Plots"]

import logging
from typing import ClassVar, Literal, TYPE_CHECKING, cast, overload

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
    """Plots."""

    dependencies: ClassVar[list[str]] = ["numpy", "plotly"]
    dependency_group = "explanations"

    X_AXIS_TITLE_DEFAULT = "Contribution"
    Y_AXIS_TITLE_DEFAULT = "Predictor"

    def __init__(self, explanations: "Explanations"):
        self.explanations = explanations
        self.aggregate = self.explanations.aggregate
        super().__init__()

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
        """Plot contributions for overall."""
        display_by_enum = _resolve_contribution_type(display_by)
        df = self.aggregate.get_predictor_contributions(
            top_n=top_n,
            sort_by=sort_by,
            descending=descending,
            missing=missing,
            remaining=remaining,
            include_numeric_single_bin=include_numeric_single_bin,
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
            sort_by=sort_by,
            descending=descending,
            missing=missing,
            remaining=remaining,
            include_numeric_single_bin=include_numeric_single_bin,
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
        """Plot contributions by context."""
        display_by_enum = _resolve_contribution_type(display_by)
        df_context = self.aggregate.get_predictor_contributions(
            context,
            top_n=top_n,
            sort_by=sort_by,
            descending=descending,
            missing=missing,
            remaining=remaining,
            include_numeric_single_bin=include_numeric_single_bin,
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
            sort_by=sort_by,
            descending=descending,
            missing=missing,
            remaining=remaining,
            include_numeric_single_bin=include_numeric_single_bin,
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
        include_frequency: bool = True,
    ):
        """Build customdata array and hovertemplate for contribution plots.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame. Must contain a ``frequency_pct`` column when
            ``include_frequency=True``.
        x_col : str
            Column used as the contribution value.
        include_frequency : bool, default True
            When False, omits the frequency row from the hover tooltip
            (e.g. for the whole-model view where it is always 100%).

        Returns
        -------
        tuple[numpy.ndarray, str]
            Tuple of (customdata, hovertemplate).
        """
        select_cols = [
            _COL.PREDICTOR_NAME.value,
            _COL.PREDICTOR_TYPE.value,
            pl.col(x_col).alias("contribution"),
        ]
        if include_frequency:
            select_cols.append(pl.col("frequency_pct"))

        customdata = df.select(select_cols).to_numpy()

        hovertemplate = (
            "predictor_name: %{customdata[0]}<br>predictor_type: %{customdata[1]}<br>contribution: %{customdata[2]:.8f}"
        )
        if include_frequency:
            hovertemplate += "<br>frequency: %{customdata[3]}%"
        hovertemplate += "<extra></extra>"

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
            customdata, hovertemplate = self._build_hover_customdata(df, x_col, include_frequency=False)
        else:
            title += "-".join([f"{v}" for k, v in context.items()])
            # Show each predictor's context frequency as a share of the overall model.
            df_with_pct = self.aggregate.add_context_frequency_pct_to_df(
                df,
                join_on=[_COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value],
            )
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
            df, group_by=[_COL.PARTITION.value, _COL.PREDICTOR_NAME.value, _COL.PREDICTOR_TYPE.value]
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
