from __future__ import annotations

__all__ = ["Plots"]

import logging
from typing import ClassVar, Literal, TYPE_CHECKING, overload

import polars as pl

from ..utils.namespaces import LazyNamespace
from ._constants import (
    CONTRIBUTION_LABELS,
    REMAINING,
    ContributionType,
    validate_contribution_type,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from .Explanations import Explanations


class Plots(LazyNamespace):
    """Plots."""

    dependencies: ClassVar[list[str]] = ["numpy", "plotly"]
    dependency_group = "explanations"

    def __init__(self, explanations: "Explanations"):
        self.explanations = explanations
        self.aggregates = self.explanations.aggregates
        super().__init__()

    @overload
    def contributions_overall(
        self,
        top_n: int = ...,
        top_k: int = ...,
        *,
        return_df: Literal[False] = ...,
        sort_by: ContributionType = ...,
        display_by: ContributionType = ...,
        descending: bool = ...,
        missing: bool = ...,
        remaining: bool = ...,
        include_numeric_single_bin: bool = ...,
    ) -> tuple[go.Figure, list[go.Figure]]: ...

    @overload
    def contributions_overall(
        self,
        top_n: int = ...,
        top_k: int = ...,
        *,
        return_df: Literal[True],
        sort_by: ContributionType = ...,
        display_by: ContributionType = ...,
        descending: bool = ...,
        missing: bool = ...,
        remaining: bool = ...,
        include_numeric_single_bin: bool = ...,
    ) -> tuple[pl.DataFrame, pl.DataFrame]: ...

    def contributions_overall(
        self,
        top_n: int = 20,
        top_k: int = 20,
        *,
        return_df: bool = False,
        sort_by: ContributionType = "contribution_abs",
        display_by: ContributionType = "contribution",
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        include_numeric_single_bin: bool = False,
    ) -> tuple[go.Figure, list[go.Figure]] | tuple[pl.DataFrame, pl.DataFrame]:
        """Plot contributions for overall."""
        display_by = validate_contribution_type(display_by)
        display_by_label = CONTRIBUTION_LABELS[display_by][0]
        df = self.aggregates.predictor_contributions(
            top_n=top_n,
            sort_by=sort_by,
            descending=descending,
            missing=missing,
            remaining=remaining,
            include_numeric_single_bin=include_numeric_single_bin,
        )

        predictors = (
            df.filter(pl.col("predictor_name") != REMAINING)
            .select("predictor_name")
            .unique(maintain_order=True)
            .to_series()
            .to_list()
        )

        df_predictors = self.aggregates.predictor_value_contributions(
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

        overall_fig = self._overall_figure(
            df,
            x_col=display_by,
            y_col="predictor_name",
            x_title=display_by_label,
        )
        predictors_figs = self._predictor_figures(
            df_predictors,
            x_col=display_by,
            y_col="bin_contents",
            x_title=display_by_label,
        )

        return overall_fig, predictors_figs

    @overload
    def contributions_by_context(
        self,
        context: dict[str, str],
        top_n: int = ...,
        top_k: int = ...,
        *,
        return_df: Literal[False] = ...,
        sort_by: ContributionType = ...,
        display_by: ContributionType = ...,
        descending: bool = ...,
        missing: bool = ...,
        remaining: bool = ...,
        include_numeric_single_bin: bool = ...,
    ) -> tuple[go.Figure, go.Figure, list[go.Figure]]: ...

    @overload
    def contributions_by_context(
        self,
        context: dict[str, str],
        top_n: int = ...,
        top_k: int = ...,
        *,
        return_df: Literal[True],
        sort_by: ContributionType = ...,
        display_by: ContributionType = ...,
        descending: bool = ...,
        missing: bool = ...,
        remaining: bool = ...,
        include_numeric_single_bin: bool = ...,
    ) -> tuple[pl.DataFrame, pl.DataFrame]: ...

    def contributions_by_context(
        self,
        context: dict[str, str],
        top_n: int = 20,
        top_k: int = 20,
        *,
        return_df: bool = False,
        sort_by: ContributionType = "contribution_abs",
        display_by: ContributionType = "contribution",
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        include_numeric_single_bin: bool = False,
    ) -> tuple[go.Figure, go.Figure, list[go.Figure]] | tuple[pl.DataFrame, pl.DataFrame]:
        """Plot contributions by context."""
        display_by = validate_contribution_type(display_by)
        display_by_label = CONTRIBUTION_LABELS[display_by][0]
        df_context = self.aggregates.predictor_contributions(
            context,
            top_n=top_n,
            sort_by=sort_by,
            descending=descending,
            missing=missing,
            remaining=remaining,
            include_numeric_single_bin=include_numeric_single_bin,
        )

        # filter out the context rows for plotting by context
        # is_in yields null for null predictor_name, and ~null is null, which
        # filter drops — fill_null keeps those rows instead.
        df_context = df_context.filter(
            ~pl.col("predictor_name").is_in(list(context.keys())).fill_null(False),
        )

        predictors = (
            df_context.filter(
                pl.col("predictor_name") != REMAINING,
            )
            .select("predictor_name")
            .unique(maintain_order=True)
            .to_series()
            .to_list()
        )

        df = self.aggregates.predictor_value_contributions(
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

        header_fig = self._context_table_figure(context)

        overall_fig = self._overall_figure(
            df_context,
            x_col=display_by,
            y_col="predictor_name",
            x_title=display_by_label,
            context=context,
        )

        predictors_figs = self._predictor_figures(
            df,
            x_col=display_by,
            y_col="bin_contents",
            x_title=display_by_label,
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
            "predictor_name",
            "predictor_type",
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

    def _overall_figure(
        self,
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        x_title: str,
        y_title: str = "Predictor",
        context: dict[str, str] | None = None,
    ) -> go.Figure:
        import plotly.graph_objects as go

        title = "Overall average predictor contributions for "
        if context is None:
            title += "the whole model"
            customdata, hovertemplate = self._build_hover_customdata(df, x_col, include_frequency=False)
        else:
            title += "-".join([f"{v}" for k, v in context.items()])
            # Show each predictor's context frequency as a share of the overall model.
            df_with_pct = self.aggregates._add_context_frequency_pct(
                df,
                join_on=["predictor_name", "predictor_type"],
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

    def _predictor_figures(
        self,
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        x_title: str,
        y_title: str = "Predictor",
    ) -> list[go.Figure]:
        import plotly.graph_objects as go

        df_with_frequency_pct = self.aggregates._add_frequency_pct(
            df, group_by=["context_partition", "predictor_name", "predictor_type"]
        )

        predictor_info = df.select(["predictor_name", "predictor_type"]).unique(maintain_order=True)

        plots = []
        for predictor, predictor_type in predictor_info.iter_rows():
            predictor_df = df_with_frequency_pct.filter(pl.col("predictor_name") == predictor)

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
    def _context_table_figure(context_info: dict[str, str]) -> go.Figure:
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
