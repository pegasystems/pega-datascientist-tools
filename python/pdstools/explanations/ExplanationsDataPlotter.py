__all__ = ["ExplanationsDataPlotter"]

import logging
from typing import List, Optional
import plotly.graph_objects as go
import polars as pl

from .ExplanationsDataLoader import ExplanationsDataLoader as DataLoader
from .ExplanationsUtils import ContextInfo, _CONTRIBUTION_TYPE, _COL, _SPECIAL
logger = logging.getLogger(__name__)

X_AXIS_TITLE_DEFAULT = "Contribution"
Y_AXIS_TITLE_DEFAULT = "Predictor"
class ExplanationsDataPlotter:
    
    # plot overall contribution for single context
    @classmethod
    def plot_contributions_for_overall(
        cls,
        data_loader: DataLoader,
        top_n: int = 10,
        top_k: int = 10,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_calculation: _CONTRIBUTION_TYPE = _CONTRIBUTION_TYPE.CONTRIBUTION,
    ) -> tuple[go.Figure, List[go.Figure]]:
        
        df = data_loader.get_top_n_predictor_contribution_overall(
            top_n = top_n,
            descending=descending,
            missing=missing,
            remaining=remaining,
            contribution_type=contribution_calculation,
        )

        predictors = (
            df.filter(pl.col(_COL.PREDICTOR_NAME.value) != _SPECIAL.REMAINING.value)
            .select(_COL.PREDICTOR_NAME.value)
            .unique()
            .to_series()
            .to_list()
        )

        df_predictors = data_loader.get_top_k_predictor_value_contribution_overall(
            predictors=predictors,
            top_k=top_k,
            descending=descending,
            missing=missing,
            remaining=remaining,
            contribution_type=contribution_calculation,
        )
        
        overall_fig = cls._plot_overall_contributions(df, 
                                            x_col=contribution_calculation.value, 
                                            y_col=_COL.PREDICTOR_NAME.value)
        predictors_figs = cls._plot_predictor_contributions(df_predictors,
                                               x_col=contribution_calculation.value, 
                                               y_col=_COL.BIN_CONTENTS.value)
        
        return overall_fig, predictors_figs

    @classmethod
    def plot_contributions_by_contexts_list(cls,
        data_loader: DataLoader,
        contexts: List[ContextInfo],
        top_n: int = 10,
        top_k: int = 10,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_type: _CONTRIBUTION_TYPE = _CONTRIBUTION_TYPE.CONTRIBUTION,):
        pass
        
    @classmethod
    def plot_contributions_by_context(
        cls,
        data_loader: DataLoader,
        context: ContextInfo,
        top_n: int = 10,
        top_k: int = 10,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_type: _CONTRIBUTION_TYPE = _CONTRIBUTION_TYPE.CONTRIBUTION,
    ) -> tuple[go.Figure, go.Figure, List[go.Figure]]:     
                
        df_context = data_loader.get_top_n_predictor_contribution_by_context(
            context,
            top_n,
            descending,
            missing,
            remaining,
            contribution_type,
        )
        
        predictors = (
            df_context.filter(pl.col(_COL.PREDICTOR_NAME.value) != _SPECIAL.REMAINING.value)
            .select(_COL.PREDICTOR_NAME.value)
            .unique()
            .to_series()
            .to_list()
        )
        
        df = data_loader.get_top_k_predictor_value_contribution_by_context(
            context,
            predictors,
            top_k,
            descending,
            missing,
            remaining,
            contribution_type,
        )
        
        header_fig = cls._plot_context_table(context)
        
        overall_fig = cls._plot_overall_contributions(df_context,
                                            x_col=contribution_type.value,
                                            y_col=_COL.PREDICTOR_NAME.value,
                                            context=context)
        
        predictors_figs = cls._plot_predictor_contributions(df,
                                               x_col=contribution_type.value, 
                                               y_col=_COL.BIN_CONTENTS.value)
        
        return header_fig, overall_fig, predictors_figs
        
    @staticmethod
    def _plot_overall_contributions(
        df: pl.DataFrame, 
        x_col: str,
        y_col: str,
        x_title: str = X_AXIS_TITLE_DEFAULT,
        y_title: str = Y_AXIS_TITLE_DEFAULT,
        context: Optional[ContextInfo] = None
    ) -> go.Figure:
        
        title = "Overall average predictor contributions for "
        if context is None:
            title += "the whole model"
        else:
            title += "-".join([f'{v}' for k, v in context.items()])

        fig = go.Figure(
            data=[go.Bar(
                x=df[x_col].to_list(),
                y=df[y_col].to_list(),
                orientation="h",
                )]
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
        fig.update_layout(
            xaxis_title=x_title,
            yaxis_title=y_title,
            height=600
        )
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

            fig = go.Figure(data=[go.Bar(
                x=predictor_df[x_col].to_list(),
                y=predictor_df[y_col].to_list(),
                orientation="h",
                )
            ])

            colors_values = predictor_df.select(pl.col(x_col)).to_series().to_list()
            fig.update_traces(
                marker=dict(
                    color=colors_values,
                    colorscale="RdBu_r",
                    cmid=0.0,
                )
            )
            fig.update_layout(
                yaxis_title=predictor,
                title=predictor,
            )
            plots.append(fig)
        return plots

    @staticmethod
    def _plot_context_table(context_info: ContextInfo) -> go.Figure:
        fig = go.Figure(data=[
            go.Table(
                header=dict(values=['Context key', 'Context value'], align='left'),
                cells=dict(values=[list(context_info.keys()), list(context_info.values())], align='left', height = 25)
            )
        ])
        fig.update_layout(title="Context Information", height=context_info.__len__() * 30 + 200)
        return fig

