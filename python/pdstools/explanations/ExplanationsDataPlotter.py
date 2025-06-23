__all__ = ["ExplanationsDataPlotter"]

import logging
from typing import List, Optional
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from .ExplanationsDataLoader import ExplanationsDataLoader as DataLoader
from .ExplanationsUtils import ContextInfo, _CONTRIBUTION_TYPE, _COL
logger = logging.getLogger(__name__)

class ExplanationsDataPlotter:
    
    # plot overall contribution for single context
    @classmethod
    def plot_overall_contributions(
        cls,
        data_loader: DataLoader,
        context: Optional[ContextInfo] = None,
        top_n: int = 20,
        top_k: int = 20,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_calculation: _CONTRIBUTION_TYPE = _CONTRIBUTION_TYPE.CONTRIBUTION,
    ) -> List[go.Figure]:
        
        context_list = [context] if context is not None else None
        
        # get data for single context with top_n predictors
        df = data_loader.get_predictor_contributions(
            context_list,
            None,
            top_n,
            descending,
            missing,
            remaining,
            contribution_calculation,
        )

        predictors = (
            df.filter(pl.col(_COL.PREDICTOR_NAME.value) != "REMAINING")
            .select(_COL.PREDICTOR_NAME.value)
            .unique()
            .to_series()
            .to_list()
        )

        # get data for single context with top_n predictors and top_k predictor values for each predictor
        df_predictors = data_loader.get_predictor_contributions(
            context_list,
            predictors,
            top_k,
            descending,
            missing,
            remaining,
            contribution_calculation,
        )
        
        return [
            cls._plot_overall_contributions(df, context), 
            *cls._plot_predictor_contributions(df_predictors)
        ]

    @staticmethod
    def _plot_overall_contributions(
        df: pl.DataFrame, 
        context: Optional[ContextInfo] = None
    ) -> go.Figure:
        title = "Overall average predictor contributions for "
        if context is None:
            title += "the whole model"
        else:
            title += "-".join([f'{v}' for k, v in context.items()])

        fig = px.bar(df, x=df.columns[0], y=df.columns[1], orientation="h", title=title)
        fig.data[0].marker = {
            "color": df[df.columns[0]].to_list(),
            "colorscale": "RdBu_r",
            "cmid": 0.0,
        }
        fig.update_layout(
            xaxis_title=df.columns[0],
            yaxis_title='Predictor',
            height=600
        )
        return fig

    @classmethod
    def plot_predictor_contributions(
        cls,
        data_loader: DataLoader,
        context: ContextInfo,
        top_n: int = 20,
        top_k: int = 20,
        descending: bool = True,
        missing: bool = True,
        remaining: bool = True,
        contribution_calculation: _CONTRIBUTION_TYPE = _CONTRIBUTION_TYPE.CONTRIBUTION,
    ) -> List[go.Figure]:
        df_context = data_loader.get_predictor_contributions(
            [context],
            [],
            top_n,
            descending,
            missing,
            remaining,
            contribution_calculation,
        )
        predictors = (
            df_context.filter(pl.col(_COL.PREDICTOR_NAME.value) != "REMAINING")
            .select(_COL.PREDICTOR_NAME.value)
            .unique()
            .to_series()
            .to_list()
        )
        df = data_loader.get_predictor_contributions(
            [context],
            predictors,
            top_k,
            descending,
            missing,
            remaining,
            contribution_calculation,
        )
        return [
            cls._plot_context_table(context),
            cls._plot_overall_contributions(df_context, context),
            *cls._plot_predictor_contributions(df),
        ]

    @staticmethod
    def _plot_predictor_contributions(df: pl.DataFrame) -> list[go.Figure]:
        predictors = df.select(_COL.PREDICTOR_NAME.value).unique().to_series().to_list()
        plots = []
        for predictor in predictors:
            predictor_df = df.filter(pl.col(_COL.PREDICTOR_NAME.value) == predictor)

            fig = px.bar(
                predictor_df,
                x=predictor_df.columns[0],
                y=predictor_df.columns[2],
                orientation="h",
                title=predictor,
            )
            fig.data[0].marker = {
                "color": predictor_df[predictor_df.columns[0]].to_list(),
                "colorscale": "RdBu_r",
                "cmid": 0.0,
            }
            fig.update_layout(
                yaxis_title=predictor,
            )
            plots.append(fig)
        return plots

    @staticmethod
    def _plot_context_table(context_info: ContextInfo) -> go.Figure:
        fig = go.Figure(
            go.Table(
                header=dict(values=["Context key", "Context value"], align="left"),
                cells=dict(
                    values=[list(context_info.keys()), list(context_info.values())],
                    align="left",
                ),
            )
        )
        fig.update_layout(title="Context Information", height=200)
        return fig

