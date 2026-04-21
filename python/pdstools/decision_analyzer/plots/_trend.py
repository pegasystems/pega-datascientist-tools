"""Trend plots."""

import plotly.express as px
import plotly.graph_objects as go
import polars as pl


def trend_chart(self, stage: str, scope: str, return_df=False, additional_filters=None) -> tuple[go.Figure, str | None]:
    df = self._decision_data.get_trend_data(stage, scope, additional_filters=additional_filters).collect()

    if return_df:
        return df.lazy()

    warning_message = None
    if df.select(pl.col("day").n_unique()).get_column("day")[0] <= 1:
        warning_message = (
            "Insufficient data: Trend analysis requires data from multiple days. "
            "Currently, the dataset contains information for only one day."
        )

    color_discrete_map = self._decision_data.color_mappings.get(scope)

    fig = px.area(
        data_frame=df,
        x="day",
        y="Decisions",
        color=scope,
        color_discrete_map=color_discrete_map,
        template="pega",
    )

    fig.update_layout(xaxis_title="")

    return fig, warning_message
