"""Plot sub-namespace: Plotly figures for topic data quality."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Literal, overload

import polars as pl

from pdstools.utils.namespaces import LazyNamespace

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

    from ._topic_data_quality import TopicDataQuality

logger = logging.getLogger(__name__)


class Plot(LazyNamespace):
    """Plotly visualizations for topic data quality.

    Attached to :pyattr:`TopicDataQuality.plot`.
    """

    dependencies: ClassVar[list[str]] = ["plotly"]
    dependency_group = "nlp"

    def __init__(self, parent: TopicDataQuality) -> None:
        super().__init__()
        self.parent = parent

    # ------------------------------------------------------------------
    # Topic distribution
    # ------------------------------------------------------------------

    @overload
    def topic_distribution(self, *, return_df: Literal[False] = ...) -> Figure: ...

    @overload
    def topic_distribution(self, *, return_df: Literal[True]) -> pl.DataFrame: ...

    def topic_distribution(self, *, return_df: bool = False) -> Figure | pl.DataFrame:
        """Bar chart of topic percentage distribution.

        Parameters
        ----------
        return_df : bool, default False
            If ``True``, return the underlying DataFrame instead of a figure.

        Returns
        -------
        Figure or pl.DataFrame
        """
        counts = self.parent.topic_counts
        total = counts.get_column("count").sum()
        dist = counts.with_columns(
            (pl.col("count") / total * 100).round(1).alias("percent"),
        )

        if return_df:
            return dist

        import plotly.express as px

        fig = px.bar(
            x=dist.get_column(self.parent.topic_col).to_list(),
            y=dist.get_column("percent").to_list(),
            labels={"x": "Topic", "y": "Percentage (%)"},
            title="Distribution of Topics in Your Dataset",
            color=dist.get_column("percent").to_list(),
            color_continuous_scale="RdYlGn",
            text=dist.get_column("percent").to_list(),
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=400)
        return fig

    # ------------------------------------------------------------------
    # UMAP 2-D scatter
    # ------------------------------------------------------------------

    @overload
    def umap_2d(self, *, return_df: Literal[False] = ...) -> Figure: ...

    @overload
    def umap_2d(self, *, return_df: Literal[True]) -> pl.DataFrame: ...

    def umap_2d(self, *, return_df: bool = False) -> Figure | pl.DataFrame:
        """UMAP 2-D scatter coloured by topic.

        Parameters
        ----------
        return_df : bool, default False
            If ``True``, return the underlying DataFrame instead of a figure.

        Returns
        -------
        Figure or pl.DataFrame
        """
        coords = self.parent.compute.umap()
        umap_df = self.parent.df.with_columns(
            pl.Series("x", coords[:, 0]),
            pl.Series("y", coords[:, 1]),
        )

        if return_df:
            return umap_df

        import plotly.graph_objects as go

        fig = go.Figure()
        for topic in umap_df.get_column(self.parent.topic_col).unique().sort().to_list():
            subset = umap_df.filter(pl.col(self.parent.topic_col) == topic)
            texts = subset.get_column(self.parent.text_col).str.slice(0, 100).to_list()
            fig.add_trace(
                go.Scatter(
                    x=subset.get_column("x").to_list(),
                    y=subset.get_column("y").to_list(),
                    mode="markers",
                    name=str(topic),
                    marker=dict(size=6, opacity=0.6),
                    text=[t + "…" for t in texts],
                    hovertemplate=("<b>Topic:</b> %{fullData.name}<br><b>Text:</b> %{text}<extra></extra>"),
                )
            )
        fig.update_layout(
            title="Topic Separation Visualization (UMAP Projection)",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            height=600,
            hovermode="closest",
        )
        return fig

    # ------------------------------------------------------------------
    # Similarity heatmap
    # ------------------------------------------------------------------

    @overload
    def similarity_heatmap(self, *, return_df: Literal[False] = ...) -> Figure: ...

    @overload
    def similarity_heatmap(self, *, return_df: Literal[True]) -> pl.DataFrame: ...

    def similarity_heatmap(self, *, return_df: bool = False) -> Figure | pl.DataFrame:
        """Heatmap of pairwise topic TF-IDF cosine similarity.

        Parameters
        ----------
        return_df : bool, default False
            If ``True``, return the similarity matrix as a DataFrame.

        Returns
        -------
        Figure or pl.DataFrame
        """
        sim_df = self.parent.compute.topic_similarity()

        if return_df:
            return sim_df

        import plotly.graph_objects as go

        topics = sim_df.get_column("topic").to_list()
        values = sim_df.drop("topic").to_numpy()

        fig = go.Figure(
            data=go.Heatmap(
                z=values,
                x=topics,
                y=topics,
                colorscale="RdYlGn_r",
                zmin=0,
                zmax=1,
                colorbar_title="Overlap<br>Score",
            )
        )
        fig.update_layout(
            title="Topic Similarity Heatmap (Red = High Overlap)",
            height=600,
        )
        return fig
