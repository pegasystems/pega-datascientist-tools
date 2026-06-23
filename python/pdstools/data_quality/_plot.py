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

    # ------------------------------------------------------------------
    # Cleanlab label quality
    # ------------------------------------------------------------------

    @overload
    def label_quality_histogram(self, *, return_df: Literal[False] = ...) -> Figure: ...

    @overload
    def label_quality_histogram(self, *, return_df: Literal[True]) -> pl.DataFrame: ...

    def label_quality_histogram(self, *, return_df: bool = False) -> Figure | pl.DataFrame:
        """Histogram of per-sample label quality scores from Cleanlab.

        Parameters
        ----------
        return_df : bool, default False
            If ``True``, return the underlying DataFrame instead of a figure.

        Returns
        -------
        Figure or pl.DataFrame
        """
        results = self.parent.compute.cleanlab_audit()
        label_issues = results["label_issues"]

        scores = label_issues["label_score"].values
        is_issue = label_issues["is_label_issue"].values

        df = pl.DataFrame(
            {
                "label_score": scores,
                "is_label_issue": is_issue,
            }
        )

        if return_df:
            return df

        import plotly.express as px

        df_pd = df.with_columns(
            pl.when(pl.col("is_label_issue")).then(pl.lit("Likely Mislabeled")).otherwise(pl.lit("OK")).alias("status")
        ).to_pandas()

        fig = px.histogram(
            df_pd,
            x="label_score",
            color="status",
            nbins=50,
            title="Label Quality Score Distribution (Cleanlab)",
            labels={"label_score": "Label Quality Score", "count": "Count"},
            color_discrete_map={
                "Likely Mislabeled": "#dc3545",
                "OK": "#28a745",
            },
            barmode="overlay",
        )
        fig.update_layout(height=400)
        return fig

    @overload
    def cleanlab_issue_summary(self, *, return_df: Literal[False] = ...) -> Figure: ...

    @overload
    def cleanlab_issue_summary(self, *, return_df: Literal[True]) -> pl.DataFrame: ...

    def cleanlab_issue_summary(self, *, return_df: bool = False) -> Figure | pl.DataFrame:
        """Bar chart summarizing Cleanlab issue counts by type.

        Parameters
        ----------
        return_df : bool, default False
            If ``True``, return the summary DataFrame instead of a figure.

        Returns
        -------
        Figure or pl.DataFrame
        """
        results = self.parent.compute.cleanlab_audit()
        summary = results["summary"]

        df = (
            pl.from_pandas(summary.reset_index())
            .rename({"issue_type": "Issue Type", "num_issues": "Count"})
            .select("Issue Type", "Count")
        )

        if return_df:
            return df

        import plotly.express as px

        fig = px.bar(
            x=df.get_column("Issue Type").to_list(),
            y=df.get_column("Count").to_list(),
            labels={"x": "Issue Type", "y": "Number of Issues"},
            title="Cleanlab Data Issues Summary",
            color=df.get_column("Count").to_list(),
            color_continuous_scale="Reds",
            text=df.get_column("Count").to_list(),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # Topic learnability scorecard
    # ------------------------------------------------------------------

    @overload
    def learnability_scorecard(self, *, return_df: Literal[False] = ...) -> Figure: ...

    @overload
    def learnability_scorecard(self, *, return_df: Literal[True]) -> pl.DataFrame: ...

    def learnability_scorecard(self, *, return_df: bool = False) -> Figure | pl.DataFrame:
        """Horizontal bar chart of per-topic F1 scores with error bars.

        Color-coded: green (>0.8), orange (0.5–0.8), red (<0.5).

        Parameters
        ----------
        return_df : bool, default False
            If ``True``, return the underlying DataFrame instead of a figure.

        Returns
        -------
        Figure or pl.DataFrame
        """
        df = self.parent.compute.topic_learnability()

        if return_df:
            return df

        import plotly.graph_objects as go

        topics = df.get_column("topic").to_list()
        means = df.get_column("f1_mean").to_list()
        stds = df.get_column("f1_std").to_list()

        # Color based on F1 thresholds
        colors = []
        for m in means:
            if m > 0.8:
                colors.append("#28a745")  # green
            elif m >= 0.5:
                colors.append("#ffa500")  # orange
            else:
                colors.append("#dc3545")  # red

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=topics,
                x=means,
                orientation="h",
                marker_color=colors,
                error_x=dict(type="data", array=stds, visible=True),
            )
        )

        # Threshold lines
        fig.add_vline(x=0.8, line_dash="dash", line_color="#28a745", opacity=0.5)
        fig.add_vline(x=0.5, line_dash="dash", line_color="#ffa500", opacity=0.5)

        # Legend via invisible traces
        for label, color in [
            ("Good (>0.8)", "#28a745"),
            ("Needs attention (0.5-0.8)", "#ffa500"),
            ("Problematic (<0.5)", "#dc3545"),
        ]:
            fig.add_trace(
                go.Bar(
                    y=[None],
                    x=[None],
                    orientation="h",
                    marker_color=color,
                    name=label,
                    showlegend=True,
                )
            )

        fig.update_layout(
            title="Topic Learnability Scorecard",
            xaxis_title="F1 Score (mean ± std across 5 folds)",
            yaxis=dict(autorange="reversed"),
            height=max(400, len(topics) * 35 + 100),
            xaxis_range=[0, 1.05],
            showlegend=True,
            legend=dict(x=0.7, y=0.1),
            bargap=0.3,
        )
        return fig
