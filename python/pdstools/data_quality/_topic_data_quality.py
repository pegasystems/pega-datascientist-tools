"""Topic data quality analysis for text classification datasets.

Provides ``TopicDataQuality``, a polars-based analyzer that evaluates
the readiness of a labelled text dataset for topic classification.

All heavy-lifting lives in three ``LazyNamespace`` sub-namespaces:

- ``compute`` — embeddings, UMAP, TF-IDF similarity
- ``plot`` — Plotly figures (topic distribution, UMAP, heatmap)
- ``health`` — inspection, scoring, recommendations

Requires the ``nlp`` optional dependency group::

    pip install pdstools[nlp]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from ._compute import Compute
    from ._health import Health
    from ._plot import Plot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TopicOverlapPair:
    """A pair of topics with high similarity."""

    topic_a: str
    topic_b: str
    similarity: float


@dataclass
class TopicDataQuality:
    """Analyze topic-labelled text data for quality issues.

    Use the :meth:`from_dataframe` classmethod to build an instance from
    a raw DataFrame.  The constructor (``__init__``) is pure — it only
    stores already-clean data and wires up the sub-namespaces.

    Parameters
    ----------
    df : pl.DataFrame
        Clean input containing at least ``text_col`` and ``topic_col``,
        with nulls dropped and text cast to ``Utf8``.
    text_col : str
        Column name containing text data.
    topic_col : str
        Column name containing topic labels.
    similarity_threshold : float, default 0.8
        Topic pairs above this TF-IDF cosine similarity are flagged.
    """

    df: pl.DataFrame
    text_col: str
    topic_col: str
    similarity_threshold: float = 0.8

    # Mutable computed state (set by sub-namespace methods)
    _embeddings: Any | None = field(default=None, repr=False)
    _sim_df: pl.DataFrame | None = field(default=None, repr=False)
    _tfidf_vectorizer: Any | None = field(default=None, repr=False)
    _tfidf_matrix: Any | None = field(default=None, repr=False)
    _umap_coords: Any | None = field(default=None, repr=False)
    _topic_order: list[str] | None = field(default=None, repr=False)

    # Sub-namespace annotations (for IDE auto-complete)
    compute: Compute = field(init=False, repr=False)
    plot: Plot = field(init=False, repr=False)
    health: Health = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from ._compute import Compute
        from ._health import Health
        from ._plot import Plot

        self.compute = Compute(self)
        self.plot = Plot(self)
        self.health = Health(self)

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        text_col: str,
        topic_col: str,
        *,
        similarity_threshold: float = 0.8,
    ) -> TopicDataQuality:
        """Build from a raw DataFrame, filtering and coercing columns.

        Parameters
        ----------
        df : pl.DataFrame
            Raw input — may contain extra columns and nulls.
        text_col : str
            Column name containing text data.
        topic_col : str
            Column name containing topic labels.
        similarity_threshold : float, default 0.8
            Topic pairs above this TF-IDF cosine similarity are flagged.

        Returns
        -------
        TopicDataQuality
        """
        clean = df.select(text_col, topic_col).drop_nulls().with_columns(pl.col(text_col).cast(pl.Utf8))
        return cls(
            df=clean,
            text_col=text_col,
            topic_col=topic_col,
            similarity_threshold=similarity_threshold,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def total_samples(self) -> int:
        """Total number of rows."""
        return self.df.height

    @property
    def num_topics(self) -> int:
        """Number of unique topic labels."""
        return self.df.get_column(self.topic_col).n_unique()

    @property
    def topic_counts(self) -> pl.DataFrame:
        """Per-topic sample counts, sorted descending."""
        return self.df.get_column(self.topic_col).value_counts().sort("count", descending=True)

    @property
    def imbalance_ratio(self) -> float:
        """Ratio of the largest to the smallest topic count."""
        counts = self.topic_counts.get_column("count")
        mn = counts.min()
        if mn == 0:
            return float("inf")
        return counts.max() / mn

    # ------------------------------------------------------------------
    # Convenience (delegate to sub-namespaces)
    # ------------------------------------------------------------------

    def high_similarity_pairs(self) -> list[TopicOverlapPair]:
        """Return topic pairs exceeding the similarity threshold."""
        sim_df = self.compute.topic_similarity()
        topics = sim_df.get_column("topic").to_list()
        values = sim_df.drop("topic").to_numpy()

        pairs: list[TopicOverlapPair] = []
        for i, ti in enumerate(topics):
            for j, tj in enumerate(topics):
                if i < j and values[i, j] > self.similarity_threshold:
                    pairs.append(TopicOverlapPair(ti, tj, float(values[i, j])))
        return pairs
