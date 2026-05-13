"""Compute sub-namespace: embeddings, UMAP, TF-IDF similarity."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from pdstools.utils.namespaces import LazyNamespace

if TYPE_CHECKING:
    import numpy as np

    from ._topic_data_quality import TopicDataQuality

logger = logging.getLogger(__name__)


class Compute(LazyNamespace):
    """Embedding, UMAP, and similarity computations.

    Attached to :pyattr:`TopicDataQuality.compute`.
    """

    dependencies: ClassVar[list[str]] = ["numpy", "sklearn", "sentence_transformers", "umap"]
    dependency_group = "data_quality"

    def __init__(self, parent: TopicDataQuality) -> None:
        super().__init__()
        self.parent = parent

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embeddings(self) -> np.ndarray:
        """Compute sentence embeddings using ``sentence-transformers``.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(n_samples, embedding_dim)``.
        """
        if self.parent._embeddings is not None:
            return self.parent._embeddings

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        self.parent._embeddings = model.encode(
            self.parent.df.get_column(self.parent.text_col).to_list(),
            show_progress_bar=False,
        )
        return self.parent._embeddings

    # ------------------------------------------------------------------
    # UMAP
    # ------------------------------------------------------------------

    def umap(self) -> np.ndarray:
        """Compute 2-D UMAP projection of the embeddings.

        Coordinates are stored on ``parent._umap_coords`` (not appended
        to the input DataFrame).

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples, 2)``.
        """
        if self.parent._umap_coords is not None:
            return self.parent._umap_coords

        import umap

        emb = self.embeddings()
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        self.parent._umap_coords = reducer.fit_transform(emb)
        return self.parent._umap_coords

    # ------------------------------------------------------------------
    # TF-IDF topic similarity
    # ------------------------------------------------------------------

    def topic_similarity(self) -> pl.DataFrame:
        """Compute pairwise topic similarity using TF-IDF + cosine.

        Returns
        -------
        pl.DataFrame
            Square similarity matrix with a ``topic`` column and one
            column per topic.
        """
        if self.parent._sim_df is not None:
            return self.parent._sim_df

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Concatenate all texts per topic
        topic_texts = (
            self.parent.df.group_by(self.parent.topic_col)
            .agg(pl.col(self.parent.text_col).str.join(" "))
            .sort(self.parent.topic_col)
        )
        topics = topic_texts.get_column(self.parent.topic_col).to_list()
        texts = topic_texts.get_column(self.parent.text_col).to_list()

        vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        tfidf = vectorizer.fit_transform(texts)
        sim_matrix: np.ndarray = cosine_similarity(tfidf)

        self.parent._tfidf_vectorizer = vectorizer
        self.parent._tfidf_matrix = tfidf
        self.parent._topic_order = topics

        # Build polars DataFrame: topic col + one column per topic
        data: dict[str, list] = {"topic": topics}
        for i, t in enumerate(topics):
            data[t] = sim_matrix[i].tolist()
        self.parent._sim_df = pl.DataFrame(data)
        return self.parent._sim_df
