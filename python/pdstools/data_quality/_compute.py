"""Compute sub-namespace: embeddings, UMAP, TF-IDF similarity."""

from __future__ import annotations

import logging
from collections import Counter
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
    dependency_group = "nlp"

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

    # ------------------------------------------------------------------
    # Cross-validated diagnostics readiness
    # ------------------------------------------------------------------

    def cross_validated_checks_warning(self) -> str | None:
        """Return a warning when CV-based diagnostics cannot run safely.

        Cleanlab audit and topic learnability both rely on stratified
        cross-validation. That requires at least two topics and at least
        two samples in every topic; otherwise one training fold ends up
        single-class and the classifier fit fails.

        Returns
        -------
        str | None
            User-facing warning text when the diagnostics should be
            skipped, otherwise ``None``.
        """
        topic_counts = Counter(self.parent.df.get_column(self.parent.topic_col).to_list())
        if len(topic_counts) < 2:
            return (
                "Cleanlab audit and topic learnability are skipped because the dataset "
                "needs at least 2 topics for cross-validated diagnostics."
            )

        min_class_count = min(topic_counts.values())
        if min_class_count < 2:
            return (
                "Cleanlab audit and topic learnability are skipped because every topic "
                "needs at least 2 samples for cross-validated diagnostics."
            )

        return None

    def _validated_cross_validation_folds(self, *, max_folds: int = 5) -> int:
        """Return a safe stratified CV fold count or raise a clear error.

        Parameters
        ----------
        max_folds : int, default 5
            Upper bound for the number of folds.

        Returns
        -------
        int
            Fold count safe for stratified cross-validation.

        Raises
        ------
        ValueError
            If the dataset does not have enough class support for
            cross-validated diagnostics.
        """
        warning = self.cross_validated_checks_warning()
        if warning is not None:
            raise ValueError(warning)

        topic_counts = Counter(self.parent.df.get_column(self.parent.topic_col).to_list())
        return min(max_folds, min(topic_counts.values()))

    # ------------------------------------------------------------------
    # Cleanlab audit
    # ------------------------------------------------------------------

    def cleanlab_audit(self) -> dict:
        """Run Cleanlab Datalab audit for label issues, outliers, and near-duplicates.

        Uses the cached sentence embeddings and trains a cross-validated
        Logistic Regression to produce out-of-sample predicted probabilities,
        then passes both to Cleanlab's ``Datalab.find_issues()``.

        Results are cached on ``parent._cleanlab_results``.

        Returns
        -------
        dict
            Keys: ``"label_issues"`` (pd.DataFrame), ``"outlier_issues"``
            (pd.DataFrame), ``"near_duplicate_issues"`` (pd.DataFrame),
            ``"summary"`` (pd.DataFrame).
        """
        if self.parent._cleanlab_results is not None:
            return self.parent._cleanlab_results

        from cleanlab import Datalab
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict

        labels = self.parent.df.get_column(self.parent.topic_col).to_list()
        texts = self.parent.df.get_column(self.parent.text_col).to_list()
        cv = self._validated_cross_validation_folds(max_folds=5)
        embeddings = self.embeddings()

        # Out-of-sample predicted probabilities via cross-validation
        model = LogisticRegression(max_iter=400)
        pred_probs = cross_val_predict(model, embeddings, labels, cv=cv, method="predict_proba")

        # Cleanlab Datalab audit
        data_dict = {"texts": texts, "labels": labels}
        lab = Datalab(data_dict, label_name="labels")
        lab.find_issues(pred_probs=pred_probs, features=embeddings)

        self.parent._cleanlab_results = {
            "label_issues": lab.get_issues("label"),
            "outlier_issues": lab.get_issues("outlier"),
            "near_duplicate_issues": lab.get_issues("near_duplicate"),
            "summary": lab.get_issue_summary(),
        }
        return self.parent._cleanlab_results

    # ------------------------------------------------------------------
    # Topic learnability scorecard
    # ------------------------------------------------------------------

    def topic_learnability(self, *, n_folds: int = 5) -> pl.DataFrame:
        """Compute per-topic F1 scores via stratified cross-validation.

        Trains a Logistic Regression on the sentence embeddings and
        measures per-class F1 across folds.

        Parameters
        ----------
        n_folds : int, default 5
            Number of stratified cross-validation folds.

        Returns
        -------
        pl.DataFrame
            Columns: ``topic``, ``f1_mean``, ``f1_std``.
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import LabelEncoder

        embeddings = self.embeddings()
        labels = np.array(self.parent.df.get_column(self.parent.topic_col).to_list())

        le = LabelEncoder()
        y = le.fit_transform(labels)
        classes = le.classes_
        effective_folds = self._validated_cross_validation_folds(max_folds=n_folds)
        skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)
        # per-class F1 per fold: shape (n_folds, n_classes)
        fold_f1s = []

        for train_idx, test_idx in skf.split(embeddings, y):
            model = LogisticRegression(max_iter=400)
            model.fit(embeddings[train_idx], y[train_idx])
            preds = model.predict(embeddings[test_idx])
            per_class_f1 = f1_score(y[test_idx], preds, labels=range(len(classes)), average=None, zero_division=0.0)
            fold_f1s.append(per_class_f1)

        fold_f1s_arr = np.array(fold_f1s)  # (n_folds, n_classes)
        f1_means = fold_f1s_arr.mean(axis=0)
        f1_stds = fold_f1s_arr.std(axis=0)

        return pl.DataFrame(
            {
                "topic": classes.tolist(),
                "f1_mean": f1_means.tolist(),
                "f1_std": f1_stds.tolist(),
            }
        ).sort("f1_mean", descending=True)
