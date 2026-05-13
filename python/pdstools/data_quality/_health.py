"""Health sub-namespace: inspection, scoring, recommendations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from pdstools.utils.namespaces import LazyNamespace

if TYPE_CHECKING:
    from ._topic_data_quality import TopicDataQuality

logger = logging.getLogger(__name__)


class Health(LazyNamespace):
    """Inspection, health scoring, and recommendation generation.

    Attached to :pyattr:`TopicDataQuality.health`.
    """

    dependencies: ClassVar[list[str]] = ["numpy", "sklearn"]
    dependency_group = "data_quality"

    def __init__(self, parent: TopicDataQuality) -> None:
        super().__init__()
        self.parent = parent

    # ------------------------------------------------------------------
    # Text quality
    # ------------------------------------------------------------------

    def analyze_text_quality(self) -> dict:
        """Compute text quality metrics.

        Returns
        -------
        dict
            Keys: ``short_texts`` (pl.DataFrame), ``short_texts_pct``,
            ``very_long_texts`` (pl.DataFrame), ``very_long_pct``,
            ``avg_words``, ``median_words``.
        """
        df = self.parent.df.with_columns(
            pl.col(self.parent.text_col)
            .str.split(" ")
            .list.len()
            .alias("word_count"),
        )
        total = df.height

        short = df.filter(pl.col("word_count") < 5)
        very_long = df.filter(pl.col("word_count") > 200)

        stats = df.select(
            pl.col("word_count").mean().alias("avg"),
            pl.col("word_count").median().alias("median"),
        ).row(0)

        return {
            "short_texts": short,
            "short_texts_pct": (short.height / total) * 100,
            "very_long_texts": very_long,
            "very_long_pct": (very_long.height / total) * 100,
            "avg_words": stats[0],
            "median_words": stats[1],
        }

    # ------------------------------------------------------------------
    # Duplicates
    # ------------------------------------------------------------------

    def find_duplicates(self) -> dict:
        """Find exact and cross-topic duplicates.

        Returns
        -------
        dict
            Keys: ``duplicates`` (pl.DataFrame), ``cross_topic``
            (pl.DataFrame with text + n_topics), ``duplicate_pct``.
        """
        df = self.parent.df
        text_col = self.parent.text_col
        topic_col = self.parent.topic_col

        # Rows whose text appears more than once
        dup_texts = (
            df.group_by(text_col)
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") > 1)
            .select(text_col)
        )
        duplicates = df.join(dup_texts, on=text_col, how="semi")

        # Texts that appear in more than one topic
        cross_topic = (
            df.group_by(text_col)
            .agg(pl.col(topic_col).n_unique().alias("n_topics"))
            .filter(pl.col("n_topics") > 1)
        )

        return {
            "duplicates": duplicates,
            "cross_topic": cross_topic,
            "duplicate_pct": (duplicates.height / df.height) * 100,
        }

    # ------------------------------------------------------------------
    # Sample adequacy
    # ------------------------------------------------------------------

    def check_sample_adequacy(self) -> pl.DataFrame:
        """Evaluate per-topic sample counts.

        Returns
        -------
        pl.DataFrame
            Columns: ``Topic``, ``Sample Count``, ``Status``,
            ``Recommendation``.
        """
        counts = self.parent.topic_counts

        topics = counts.get_column(self.parent.topic_col).to_list()
        cnts = counts.get_column("count").to_list()

        rows: list[dict] = []
        for topic, count in zip(topics, cnts, strict=True):
            if count < 10:
                status, rec = "[CRITICAL]", f"Need {30 - count} more samples minimum"
            elif count < 30:
                status, rec = "[LOW]", f"Recommend {50 - count} more samples"
            elif count < 50:
                status, rec = "[ACCEPTABLE]", "Consider adding more for robustness"
            else:
                status, rec = "[GOOD]", "Sufficient samples"
            rows.append({
                "Topic": topic,
                "Sample Count": count,
                "Status": status,
                "Recommendation": rec,
            })

        return pl.DataFrame(rows).sort("Sample Count")

    # ------------------------------------------------------------------
    # Cluster tightness
    # ------------------------------------------------------------------

    def calculate_cluster_tightness(self) -> pl.DataFrame:
        """Measure how compact each topic cluster is in embedding space.

        Returns
        -------
        pl.DataFrame
            Per-topic tightness scores (0–100).
        """
        import numpy as np

        embeddings = self.parent.compute.embeddings()
        df = self.parent.df
        topic_col = self.parent.topic_col

        rows: list[dict] = []
        for topic in df.get_column(topic_col).unique().sort().to_list():
            mask = df.get_column(topic_col) == topic
            indices = [i for i, m in enumerate(mask.to_list()) if m]
            if len(indices) < 2:
                continue

            topic_emb = embeddings[indices]
            centroid = topic_emb.mean(axis=0)
            distances = np.linalg.norm(topic_emb - centroid, axis=1)
            avg_dist = float(distances.mean())
            score = max(0.0, 100.0 - avg_dist * 50)

            if score > 80:
                quality = "[EXCELLENT]"
            elif score > 60:
                quality = "[GOOD]"
            elif score > 40:
                quality = "[FAIR]"
            else:
                quality = "[POOR]"

            rows.append({
                "Topic": topic,
                "Tightness Score": round(score, 1),
                "Quality": quality,
                "Avg Distance": round(avg_dist, 3),
            })

        return pl.DataFrame(rows).sort("Tightness Score", descending=True)

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------

    def find_outliers(self) -> pl.DataFrame:
        """Detect potential mislabelled samples (per-topic outliers).

        Returns
        -------
        pl.DataFrame
            Outlier rows with distance from their topic centroid.
        """
        import numpy as np

        embeddings = self.parent.compute.embeddings()
        df = self.parent.df
        text_col = self.parent.text_col
        topic_col = self.parent.topic_col

        outliers: list[dict] = []
        for topic in df.get_column(topic_col).unique().sort().to_list():
            mask = df.get_column(topic_col) == topic
            indices = [i for i, m in enumerate(mask.to_list()) if m]
            if len(indices) < 3:
                continue

            topic_emb = embeddings[indices]
            centroid = topic_emb.mean(axis=0)
            distances = np.linalg.norm(topic_emb - centroid, axis=1)
            threshold = float(np.percentile(distances, 95))

            for pos, idx in enumerate(indices):
                if distances[pos] > threshold:
                    outliers.append({
                        "index": idx,
                        "text": df.row(idx, named=True)[text_col],
                        "topic": topic,
                        "distance": float(distances[pos]),
                    })

        return pl.DataFrame(outliers) if outliers else pl.DataFrame(
            schema={"index": pl.Int64, "text": pl.Utf8, "topic": pl.Utf8, "distance": pl.Float64}
        )

    # ------------------------------------------------------------------
    # Confused samples
    # ------------------------------------------------------------------

    def find_confused_samples(self, top_n: int = 10) -> pl.DataFrame:
        """Find samples sitting on the boundary between similar topics.

        Parameters
        ----------
        top_n : int, default 10
            Maximum rows to return.

        Returns
        -------
        pl.DataFrame
            Confused sample rows with confusion risk scores.
        """
        import numpy as np

        embeddings = self.parent.compute.embeddings()
        sim_df = self.parent.compute.topic_similarity()
        df = self.parent.df
        text_col = self.parent.text_col
        topic_col = self.parent.topic_col
        topics = sim_df.get_column("topic").to_list()
        sim_values = sim_df.drop("topic").to_numpy()

        # Collect high-sim pairs
        high_pairs: list[tuple[str, str, float]] = []
        for i, ti in enumerate(topics):
            for j, tj in enumerate(topics):
                if i < j and sim_values[i, j] > 0.7:
                    high_pairs.append((ti, tj, float(sim_values[i, j])))

        confused: list[dict] = []
        for topic_a, topic_b, similarity in high_pairs[:5]:
            mask_a = [
                k for k, v in enumerate(df.get_column(topic_col).to_list())
                if v == topic_a
            ]
            mask_b = [
                k for k, v in enumerate(df.get_column(topic_col).to_list())
                if v == topic_b
            ]
            if not mask_a or not mask_b:
                continue

            centroid_a = embeddings[mask_a].mean(axis=0)
            centroid_b = embeddings[mask_b].mean(axis=0)

            for idx in mask_a[:50]:
                dist_own = float(np.linalg.norm(embeddings[idx] - centroid_a))
                dist_other = float(np.linalg.norm(embeddings[idx] - centroid_b))

                if dist_other <= dist_own * 1.2:
                    text_val = df.row(idx, named=True)[text_col]
                    confused.append({
                        "text": text_val[:100] + "...",
                        "assigned_topic": topic_a,
                        "confused_with": topic_b,
                        "confusion_risk": round(
                            (1 - dist_other / (dist_own + dist_other)) * 100, 1
                        ),
                        "similarity_score": round(similarity, 3),
                    })

        if not confused:
            return pl.DataFrame(schema={
                "text": pl.Utf8,
                "assigned_topic": pl.Utf8,
                "confused_with": pl.Utf8,
                "confusion_risk": pl.Float64,
                "similarity_score": pl.Float64,
            })

        return (
            pl.DataFrame(confused)
            .sort("confusion_risk", descending=True)
            .head(top_n)
        )

    # ------------------------------------------------------------------
    # Keyword overlap
    # ------------------------------------------------------------------

    def keyword_overlap_analysis(self) -> list[dict]:
        """Analyse shared vs unique keywords between overlapping topic pairs.

        Returns
        -------
        list[dict]
            One dict per high-sim pair with shared/unique keyword lists.
        """
        import numpy as np

        sim_df = self.parent.compute.topic_similarity()
        pairs = self.parent.high_similarity_pairs()

        if self.parent._tfidf_vectorizer is None or self.parent._tfidf_matrix is None:
            return []

        feature_names = np.array(self.parent._tfidf_vectorizer.get_feature_names_out())
        topics = self.parent._topic_order or sim_df.get_column("topic").to_list()
        results: list[dict] = []

        for pair in pairs:
            vec_a = self.parent._tfidf_matrix[topics.index(pair.topic_a)].toarray()[0]
            vec_b = self.parent._tfidf_matrix[topics.index(pair.topic_b)].toarray()[0]

            top_a = set(feature_names[np.argsort(vec_a)[-50:]])
            top_b = set(feature_names[np.argsort(vec_b)[-50:]])

            shared = top_a & top_b
            unique_a = list(top_a - top_b)[-15:]
            unique_b = list(top_b - top_a)[-15:]
            overlap_pct = (len(shared) / len(top_a | top_b)) * 100

            results.append({
                "topic_a": pair.topic_a,
                "topic_b": pair.topic_b,
                "keyword_overlap_pct": round(overlap_pct, 1),
                "shared_keywords": sorted(shared)[:20],
                "unique_to_a": sorted(unique_a),
                "unique_to_b": sorted(unique_b),
            })

        return results

    # ------------------------------------------------------------------
    # Health score
    # ------------------------------------------------------------------

    def calculate_health_score(self) -> dict:
        """Compute the overall dataset health score (0–100).

        Returns
        -------
        dict
            Keys: ``score``, ``status``, ``color``, ``reasons``.
        """
        tq = self.analyze_text_quality()
        dup_info = self.find_duplicates()
        duplicate_pct = dup_info["duplicate_pct"]
        high_pairs = self.parent.high_similarity_pairs()

        score = 100
        reasons: list[str] = []

        # Class imbalance (max -30)
        ratio = self.parent.imbalance_ratio
        if ratio > 10:
            penalty = 30
        elif ratio > 5:
            penalty = 20
        elif ratio > 3:
            penalty = 10
        else:
            penalty = 0
        if penalty:
            label = "Severe" if ratio > 10 else "High" if ratio > 5 else "Moderate"
            reasons.append(f"{label} class imbalance (-{penalty} pts)")
        score -= penalty

        # Topic overlap (max -25)
        if high_pairs:
            overlap_penalty = min(25, len(high_pairs) * 5)
            reasons.append(f"{len(high_pairs)} topic pairs highly overlap (-{overlap_penalty} pts)")
            score -= overlap_penalty

        # Sample size (max -20)
        min_samples = self.parent.topic_counts.get_column("count").min()
        if min_samples < 10:
            penalty = 20
            reasons.append(f"Very low samples for some topics (-{penalty} pts)")
        elif min_samples < 30:
            penalty = 10
            reasons.append(f"Low samples for some topics (-{penalty} pts)")
        else:
            penalty = 0
        score -= penalty

        # Text quality (max -10)
        tq_penalty = 0
        if tq["short_texts_pct"] > 30:
            tq_penalty += 5
            reasons.append(f"{tq['short_texts_pct']:.1f}% texts too short (-5 pts)")
        if duplicate_pct > 10:
            tq_penalty += 5
            reasons.append(f"{duplicate_pct:.1f}% duplicates found (-5 pts)")
        score -= tq_penalty

        # Topic count (max -10)
        n = self.parent.num_topics
        if n > 30:
            reasons.append(f"Too many topics ({n}) (-10 pts)")
            score -= 10
        elif n < 2:
            reasons.append(f"Too few topics ({n}) (-10 pts)")
            score -= 10

        score = max(0, score)

        if score >= 85:
            status, color = "Excellent", "GREEN"
        elif score >= 70:
            status, color = "Good", "YELLOW"
        elif score >= 50:
            status, color = "Fair", "ORANGE"
        else:
            status, color = "Needs Improvement", "RED"

        return {"score": score, "status": status, "color": color, "reasons": reasons}

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def generate_recommendations(self) -> dict[str, list[dict]]:
        """Generate prioritised action items.

        Returns
        -------
        dict[str, list[dict]]
            Keys ``"high"``, ``"medium"``, ``"low"`` mapping to lists of
            recommendation dicts.
        """
        health = self.calculate_health_score()
        tq = self.analyze_text_quality()
        dup_info = self.find_duplicates()
        duplicate_pct = dup_info["duplicate_pct"]
        cross_topic = dup_info["cross_topic"]
        duplicates = dup_info["duplicates"]
        adequacy = self.check_sample_adequacy()
        high_pairs = self.parent.high_similarity_pairs()

        high_items: list[dict] = []
        medium_items: list[dict] = []
        low_items: list[dict] = []

        if cross_topic.height > 0:
            high_items.append({
                "Action": "[CRITICAL] Fix cross-topic duplicates",
                "Why": "Same text with different labels will break your model",
                "How": f"Review {cross_topic.height} duplicate texts and fix labels",
                "Impact": "CRITICAL - Must fix before training",
            })

        if health["score"] < 50:
            high_items.append({
                "Action": "[CRITICAL] Address major data quality issues",
                "Why": f"Health score {health['score']}/100 is too low for production",
                "How": "Focus on issues marked as critical in sections above",
                "Impact": "HIGH - Model will perform poorly",
            })

        critical_topics = adequacy.filter(pl.col("Status").str.contains("CRITICAL"))
        if critical_topics.height > 0:
            topic_list = ", ".join(
                critical_topics.get_column("Topic").cast(pl.Utf8).head(3).to_list()
            )
            suffix = "..." if critical_topics.height > 3 else ""
            high_items.append({
                "Action": f"Add more data for {critical_topics.height} topic(s)",
                "Why": "Too few samples will result in poor accuracy for these topics",
                "How": f"Collect more examples for: {topic_list}{suffix}",
                "Impact": "HIGH - These topics will have poor performance",
            })

        if self.parent.imbalance_ratio > 5:
            medium_items.append({
                "Action": "Balance your dataset",
                "Why": f"{self.parent.imbalance_ratio:.1f}x imbalance will bias model toward common topics",
                "How": "Add samples to underrepresented topics or use class weights",
                "Impact": "MEDIUM - Will reduce overall accuracy by 10-20%",
            })

        if high_pairs:
            pair_strs = ", ".join(f"'{p.topic_a}' & '{p.topic_b}'" for p in high_pairs[:2])
            suffix = "..." if len(high_pairs) > 2 else ""
            medium_items.append({
                "Action": "Address overlapping topics",
                "Why": "Model will confuse similar topics",
                "How": f"Review and possibly merge: {pair_strs}{suffix}",
                "Impact": "MEDIUM - Will cause confusion errors",
            })

        if tq["short_texts_pct"] > 30:
            medium_items.append({
                "Action": "Review short texts",
                "Why": f"{tq['short_texts_pct']:.1f}% texts may be too short",
                "How": "Filter out texts < 5 words or collect more detailed examples if needed",
                "Impact": "MEDIUM - May reduce data quality",
            })

        if self.parent.num_topics > 25:
            low_items.append({
                "Action": "Consider reducing topic count",
                "Why": f"{self.parent.num_topics} topics makes model complex and hard to interpret",
                "How": "Group similar topics or use hierarchical classification",
                "Impact": "LOW - Complexity issue, not accuracy",
            })

        if duplicate_pct > 10:
            low_items.append({
                "Action": "Remove duplicates",
                "Why": "May waste resources and can cause overfitting",
                "How": f"Remove {duplicates.height} duplicate samples",
                "Impact": "LOW - Optimization issue",
            })

        return {"high": high_items, "medium": medium_items, "low": low_items}

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def summary_report(self) -> pl.DataFrame:
        """Build a summary metrics table suitable for CSV export.

        Returns
        -------
        pl.DataFrame
            One-row-per-metric report with columns ``Metric``, ``Value``,
            ``Status``.
        """
        health = self.calculate_health_score()
        tq = self.analyze_text_quality()
        dup_info = self.find_duplicates()
        duplicate_pct = dup_info["duplicate_pct"]
        high_pairs = self.parent.high_similarity_pairs()
        tightness = self.calculate_cluster_tightness()

        avg_tightness = (
            tightness.get_column("Tightness Score").mean()
            if tightness.height > 0
            else 0.0
        )

        return pl.DataFrame({
            "Metric": [
                "Total Samples",
                "Unique Topics",
                "Health Score",
                "Balance Ratio",
                "Short Text %",
                "Duplicate %",
                "High Overlap Pairs",
                "Average Tightness",
            ],
            "Value": [
                str(self.parent.total_samples),
                str(self.parent.num_topics),
                str(health["score"]),
                f"{self.parent.imbalance_ratio:.2f}",
                f"{tq['short_texts_pct']:.1f}%",
                f"{duplicate_pct:.1f}%",
                str(len(high_pairs)),
                f"{avg_tightness:.1f}",
            ],
            "Status": [
                "OK",
                "OK" if 3 <= self.parent.num_topics <= 25 else "CHECK",
                health["status"],
                "OK" if self.parent.imbalance_ratio < 3 else "CHECK",
                "OK" if tq["short_texts_pct"] < 30 else "CHECK",
                "OK" if duplicate_pct < 10 else "CHECK",
                "OK" if not high_pairs else "CHECK",
                "OK" if avg_tightness > 60 else "CHECK",
            ],
        })
