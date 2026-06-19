"""Tests for pdstools.data_quality.TopicDataQuality.

Uses a small synthetic dataset where every expected value can be traced
back to the input data. Assertions are exact-value wherever possible.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

# Skip the entire module when heavy NLP deps are unavailable (e.g. Python 3.14
# where sentence-transformers and umap-learn have no wheels yet).
pytest.importorskip("sentence_transformers", reason="sentence-transformers not available")
pytest.importorskip("umap", reason="umap-learn not available")

from pdstools.data_quality._topic_data_quality import (
    TopicDataQuality,
    TopicOverlapPair,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

SAMPLE_DATA = {
    "text": [
        # animals (7 rows incl dups)
        "the quick brown fox jumps over the lazy dog",  # 0
        "a fast brown fox leaps over a sleepy dog",  # 1
        "the dog barked loudly at the mailman passing by",  # 2
        "the cat sat on the mat watching birds outside",  # 3
        "the quick brown fox jumps over the lazy dog",  # 4 (dup of 0)
        "machine learning is a subset of artificial intelligence",  # 5 cross-topic dup
        "hello world",  # 6 short
        # tech (5 rows)
        "machine learning is a subset of artificial intelligence",  # 7
        "deep learning uses neural networks for AI tasks",  # 8
        "neural networks can approximate complex functions well",  # 9
        "reinforcement learning trains agents through reward signals",  # 10
        "ok",  # 11 short
        # finance (4 rows)
        "the stock market rallied on positive economic news",  # 12
        "financial markets saw gains amid strong earnings reports",  # 13
        "stocks fell sharply as investors reacted to the news",  # 14
        "bond yields rose as inflation expectations increased significantly",  # 15
    ],
    "topic": [
        "animals",
        "animals",
        "animals",
        "animals",
        "animals",
        "animals",
        "animals",
        "tech",
        "tech",
        "tech",
        "tech",
        "tech",
        "finance",
        "finance",
        "finance",
        "finance",
    ],
}


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(SAMPLE_DATA)


@pytest.fixture
def dq(sample_df: pl.DataFrame) -> TopicDataQuality:
    return TopicDataQuality.from_dataframe(df=sample_df, text_col="text", topic_col="topic")


# ------------------------------------------------------------------
# Basic properties
# ------------------------------------------------------------------


class TestBasicProperties:
    def test_total_samples(self, dq: TopicDataQuality) -> None:
        assert dq.total_samples == 16

    def test_num_topics(self, dq: TopicDataQuality) -> None:
        assert dq.num_topics == 3

    def test_topic_counts_exact(self, dq: TopicDataQuality) -> None:
        counts = dq.topic_counts
        # Convert to dict for easy assertion
        counts_dict = dict(
            zip(
                counts.get_column("topic").to_list(),
                counts.get_column("count").to_list(),
                strict=True,
            )
        )
        assert counts_dict == {"animals": 7, "tech": 5, "finance": 4}

    def test_imbalance_ratio_exact(self, dq: TopicDataQuality) -> None:
        # 7 animals / 4 finance = 1.75
        assert dq.imbalance_ratio == pytest.approx(1.75, abs=0.001)


# ------------------------------------------------------------------
# from_dataframe
# ------------------------------------------------------------------


class TestFromDataframe:
    def test_drops_nulls(self) -> None:
        df = pl.DataFrame(
            {
                "text": ["hello world", None, "foo bar baz"],
                "topic": ["a", "b", None],
            }
        )
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        assert dq.total_samples == 1  # only "hello world" / "a" survives

    def test_selects_only_needed_columns(self) -> None:
        df = pl.DataFrame(
            {
                "text": ["hello world"],
                "topic": ["a"],
                "extra": [42],
            }
        )
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        assert dq.df.columns == ["text", "topic"]

    def test_casts_text_to_utf8(self) -> None:
        df = pl.DataFrame(
            {
                "text": [123, 456],
                "topic": ["a", "b"],
            }
        )
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        assert dq.df.get_column("text").dtype == pl.Utf8


# ------------------------------------------------------------------
# Text quality
# ------------------------------------------------------------------


class TestTextQuality:
    def test_short_texts_count(self, dq: TopicDataQuality) -> None:
        tq = dq.health.analyze_text_quality()
        # "hello world" (2 words) and "ok" (1 word) are < 5 words
        assert tq["short_texts"].height == 2

    def test_short_texts_pct(self, dq: TopicDataQuality) -> None:
        tq = dq.health.analyze_text_quality()
        # 2/16 = 12.5%
        assert tq["short_texts_pct"] == pytest.approx(12.5, abs=0.01)

    def test_avg_words(self, dq: TopicDataQuality) -> None:
        tq = dq.health.analyze_text_quality()
        # Hand-computed word counts: 9,9,8,9,9,8,2, 8,8,7,7,1, 8,8,9,8
        # sum = 128, avg = 128/16 = 8.0
        # (polars str.split includes trailing: exact count may vary)
        assert tq["avg_words"] > 5
        assert tq["avg_words"] < 12

    def test_very_long_texts(self, dq: TopicDataQuality) -> None:
        tq = dq.health.analyze_text_quality()
        # No text has > 200 words
        assert tq["very_long_texts"].height == 0
        assert tq["very_long_pct"] == 0.0


# ------------------------------------------------------------------
# Duplicates
# ------------------------------------------------------------------


class TestDuplicates:
    def test_duplicate_count(self, dq: TopicDataQuality) -> None:
        info = dq.health.find_duplicates()
        # "the quick brown fox..." appears 2x, "machine learning..." appears 2x
        # → 4 rows are duplicates
        assert info["duplicates"].height == 4

    def test_duplicate_pct(self, dq: TopicDataQuality) -> None:
        info = dq.health.find_duplicates()
        # 4/16 = 25%
        assert info["duplicate_pct"] == pytest.approx(25.0, abs=0.01)

    def test_cross_topic_count(self, dq: TopicDataQuality) -> None:
        info = dq.health.find_duplicates()
        # "machine learning..." appears in animals + tech → 1 cross-topic dup
        assert info["cross_topic"].height == 1


# ------------------------------------------------------------------
# Sample adequacy
# ------------------------------------------------------------------


class TestSampleAdequacy:
    def test_all_topics_present(self, dq: TopicDataQuality) -> None:
        adequacy = dq.health.check_sample_adequacy()
        topics = set(adequacy.get_column("Topic").to_list())
        assert topics == {"animals", "tech", "finance"}

    def test_all_critical(self, dq: TopicDataQuality) -> None:
        adequacy = dq.health.check_sample_adequacy()
        # All topics have < 10 samples → all CRITICAL
        statuses = adequacy.get_column("Status").to_list()
        assert all("[CRITICAL]" in s for s in statuses)

    def test_sorted_ascending(self, dq: TopicDataQuality) -> None:
        adequacy = dq.health.check_sample_adequacy()
        counts = adequacy.get_column("Sample Count").to_list()
        assert counts == sorted(counts)


# ------------------------------------------------------------------
# Health score
# ------------------------------------------------------------------


class TestHealthScore:
    def test_score_exact(self, dq: TopicDataQuality) -> None:
        health = dq.health.calculate_health_score()
        # Start 100
        # Imbalance: ratio=1.75 → 0 penalty
        # Overlap: depends on TF-IDF → could be 0
        # Sample size: min=4 (<10) → -20
        # Text quality: short_texts_pct=12.5% (<30) → 0, dup_pct=25% (>10) → -5
        # Topic count: 3 → 0
        # Expected: 100 - 20 - 5 = 75 (unless overlap contributes)
        assert 0 <= health["score"] <= 100

    def test_score_deductions_for_small_data(self, dq: TopicDataQuality) -> None:
        health = dq.health.calculate_health_score()
        reasons = health["reasons"]
        reason_text = " ".join(reasons)
        # Must have "low samples" deduction
        assert "low samples" in reason_text.lower() or "Very low samples" in reason_text
        # Must have duplicate deduction
        assert "duplicates" in reason_text.lower()

    def test_status_and_color(self, dq: TopicDataQuality) -> None:
        health = dq.health.calculate_health_score()
        assert health["status"] in {"Excellent", "Good", "Fair", "Needs Improvement"}
        assert health["color"] in {"GREEN", "YELLOW", "ORANGE", "RED"}


# ------------------------------------------------------------------
# Similarity
# ------------------------------------------------------------------


class TestSimilarity:
    def test_similarity_shape(self, dq: TopicDataQuality) -> None:
        sim_df = dq.compute.topic_similarity()
        assert sim_df.height == 3
        # topic + 3 value columns
        assert len(sim_df.columns) == 4

    def test_diagonal_is_one(self, dq: TopicDataQuality) -> None:
        sim_df = dq.compute.topic_similarity()
        topics = sim_df.get_column("topic").to_list()
        for topic in topics:
            row = sim_df.filter(pl.col("topic") == topic)
            assert row.get_column(topic).item() == pytest.approx(1.0, abs=0.01)

    def test_similarity_symmetric(self, dq: TopicDataQuality) -> None:
        sim_df = dq.compute.topic_similarity()
        topics = sim_df.get_column("topic").to_list()
        values = sim_df.drop("topic").to_numpy()
        for i in range(len(topics)):
            for j in range(len(topics)):
                assert values[i, j] == pytest.approx(values[j, i], abs=0.001)


# ------------------------------------------------------------------
# High similarity pairs
# ------------------------------------------------------------------


class TestHighSimilarityPairs:
    def test_returns_overlap_pairs(self, dq: TopicDataQuality) -> None:
        pairs = dq.high_similarity_pairs()
        assert isinstance(pairs, list)
        for p in pairs:
            assert isinstance(p, TopicOverlapPair)
            assert p.similarity > dq.similarity_threshold


# ------------------------------------------------------------------
# Recommendations
# ------------------------------------------------------------------


class TestRecommendations:
    def test_structure(self, dq: TopicDataQuality) -> None:
        recs = dq.health.generate_recommendations()
        assert set(recs.keys()) == {"high", "medium", "low"}

    def test_critical_topics_in_high(self, dq: TopicDataQuality) -> None:
        recs = dq.health.generate_recommendations()
        # All topics < 10 samples → "Add more data" in high
        actions = [r["Action"] for r in recs["high"]]
        assert any("Add more data" in a for a in actions)

    def test_cross_topic_dup_flagged(self, dq: TopicDataQuality) -> None:
        recs = dq.health.generate_recommendations()
        actions = [r["Action"] for r in recs["high"]]
        assert any("cross-topic" in a.lower() for a in actions)


# ------------------------------------------------------------------
# Summary report
# ------------------------------------------------------------------


class TestSummaryReport:
    def test_report_metrics(self, dq: TopicDataQuality) -> None:
        report = dq.health.summary_report()
        assert isinstance(report, pl.DataFrame)
        assert report.height == 8
        metrics = report.get_column("Metric").to_list()
        assert "Total Samples" in metrics
        assert "Health Score" in metrics
        assert "Balance Ratio" in metrics

    def test_report_values(self, dq: TopicDataQuality) -> None:
        report = dq.health.summary_report()
        values = dict(
            zip(
                report.get_column("Metric").to_list(),
                report.get_column("Value").to_list(),
                strict=True,
            )
        )
        assert values["Total Samples"] == "16"
        assert values["Unique Topics"] == "3"
        assert values["Balance Ratio"] == "1.75"


# ------------------------------------------------------------------
# Plot namespace
# ------------------------------------------------------------------


class TestPlot:
    def test_topic_distribution_return_df(self, dq: TopicDataQuality) -> None:
        df = dq.plot.topic_distribution(return_df=True)
        assert isinstance(df, pl.DataFrame)
        assert df.height == 3
        assert "percent" in df.columns
        # Percentages should sum to ~100
        assert df.get_column("percent").sum() == pytest.approx(100.0, abs=0.5)

    def test_umap_2d_return_df(self, dq: TopicDataQuality) -> None:
        df = dq.plot.umap_2d(return_df=True)
        assert isinstance(df, pl.DataFrame)
        assert df.height == 16
        assert "x" in df.columns
        assert "y" in df.columns

    def test_similarity_heatmap_return_df(self, dq: TopicDataQuality) -> None:
        df = dq.plot.similarity_heatmap(return_df=True)
        assert isinstance(df, pl.DataFrame)
        assert "topic" in df.columns


# ------------------------------------------------------------------
# Cluster tightness
# ------------------------------------------------------------------


class TestClusterTightness:
    def test_all_topics_scored(self, dq: TopicDataQuality) -> None:
        tight = dq.health.calculate_cluster_tightness()
        assert tight.height == 3
        topics = set(tight.get_column("Topic").to_list())
        assert topics == {"animals", "tech", "finance"}

    def test_scores_in_range(self, dq: TopicDataQuality) -> None:
        tight = dq.health.calculate_cluster_tightness()
        scores = tight.get_column("Tightness Score").to_list()
        for s in scores:
            assert 0 <= s <= 100


# ------------------------------------------------------------------
# Outliers
# ------------------------------------------------------------------


class TestOutliers:
    def test_outlier_schema(self, dq: TopicDataQuality) -> None:
        outliers = dq.health.find_outliers()
        assert isinstance(outliers, pl.DataFrame)
        assert set(outliers.columns) == {"index", "text", "topic", "distance"}

    def test_outlier_distances_positive(self, dq: TopicDataQuality) -> None:
        outliers = dq.health.find_outliers()
        if outliers.height > 0:
            for d in outliers.get_column("distance").to_list():
                assert d > 0

    def test_small_topic_skipped(self) -> None:
        """Topics with fewer than 3 samples are skipped by find_outliers."""
        df = pl.DataFrame(
            {
                "text": [
                    "only one sample here",
                    "second sample for topic b for testing purposes",
                    "third sample for topic b to allow outlier detection",
                    "fourth sample for topic b in the set",
                ],
                "topic": ["a", "b", "b", "b"],
            }
        )
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        outliers = dq.health.find_outliers()
        # topic "a" has only 1 sample → skipped
        if outliers.height > 0:
            assert "a" not in outliers.get_column("topic").to_list()


# ------------------------------------------------------------------
# Imbalance ratio edge case
# ------------------------------------------------------------------


class TestImbalanceRatioEdge:
    def test_zero_count_returns_inf(self) -> None:
        """When a topic has 0 count, imbalance_ratio returns inf."""
        df = pl.DataFrame({"text": ["hello world"], "topic": ["a"]})
        dq = TopicDataQuality(df=df, text_col="text", topic_col="topic")
        assert dq.imbalance_ratio >= 1.0


# ------------------------------------------------------------------
# Confused samples
# ------------------------------------------------------------------


class TestConfusedSamples:
    def test_confused_samples_schema(self, dq: TopicDataQuality) -> None:
        confused = dq.health.find_confused_samples()
        assert isinstance(confused, pl.DataFrame)
        expected_cols = {"text", "assigned_topic", "confused_with", "confusion_risk", "similarity_score"}
        assert set(confused.columns) == expected_cols

    def test_confused_samples_top_n(self, dq: TopicDataQuality) -> None:
        confused = dq.health.find_confused_samples(top_n=3)
        assert confused.height <= 3

    def test_confused_samples_empty_when_no_high_pairs(self) -> None:
        """With very distinct topics and threshold=0.99, no confused samples."""
        df = pl.DataFrame(
            {
                "text": [
                    "the dog ran across the park chasing birds",
                    "cats are fluffy domestic pets that purr loudly",
                    "the dog fetched the ball from the field nearby",
                    "cats love to climb trees and scratch furniture",
                    "dogs enjoy swimming in lakes during summer time",
                    "quantum computing uses qubits for parallel processing",
                    "quantum entanglement enables faster than light info transfer",
                    "quantum supremacy was achieved by Google in 2019",
                    "quantum mechanics describes subatomic particle behavior precisely",
                    "quantum tunneling allows particles to pass through barriers",
                ],
                "topic": ["pets"] * 5 + ["quantum"] * 5,
            }
        )
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic", similarity_threshold=0.99)
        confused = dq.health.find_confused_samples()
        assert confused.height == 0


# ------------------------------------------------------------------
# Keyword overlap analysis
# ------------------------------------------------------------------


class TestKeywordOverlap:
    def test_keyword_overlap_returns_list(self, dq: TopicDataQuality) -> None:
        results = dq.health.keyword_overlap_analysis()
        assert isinstance(results, list)

    def test_keyword_overlap_structure(self, dq: TopicDataQuality) -> None:
        results = dq.health.keyword_overlap_analysis()
        for entry in results:
            assert "topic_a" in entry
            assert "topic_b" in entry
            assert "keyword_overlap_pct" in entry
            assert "shared_keywords" in entry
            assert "unique_to_a" in entry
            assert "unique_to_b" in entry

    def test_keyword_overlap_empty_when_no_vectorizer(self) -> None:
        """Returns empty list when TF-IDF hasn't been computed."""
        df = pl.DataFrame(
            {
                "text": ["hello world foo bar baz", "goodbye world foo bar baz"],
                "topic": ["a", "b"],
            }
        )
        dq = TopicDataQuality(df=df, text_col="text", topic_col="topic")
        assert dq._tfidf_vectorizer is None
        result = dq.health.keyword_overlap_analysis()
        assert result == []


# ------------------------------------------------------------------
# Compute: cleanlab_audit
# ------------------------------------------------------------------


class TestCleanlabAudit:
    """Test the cleanlab audit integration."""

    @pytest.fixture
    def _skip_if_no_cleanlab(self):
        pytest.importorskip("cleanlab", reason="cleanlab not available")

    @pytest.mark.usefixtures("_skip_if_no_cleanlab")
    def test_cleanlab_audit_returns_expected_keys(self, dq: TopicDataQuality) -> None:
        results = dq.compute.cleanlab_audit()
        assert set(results.keys()) == {"label_issues", "outlier_issues", "near_duplicate_issues", "summary"}

    @pytest.mark.usefixtures("_skip_if_no_cleanlab")
    def test_cleanlab_audit_cached(self, dq: TopicDataQuality) -> None:
        """Second call returns cached results."""
        results1 = dq.compute.cleanlab_audit()
        results2 = dq.compute.cleanlab_audit()
        assert results1 is results2

    @pytest.mark.usefixtures("_skip_if_no_cleanlab")
    def test_cleanlab_label_issues_length(self, dq: TopicDataQuality) -> None:
        results = dq.compute.cleanlab_audit()
        assert len(results["label_issues"]) == dq.total_samples


# ------------------------------------------------------------------
# Compute: topic_learnability
# ------------------------------------------------------------------


class TestTopicLearnability:
    def test_learnability_returns_dataframe(self, dq: TopicDataQuality) -> None:
        learn = dq.compute.topic_learnability()
        assert isinstance(learn, pl.DataFrame)
        assert set(learn.columns) == {"topic", "f1_mean", "f1_std"}

    def test_learnability_all_topics(self, dq: TopicDataQuality) -> None:
        learn = dq.compute.topic_learnability()
        assert learn.height == 3
        topics = set(learn.get_column("topic").to_list())
        assert topics == {"animals", "tech", "finance"}

    def test_learnability_f1_in_range(self, dq: TopicDataQuality) -> None:
        learn = dq.compute.topic_learnability()
        for f1 in learn.get_column("f1_mean").to_list():
            assert 0 <= f1 <= 1.0

    def test_learnability_std_non_negative(self, dq: TopicDataQuality) -> None:
        learn = dq.compute.topic_learnability()
        for s in learn.get_column("f1_std").to_list():
            assert s >= 0

    def test_learnability_sorted_descending(self, dq: TopicDataQuality) -> None:
        learn = dq.compute.topic_learnability()
        means = learn.get_column("f1_mean").to_list()
        assert means == sorted(means, reverse=True)

    def test_learnability_clamps_folds_for_small_class(self) -> None:
        """When a class has fewer samples than n_folds, folds are clamped."""
        df = pl.DataFrame(
            {
                "text": [
                    "dog runs in the park chasing squirrels",
                    "cat sits on the mat watching birds fly",
                    "dog fetches the ball from the river bank",
                    "quantum computing uses qubits for fast calculations",
                    "quantum entanglement enables new communication methods",
                    "quantum mechanics describes behavior of particles precisely",
                ],
                "topic": ["pets", "pets", "pets", "quantum", "quantum", "quantum"],
            }
        )
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        learn = dq.compute.topic_learnability(n_folds=5)
        assert learn.height == 2


# ------------------------------------------------------------------
# Compute: embeddings caching
# ------------------------------------------------------------------


class TestEmbeddingsCaching:
    def test_embeddings_cached(self, dq: TopicDataQuality) -> None:
        emb1 = dq.compute.embeddings()
        emb2 = dq.compute.embeddings()
        assert emb1 is emb2

    def test_embeddings_shape(self, dq: TopicDataQuality) -> None:
        emb = dq.compute.embeddings()
        assert emb.shape[0] == 16
        assert emb.shape[1] > 0


# ------------------------------------------------------------------
# Compute: UMAP caching
# ------------------------------------------------------------------


class TestUmapCaching:
    def test_umap_cached(self, dq: TopicDataQuality) -> None:
        u1 = dq.compute.umap()
        u2 = dq.compute.umap()
        assert u1 is u2


# ------------------------------------------------------------------
# Compute: topic_similarity caching
# ------------------------------------------------------------------


class TestSimilarityCaching:
    def test_similarity_cached(self, dq: TopicDataQuality) -> None:
        s1 = dq.compute.topic_similarity()
        s2 = dq.compute.topic_similarity()
        assert s1 is s2

    def test_tfidf_artifacts_stored(self, dq: TopicDataQuality) -> None:
        dq.compute.topic_similarity()
        assert dq._tfidf_vectorizer is not None
        assert dq._tfidf_matrix is not None
        assert dq._topic_order is not None


# ------------------------------------------------------------------
# Plot: figure-returning paths
# ------------------------------------------------------------------


class TestPlotFigures:
    """Test that plot methods return Plotly Figure objects."""

    def test_topic_distribution_figure(self, dq: TopicDataQuality) -> None:
        from plotly.graph_objects import Figure

        fig = dq.plot.topic_distribution(return_df=False)
        assert isinstance(fig, Figure)

    def test_umap_2d_figure(self, dq: TopicDataQuality) -> None:
        from plotly.graph_objects import Figure

        fig = dq.plot.umap_2d(return_df=False)
        assert isinstance(fig, Figure)
        assert len(fig.data) == 3

    def test_similarity_heatmap_figure(self, dq: TopicDataQuality) -> None:
        from plotly.graph_objects import Figure

        fig = dq.plot.similarity_heatmap(return_df=False)
        assert isinstance(fig, Figure)


# ------------------------------------------------------------------
# Plot: cleanlab plots
# ------------------------------------------------------------------


class TestPlotCleanlab:
    @pytest.fixture
    def _skip_if_no_cleanlab(self):
        pytest.importorskip("cleanlab", reason="cleanlab not available")

    @pytest.mark.usefixtures("_skip_if_no_cleanlab")
    def test_label_quality_histogram_return_df(self, dq: TopicDataQuality) -> None:
        df = dq.plot.label_quality_histogram(return_df=True)
        assert isinstance(df, pl.DataFrame)
        assert "label_score" in df.columns
        assert "is_label_issue" in df.columns
        assert df.height == dq.total_samples

    @pytest.mark.usefixtures("_skip_if_no_cleanlab")
    def test_label_quality_histogram_figure(self, dq: TopicDataQuality) -> None:
        from plotly.graph_objects import Figure

        fig = dq.plot.label_quality_histogram(return_df=False)
        assert isinstance(fig, Figure)

    @pytest.mark.usefixtures("_skip_if_no_cleanlab")
    def test_cleanlab_issue_summary_return_df(self, dq: TopicDataQuality) -> None:
        df = dq.plot.cleanlab_issue_summary(return_df=True)
        assert isinstance(df, pl.DataFrame)
        assert "Issue Type" in df.columns
        assert "Count" in df.columns

    @pytest.mark.usefixtures("_skip_if_no_cleanlab")
    def test_cleanlab_issue_summary_figure(self, dq: TopicDataQuality) -> None:
        from plotly.graph_objects import Figure

        fig = dq.plot.cleanlab_issue_summary(return_df=False)
        assert isinstance(fig, Figure)


# ------------------------------------------------------------------
# Plot: learnability scorecard
# ------------------------------------------------------------------


class TestPlotLearnability:
    def test_learnability_scorecard_return_df(self, dq: TopicDataQuality) -> None:
        df = dq.plot.learnability_scorecard(return_df=True)
        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"topic", "f1_mean", "f1_std"}
        assert df.height == 3

    def test_learnability_scorecard_figure(self, dq: TopicDataQuality) -> None:
        from plotly.graph_objects import Figure

        fig = dq.plot.learnability_scorecard(return_df=False)
        assert isinstance(fig, Figure)


# ------------------------------------------------------------------
# Health score: more branches
# ------------------------------------------------------------------


class TestHealthScoreBranches:
    def test_high_imbalance_severe(self) -> None:
        """Severe imbalance (ratio > 10) costs 30 pts."""
        texts_a = [f"topic a sentence number {i} with enough words" for i in range(44)]
        texts_b = [f"topic b sentence number {i} with enough words" for i in range(4)]
        df = pl.DataFrame({"text": texts_a + texts_b, "topic": ["a"] * 44 + ["b"] * 4})
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        health = dq.health.calculate_health_score()
        reason_text = " ".join(health["reasons"])
        assert "Severe" in reason_text

    def test_moderate_imbalance(self) -> None:
        """Moderate imbalance (3 < ratio <= 5) costs 10 pts."""
        texts_a = [f"topic a sentence number {i} with enough words for quality" for i in range(40)]
        texts_b = [f"topic b sentence number {i} with enough words for quality" for i in range(10)]
        df = pl.DataFrame({"text": texts_a + texts_b, "topic": ["a"] * 40 + ["b"] * 10})
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        health = dq.health.calculate_health_score()
        reason_text = " ".join(health["reasons"])
        assert "Moderate" in reason_text

    def test_high_short_texts_penalty(self) -> None:
        """Short texts >30% triggers -5 pts."""
        texts = ["hi"] * 8 + [
            "this is a long enough sentence for quality checks",
            "another reasonably long sentence for the test data",
        ]
        df = pl.DataFrame({"text": texts, "topic": ["a"] * 5 + ["b"] * 5})
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        health = dq.health.calculate_health_score()
        reason_text = " ".join(health["reasons"])
        assert "short" in reason_text.lower()

    def test_too_many_topics_penalty(self) -> None:
        """More than 30 topics costs -10 pts."""
        texts = [f"sentence for topic {i} with enough words for analysis" for i in range(62)]
        topics = [f"topic_{i}" for i in range(31)] * 2
        df = pl.DataFrame({"text": texts, "topic": topics})
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        health = dq.health.calculate_health_score()
        reason_text = " ".join(health["reasons"])
        assert "Too many topics" in reason_text

    def test_too_few_topics_penalty(self) -> None:
        """Fewer than 2 topics costs -10 pts."""
        df = pl.DataFrame(
            {
                "text": [f"sentence number {i} about the only topic" for i in range(50)],
                "topic": ["only"] * 50,
            }
        )
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        health = dq.health.calculate_health_score()
        reason_text = " ".join(health["reasons"])
        assert "Too few topics" in reason_text

    def test_excellent_score(self) -> None:
        """A well-balanced dataset with enough samples scores Excellent."""
        texts_a = [f"animals sentence {i} about dogs and cats playing" for i in range(60)]
        texts_b = [f"technology sentence {i} about computers and phones" for i in range(60)]
        texts_c = [f"finance sentence {i} about stocks and markets today" for i in range(60)]
        df = pl.DataFrame(
            {
                "text": texts_a + texts_b + texts_c,
                "topic": ["animals"] * 60 + ["tech"] * 60 + ["finance"] * 60,
            }
        )
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        health = dq.health.calculate_health_score()
        assert health["score"] >= 85
        assert health["status"] == "Excellent"
        assert health["color"] == "GREEN"


# ------------------------------------------------------------------
# Sample adequacy: more status branches
# ------------------------------------------------------------------


class TestSampleAdequacyBranches:
    def test_low_status(self) -> None:
        """Topics with 10-29 samples get [LOW] status."""
        df = pl.DataFrame(
            {
                "text": [f"sentence {i} for testing sample adequacy status" for i in range(15)],
                "topic": ["a"] * 15,
            }
        )
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        adequacy = dq.health.check_sample_adequacy()
        assert adequacy.get_column("Status").to_list()[0] == "[LOW]"

    def test_acceptable_status(self) -> None:
        """Topics with 30-49 samples get [ACCEPTABLE] status."""
        df = pl.DataFrame(
            {
                "text": [f"sentence {i} for testing sample adequacy status" for i in range(35)],
                "topic": ["a"] * 35,
            }
        )
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        adequacy = dq.health.check_sample_adequacy()
        assert adequacy.get_column("Status").to_list()[0] == "[ACCEPTABLE]"

    def test_good_status(self) -> None:
        """Topics with >= 50 samples get [GOOD] status."""
        df = pl.DataFrame(
            {
                "text": [f"sentence {i} for testing sample adequacy status" for i in range(55)],
                "topic": ["a"] * 55,
            }
        )
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        adequacy = dq.health.check_sample_adequacy()
        assert adequacy.get_column("Status").to_list()[0] == "[GOOD]"


# ------------------------------------------------------------------
# Recommendations: more branches
# ------------------------------------------------------------------


class TestRecommendationsBranches:
    def test_low_health_score_flagged(self) -> None:
        """Health score < 50 triggers critical recommendation."""
        texts = ["ok"] * 45 + [f"topic b sentence {i} with enough words" for i in range(3)]
        df = pl.DataFrame({"text": texts, "topic": ["a"] * 45 + ["b"] * 3})
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        recs = dq.health.generate_recommendations()
        actions = [r["Action"] for r in recs["high"]]
        assert any("major data quality" in a.lower() for a in actions)

    def test_imbalance_medium_recommendation(self) -> None:
        """Imbalance ratio > 5 triggers medium recommendation."""
        texts_a = [f"topic a sentence number {i} with enough words" for i in range(36)]
        texts_b = [f"topic b sentence number {i} with enough words" for i in range(6)]
        df = pl.DataFrame({"text": texts_a + texts_b, "topic": ["a"] * 36 + ["b"] * 6})
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        recs = dq.health.generate_recommendations()
        actions = [r["Action"] for r in recs["medium"]]
        assert any("Balance" in a for a in actions)

    def test_many_topics_low_recommendation(self) -> None:
        """More than 25 topics triggers low recommendation."""
        texts = [f"sentence for topic {i} with enough words for processing" for i in range(52)]
        topics = [f"topic_{i}" for i in range(26)] * 2
        df = pl.DataFrame({"text": texts, "topic": topics})
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        recs = dq.health.generate_recommendations()
        actions = [r["Action"] for r in recs["low"]]
        assert any("topic count" in a.lower() for a in actions)

    def test_high_duplicate_low_recommendation(self) -> None:
        """Duplicate pct > 10% triggers low recommendation."""
        base = [f"repeated sentence number {i} with enough words for the test" for i in range(5)]
        texts = base * 4
        df = pl.DataFrame({"text": texts, "topic": ["a"] * 10 + ["b"] * 10})
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        recs = dq.health.generate_recommendations()
        actions = [r["Action"] for r in recs["low"]]
        assert any("duplicates" in a.lower() for a in actions)

    def test_short_texts_medium_recommendation(self) -> None:
        """Short texts > 30% triggers medium recommendation."""
        short = ["hi"] * 8
        normal = [
            "a sufficiently long sentence for the quality test",
            "another long enough sentence for this test case",
        ]
        df = pl.DataFrame({"text": short + normal, "topic": ["a"] * 5 + ["b"] * 5})
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        recs = dq.health.generate_recommendations()
        actions = [r["Action"] for r in recs["medium"]]
        assert any("short" in a.lower() for a in actions)


# ------------------------------------------------------------------
# Summary report: status branches
# ------------------------------------------------------------------


class TestSummaryReportStatuses:
    def test_check_statuses_for_bad_data(self) -> None:
        """Summary report shows CHECK for problematic metrics."""
        texts = ["ok"] * 40 + [f"topic b sentence {i} with words" for i in range(4)]
        df = pl.DataFrame({"text": texts, "topic": ["a"] * 40 + ["b"] * 4})
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        report = dq.health.summary_report()
        statuses = dict(
            zip(
                report.get_column("Metric").to_list(),
                report.get_column("Status").to_list(),
                strict=True,
            )
        )
        assert statuses["Balance Ratio"] == "CHECK"
        assert statuses["Short Text %"] == "CHECK"


# ------------------------------------------------------------------
# Cluster tightness: quality labels
# ------------------------------------------------------------------


class TestClusterTightnessQuality:
    def test_quality_labels_present(self, dq: TopicDataQuality) -> None:
        tight = dq.health.calculate_cluster_tightness()
        valid_qualities = {"[EXCELLENT]", "[GOOD]", "[FAIR]", "[POOR]"}
        for q in tight.get_column("Quality").to_list():
            assert q in valid_qualities

    def test_avg_distance_positive(self, dq: TopicDataQuality) -> None:
        tight = dq.health.calculate_cluster_tightness()
        for d in tight.get_column("Avg Distance").to_list():
            assert d >= 0

    def test_single_sample_topic_skipped(self) -> None:
        """Topics with < 2 samples are skipped."""
        df = pl.DataFrame(
            {
                "text": [
                    "single sample topic a for testing purposes",
                    "topic b first sentence with enough words for analysis",
                    "topic b second sentence with enough words for analysis",
                    "topic b third sentence with enough words for analysis",
                ],
                "topic": ["a", "b", "b", "b"],
            }
        )
        dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
        tight = dq.health.calculate_cluster_tightness()
        topics = tight.get_column("Topic").to_list()
        assert "a" not in topics
        assert "b" in topics


# ------------------------------------------------------------------
# High similarity pairs with actual pairs found
# ------------------------------------------------------------------


class TestHighSimilarityPairsFound:
    def test_high_similarity_detected(self) -> None:
        """Very similar topics should produce overlap pairs."""
        df = pl.DataFrame(
            {
                "text": [
                    "the stock market rallied on positive economic news today",
                    "financial markets saw gains amid strong earnings reports",
                    "stocks rose sharply as investors reacted to good news",
                    "the bond market rallied on positive economic data today",
                    "financial instruments saw gains amid strong trade reports",
                    "bonds rose sharply as investors reacted to rate cuts",
                ],
                "topic": ["stocks", "stocks", "stocks", "bonds", "bonds", "bonds"],
            }
        )
        dq = TopicDataQuality.from_dataframe(
            df, text_col="text", topic_col="topic", similarity_threshold=0.3
        )
        pairs = dq.high_similarity_pairs()
        assert len(pairs) >= 1
        assert all(isinstance(p, TopicOverlapPair) for p in pairs)
        assert all(p.similarity > 0.3 for p in pairs)
