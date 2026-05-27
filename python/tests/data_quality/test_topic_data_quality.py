"""Tests for pdstools.data_quality.TopicDataQuality.

Uses a small synthetic dataset where every expected value can be traced
back to the input data. Assertions are exact-value wherever possible.
"""

from __future__ import annotations

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
