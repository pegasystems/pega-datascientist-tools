"""Streamlit AppTest smoke + state-transition tests for Topic Data Quality.

Exercises the Home page rendering, verifies that the Quality Report
page renders with seeded data, and tests the upload → analyze flow.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

# Skip when NLP deps unavailable (Python 3.14)
pytest.importorskip("sentence_transformers", reason="sentence-transformers not available")
pytest.importorskip("umap", reason="umap-learn not available")

from pdstools.data_quality._topic_data_quality import TopicDataQuality

DQ_APP_DIR = Path(__file__).resolve().parents[3] / "pdstools" / "app" / "data_quality"


@pytest.fixture
def sample_dq() -> TopicDataQuality:
    """A TopicDataQuality pre-loaded with synthetic data."""
    df = pl.DataFrame(
        {
            "text": [
                "the quick brown fox jumps over the lazy dog",
                "a fast brown fox leaps over a sleepy dog",
                "machine learning is a subset of artificial intelligence",
                "deep learning uses neural networks for AI tasks",
                "the stock market rallied on positive economic news",
                "financial markets saw gains amid strong earnings reports",
                "the dog barked loudly at the mailman passing by",
                "neural networks can approximate complex functions well",
                "stocks fell sharply as investors reacted to the news",
                "the cat sat on the mat watching birds outside",
                "reinforcement learning trains agents through reward signals",
                "bond yields rose as inflation expectations increased significantly",
            ],
            "topic": [
                "animals",
                "animals",
                "tech",
                "tech",
                "finance",
                "finance",
                "animals",
                "tech",
                "finance",
                "animals",
                "tech",
                "finance",
            ],
        }
    )
    dq = TopicDataQuality.from_dataframe(df, text_col="text", topic_col="topic")
    # Pre-compute so the page doesn't need sentence-transformers
    dq.compute.embeddings()
    dq.compute.umap()
    dq.compute.topic_similarity()
    return dq


def test_home_page_renders() -> None:
    """Home page renders without exception."""
    at = AppTest.from_file(str(DQ_APP_DIR / "Home.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Home.py raised: {at.exception}"


def test_quality_report_renders_with_seeded_data(sample_dq: TopicDataQuality) -> None:
    """Quality Report page renders when topic_dq is in session state."""
    at = AppTest.from_file(
        str(DQ_APP_DIR / "pages" / "1_Quality_Report.py"),
        default_timeout=60,
    )
    at.session_state["topic_dq"] = sample_dq
    at.run()
    assert not at.exception, f"Quality Report raised: {at.exception}"
    # Should contain the health score heading
    all_markdown = " ".join(m.value for m in at.markdown)
    assert "Dataset Health Status" in all_markdown


def test_quality_report_stops_without_data() -> None:
    """Quality Report page warns and stops when no data is loaded."""
    at = AppTest.from_file(
        str(DQ_APP_DIR / "pages" / "1_Quality_Report.py"),
        default_timeout=30,
    )
    at.run()
    # Should show a warning (ensure_topic_dq guard)
    assert len(at.warning) >= 1


def test_about_page_renders() -> None:
    """About page renders without exception."""
    at = AppTest.from_file(
        str(DQ_APP_DIR / "pages" / "2_About.py"),
        default_timeout=30,
    )
    at.run()
    assert not at.exception, f"About page raised: {at.exception}"


# ------------------------------------------------------------------
# State-transition test: seeding topic_dq shows summary on home page
# ------------------------------------------------------------------


def test_home_shows_summary_when_topic_dq_seeded(sample_dq: TopicDataQuality) -> None:
    """When topic_dq is already in session state, the home page shows metrics.

    This tests the post-analysis state transition: once the analyzer
    exists, the home page should display the health score summary
    and an info message directing users to the Quality Report.
    """
    at = AppTest.from_file(str(DQ_APP_DIR / "Home.py"), default_timeout=30)
    at.session_state["topic_dq"] = sample_dq
    at.run()
    assert not at.exception, f"Home.py raised: {at.exception}"

    # Should display metrics (health score, total samples, etc.)
    metric_labels = [m.label for m in at.metric]
    assert "Health Score" in metric_labels
    assert "Total Samples" in metric_labels
    assert "Unique Topics" in metric_labels

    # Should show the info banner directing to Quality Report
    info_texts = [i.value for i in at.info]
    assert any("Quality Report" in t for t in info_texts)
