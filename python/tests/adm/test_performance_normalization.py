"""Tests for Performance normalization behavior.

Performance data from Pega comes in 50-100 scale (AUC), but for RAG evaluation
against MetricLimits.csv thresholds (0.52, 0.55, 0.9, 1.0), we normalize to 0.5-1.0 scale.
Display formatting via MetricFormats scales by 100 to show 50-100 to users.
"""

import polars as pl
import pytest
from pdstools import ADMDatamart, Prediction
from pdstools.utils.metric_limits import MetricFormats


@pytest.fixture
def model_df_50_100():
    """Model data with Performance in Pega's 50-100 scale."""
    return pl.LazyFrame(
        {
            "pyModelID": ["model1", "model2", "model3"],
            "pyPerformance": [55.0, 70.0, 85.0],
            "pyResponseCount": [1000, 2000, 3000],
            "pyPositives": [100, 200, 300],
            "pyNegatives": [900, 1800, 2700],
            "pySnapshotTime": ["20210101T010000.000 GMT"] * 3,
            "pyConfigurationName": ["Config1"] * 3,
            "pyName": ["Action1", "Action2", "Action3"],
            "pyChannel": ["Web"] * 3,
            "pyDirection": ["Inbound"] * 3,
        },
    )


@pytest.fixture
def predictor_df_50_100():
    """Predictor data with Performance in Pega's 50-100 scale."""
    return pl.LazyFrame(
        {
            "pyModelID": ["model1"] * 2,
            "pyPredictorName": ["pred1", "Classifier"],
            "pyPerformance": [62.0, 70.0],
            "pyBinIndex": [1, 1],
            "pyBinPositives": [10, 30],
            "pyBinNegatives": [90, 70],
            "pyBinResponseCount": [100, 100],
            "pySnapshotTime": ["20210101T010000.000 GMT"] * 2,
            "pyEntryType": ["Active", "Classifier"],
        },
    )


class TestPerformanceNormalization:
    """Test Performance normalization from 50-100 to 0.5-1.0 scale."""

    def test_model_data_normalized(self, model_df_50_100):
        """Model Performance 55, 70, 85 → 0.55, 0.70, 0.85."""
        dm = ADMDatamart(model_df=model_df_50_100)
        perfs = dm.model_data.select("Performance").collect()["Performance"].sort().to_list()
        assert perfs == pytest.approx([0.55, 0.70, 0.85], abs=0.001)

    def test_model_data_not_double_normalized(self):
        """Data already in 0-1 scale should not be normalized again."""
        df = pl.LazyFrame(
            {
                "pyModelID": ["m1"],
                "pyPerformance": [0.65],
                "pyResponseCount": [100],
                "pyPositives": [10],
                "pyNegatives": [90],
                "pySnapshotTime": ["20210101T010000.000 GMT"],
                "pyConfigurationName": ["C1"],
                "pyName": ["A1"],
            },
        )
        dm = ADMDatamart(model_df=df)
        assert dm.model_data.select("Performance").collect().item() == pytest.approx(
            0.65,
            abs=0.001,
        )

    def test_predictor_data_normalized(self, predictor_df_50_100):
        """Predictor Performance 62, 70 → 0.62, 0.70."""
        dm = ADMDatamart(predictor_df=predictor_df_50_100)
        perfs = dm.predictor_data.select("Performance").collect()["Performance"].sort().to_list()
        assert perfs == pytest.approx([0.62, 0.70], abs=0.01)

    def test_prediction_data_normalized(self):
        """Prediction mock data (50-100 scale) should be normalized to 0.5-1.0."""
        pred = Prediction.from_mock_data(days=5)
        perfs = pred.predictions.select("Performance").collect()["Performance"]
        assert perfs.max() <= 1.0 and perfs.min() >= 0.5


class TestAggregatesPerformance:
    """Test aggregate functions return Performance in normalized scale."""

    @pytest.mark.parametrize(
        "aggregate_method",
        ["summary_by_channel", "summary_by_configuration"],
    )
    def test_aggregates_return_normalized_performance(
        self,
        model_df_50_100,
        aggregate_method,
    ):
        """Aggregate Performance values should be in 0.5-1.0 range."""
        dm = ADMDatamart(model_df=model_df_50_100)
        summary = getattr(dm.aggregates, aggregate_method)().collect()
        perf = summary["Performance"][0]
        assert 0.5 <= perf <= 1.0, f"{aggregate_method} Performance {perf} should be in 0.5-1.0"


class TestMetricFormatsDisplay:
    """Test that MetricFormats displays normalized values correctly."""

    def test_model_performance_format(self):
        """ModelPerformance: scale_by=100, decimals=2 → 0.5677 displays as 56.77."""
        fmt = MetricFormats.get("ModelPerformance")
        assert fmt.scale_by == 100 and fmt.decimals == 2
        assert "56.77" in fmt.format_value(0.5677)
