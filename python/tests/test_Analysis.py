"""Tests for the ADM Analysis module (automated health check findings)."""

import pathlib

import polars as pl
import pytest
from pdstools import ADMDatamart, datasets
from pdstools.adm.Analysis import Analysis, Finding, _gini

basePath = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture
def sample_dm():
    """Fixture using CDH sample data."""
    return datasets.cdh_sample()


@pytest.fixture
def analysis(sample_dm):
    """Fixture providing the Analysis namespace."""
    return sample_dm.analysis


# ── Finding dataclass ───────────────────────────────────────────────────


class TestFinding:
    def test_creation(self):
        f = Finding(
            severity="warning",
            category="model",
            title="Test finding",
            detail="Some detail here.",
        )
        assert f.severity == "warning"
        assert f.category == "model"
        assert f.data == {}

    def test_creation_with_data(self):
        f = Finding(
            severity="critical",
            category="channel",
            title="Test",
            detail="Detail",
            data={"count": 42},
        )
        assert f.data["count"] == 42

    def test_str_representation(self):
        f = Finding(
            severity="critical",
            category="channel",
            title="Bad channel",
            detail="Detail",
        )
        result = str(f)
        assert "❌" in result
        assert "channel" in result
        assert "Bad channel" in result

    def test_frozen(self):
        f = Finding(
            severity="info",
            category="model",
            title="Test",
            detail="Detail",
        )
        with pytest.raises(AttributeError):
            f.severity = "critical"


# ── Gini helper ─────────────────────────────────────────────────────────


class TestGini:
    def test_equal_distribution(self):
        assert _gini([1, 1, 1, 1]) == pytest.approx(0.0, abs=0.01)

    def test_perfect_inequality(self):
        assert _gini([0, 0, 0, 100]) == pytest.approx(0.75, abs=0.01)

    def test_empty(self):
        assert _gini([]) == 0.0

    def test_all_zeros(self):
        assert _gini([0, 0, 0]) == 0.0

    def test_with_nans(self):
        result = _gini([1.0, float("nan"), 2.0, 3.0])
        assert 0 < result < 1

    def test_single_value(self):
        assert _gini([42]) == 0.0


# ── Analysis namespace ──────────────────────────────────────────────────


class TestAnalysisNamespace:
    def test_accessible_on_datamart(self, sample_dm):
        assert hasattr(sample_dm, "analysis")
        assert isinstance(sample_dm.analysis, Analysis)

    def test_findings_returns_list(self, analysis):
        results = analysis.findings()
        assert isinstance(results, list)
        assert all(isinstance(f, Finding) for f in results)

    def test_findings_sorted_by_severity(self, analysis):
        results = analysis.findings()
        if len(results) > 1:
            severity_order = {"critical": 0, "warning": 1, "info": 2}
            orders = [severity_order[f.severity] for f in results]
            assert orders == sorted(orders)

    def test_findings_have_valid_severity(self, analysis):
        for f in analysis.findings():
            assert f.severity in ("critical", "warning", "info")

    def test_findings_have_valid_category(self, analysis):
        valid_categories = {
            "channel",
            "model",
            "predictor",
            "prediction",
            "taxonomy",
            "response_distribution",
            "trend",
            "configuration",
        }
        for f in analysis.findings():
            assert f.category in valid_categories

    def test_findings_have_nonempty_title_and_detail(self, analysis):
        for f in analysis.findings():
            assert len(f.title) > 0
            assert len(f.detail) > 0


# ── Individual check methods ────────────────────────────────────────────


class TestChannelChecks:
    def test_no_crash_with_sample_data(self, analysis):
        active_filter = pl.lit(True)
        results = analysis._check_channels(active_filter)
        assert isinstance(results, list)

    def test_detects_zero_responses(self):
        dm = ADMDatamart(
            model_df=pl.DataFrame(
                {
                    "ModelID": ["m1"],
                    "SnapshotTime": ["20250101"],
                    "Performance": [0.5],
                    "SuccessRate": [0.0],
                    "Positives": [0],
                    "ResponseCount": [0],
                    "Channel": ["Web"],
                    "Direction": ["Inbound"],
                    "Configuration": ["TestConfig"],
                    "Name": ["A"],
                    "Treatment": ["T1"],
                    "Issue": ["Sales"],
                    "Group": ["Cards"],
                }
            )
            .lazy()
            .with_columns(
                pl.col("SnapshotTime").str.to_datetime("%Y%m%d"),
                pl.col("Performance").cast(pl.Float64),
                pl.col("SuccessRate").cast(pl.Float64),
                pl.col("Positives").cast(pl.Int64),
                pl.col("ResponseCount").cast(pl.Int64),
            ),
        )
        results = dm.analysis._check_channels(pl.lit(True))
        critical = [f for f in results if f.severity == "critical"]
        assert any("zero responses" in f.title for f in critical)


class TestConfigurationChecks:
    def test_no_crash_with_sample_data(self, analysis, sample_dm):
        last_data = analysis._get_last_data()
        results = analysis._check_configurations(last_data)
        assert isinstance(results, list)


class TestModelMaturityChecks:
    def test_no_crash_with_sample_data(self, analysis):
        last_data = analysis._get_last_data()
        results = analysis._check_model_maturity(last_data)
        assert isinstance(results, list)
        # Sample data should have at least some maturity findings
        assert len(results) > 0

    def test_detects_model_categories(self, analysis):
        last_data = analysis._get_last_data()
        results = analysis._check_model_maturity(last_data)
        titles = " ".join(f.title for f in results)
        # Should report on mature models with decent performance
        assert "mature" in titles.lower() or "performance" in titles.lower()


class TestTaxonomyChecks:
    def test_no_crash_with_sample_data(self, analysis):
        results = analysis._check_taxonomy()
        assert isinstance(results, list)


class TestResponseDistributionChecks:
    def test_no_crash_with_sample_data(self, analysis):
        last_data = analysis._get_last_data()
        results = analysis._check_response_distribution(last_data, pl.lit(True))
        assert isinstance(results, list)


class TestTrendChecks:
    def test_no_crash_with_sample_data(self, analysis):
        results = analysis._check_trends(pl.lit(True))
        assert isinstance(results, list)


class TestPredictorChecks:
    def test_no_crash_with_sample_data(self, analysis, sample_dm):
        if sample_dm.predictor_data is not None:
            results = analysis._check_predictors()
            assert isinstance(results, list)
            # Sample data has predictors, should have some findings
            assert len(results) > 0


class TestPredictionChecks:
    def test_no_crash_without_prediction(self, analysis):
        # When no prediction data, findings should be called without it
        results = analysis.findings(prediction=None)
        assert isinstance(results, list)


# ── Integration test ────────────────────────────────────────────────────


class TestFullAnalysis:
    def test_full_findings_with_sample_data(self, sample_dm):
        """End-to-end test: run all findings on sample data."""
        results = sample_dm.analysis.findings()
        assert len(results) > 0

        # Should cover multiple categories
        categories = {f.category for f in results}
        assert len(categories) >= 2

        # Every finding should be well-formed
        for f in results:
            assert f.title
            assert f.detail
            assert isinstance(f.data, dict)

    def test_custom_active_filter(self, sample_dm):
        results = sample_dm.analysis.findings(
            active_filter=pl.col("ResponseCount") > 100,
        )
        assert isinstance(results, list)

    def test_custom_threshold_days(self, sample_dm):
        results = sample_dm.analysis.findings(active_threshold_days=7)
        assert isinstance(results, list)
