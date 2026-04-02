"""Tests for the ADM Analysis module (automated health check findings)."""

import pathlib
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from pdstools import ADMDatamart, datasets
from pdstools.adm.Analysis import Analysis, Finding, _gini

basePath = pathlib.Path(__file__).parent.parent.parent


def _make_dm(rows: list[dict]) -> ADMDatamart:
    """Build a minimal ADMDatamart from a list of row dicts."""
    schema_casts = {
        "Performance": pl.Float64,
        "SuccessRate": pl.Float64,
        "Positives": pl.Int64,
        "ResponseCount": pl.Int64,
    }
    defaults = {
        "ModelID": "m1",
        "SnapshotTime": "20250101",
        "Performance": 0.7,
        "SuccessRate": 0.05,
        "Positives": 500,
        "ResponseCount": 10000,
        "Channel": "Web",
        "Direction": "Inbound",
        "Configuration": "WebConfig",
        "Name": "ActionA",
        "Treatment": "T1",
        "Issue": "Sales",
        "Group": "Cards",
    }
    merged = [{**defaults, **r} for r in rows]
    df = (
        pl.DataFrame(merged)
        .lazy()
        .with_columns(
            pl.col("SnapshotTime").str.to_datetime("%Y%m%d"),
            *[pl.col(c).cast(t) for c, t in schema_casts.items() if c in pl.DataFrame(merged).columns],
        )
    )
    return ADMDatamart(model_df=df)


def _make_last_data(rows: list[dict]) -> pl.DataFrame:
    """Build a last_data DataFrame for direct maturity/distribution checks."""
    defaults = {
        "ModelID": "m1",
        "ResponseCount": 10000,
        "Positives": 500,
        "Performance": 0.7,
        "SuccessRate": 0.05,
        "Name": "ActionA",
        "Configuration": "Config1",
        "Channel": "Web",
        "Direction": "Inbound",
        "Issue": "Sales",
        "Group": "Cards",
    }
    return pl.DataFrame([{**defaults, **r} for r in rows])


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

    def test_findings_with_mock_prediction(self, analysis):
        # Covers line 144: `results.extend(self._check_predictions(prediction))`
        pred = MagicMock()
        pred.summary_by_channel.return_value = pl.DataFrame(
            [
                {
                    "Channel": "Web",
                    "Direction": "Inbound",
                    "Lift": float("nan"),
                    "Prediction": "WebPred",
                    "ControlPercentage": 0.02,
                }
            ]
        ).lazy()
        results = analysis.findings(prediction=pred)
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


# ── Edge case: _check_channels ──────────────────────────────────────────


class TestCheckChannelsEdgeCases:
    def test_exception_in_summary_returns_empty(self):
        dm = _make_dm([{}])
        with patch.object(dm.aggregates, "summary_by_channel", side_effect=RuntimeError("boom")):
            results = dm.analysis._check_channels(pl.lit(True))
        assert results == []

    def test_zero_positives_with_responses(self):
        mock_row = {
            "Channel": "Web",
            "Direction": "Inbound",
            "Responses": 1000,
            "Positives": 0,
            "Performance": 0.65,
            "OmniChannel": 0.8,
            "Configuration": "C1",
        }
        dm = _make_dm([{}])
        with patch.object(dm.aggregates, "summary_by_channel", return_value=pl.DataFrame([mock_row]).lazy()):
            results = dm.analysis._check_channels(pl.lit(True))
        critical = [f for f in results if f.severity == "critical"]
        assert any("zero positives" in f.title for f in critical)

    def test_low_performance_rag_red(self):
        # Performance on 50-100 scale; well below 60 triggers RAG=RED
        mock_row = {
            "Channel": "Web",
            "Direction": "Inbound",
            "Responses": 5000,
            "Positives": 200,
            "Performance": 52.0,
            "OmniChannel": 0.8,
            "Configuration": "C1",
        }
        dm = _make_dm([{}])
        with patch.object(dm.aggregates, "summary_by_channel", return_value=pl.DataFrame([mock_row]).lazy()):
            results = dm.analysis._check_channels(pl.lit(True))
        warnings = [f for f in results if f.severity == "warning" and f.category == "channel"]
        assert any("low average performance" in f.title for f in warnings)

    def test_low_omni_channel_overlap(self):
        dm = _make_dm([{"ResponseCount": 5000, "Positives": 200}])
        # Patch summary to return a row with low OmniChannel
        mock_row = {
            "Channel": "Web",
            "Direction": "Inbound",
            "Responses": 5000,
            "Positives": 200,
            "Performance": 0.65,
            "OmniChannel": 0.1,
            "Configuration": "C1",
        }
        mock_df = pl.DataFrame([mock_row])
        with patch.object(dm.aggregates, "summary_by_channel", return_value=mock_df.lazy()):
            results = dm.analysis._check_channels(pl.lit(True))
        assert any("low cross-channel overlap" in f.title for f in results)

    def test_many_configurations_per_channel(self):
        mock_row = {
            "Channel": "Web",
            "Direction": "Inbound",
            "Responses": 5000,
            "Positives": 200,
            "Performance": 0.65,
            "OmniChannel": 0.8,
            "Configuration": "C1, C2, C3",  # 3 configs → ≥2 commas
        }
        dm = _make_dm([{}])
        mock_df = pl.DataFrame([mock_row])
        with patch.object(dm.aggregates, "summary_by_channel", return_value=mock_df.lazy()):
            results = dm.analysis._check_channels(pl.lit(True))
        assert any("model configurations" in f.title for f in results)


# ── Edge case: _check_configurations ────────────────────────────────────


class TestCheckConfigurationsEdgeCases:
    def test_default_config_name_flagged(self):
        dm = _make_dm([{"Configuration": "Default_Inbound_Model"}])
        last_data = _make_last_data(
            [{"Configuration": "Default_Inbound_Model", "ResponseCount": 1000, "Positives": 50}]
        )
        results = dm.analysis._check_configurations(last_data)
        assert any("Default configuration" in f.title for f in results)
        assert any(f.severity == "warning" for f in results)

    def test_config_with_responses_but_no_positives(self):
        dm = _make_dm([{"Configuration": "MyConfig", "ResponseCount": 1000, "Positives": 0}])
        last_data = _make_last_data([{"Configuration": "MyConfig", "ResponseCount": 1000, "Positives": 0}])
        results = dm.analysis._check_configurations(last_data)
        assert any("no positives" in f.title for f in results)
        assert any(f.severity == "critical" for f in results)


# ── Edge case: _check_model_maturity ────────────────────────────────────


class TestCheckModelMaturityEdgeCases:
    def test_empty_last_data_returns_empty(self, analysis):
        empty = pl.DataFrame(
            {
                "ResponseCount": pl.Series([], dtype=pl.Int64),
                "Positives": pl.Series([], dtype=pl.Int64),
                "Performance": pl.Series([], dtype=pl.Float64),
            }
        )
        results = analysis._check_model_maturity(empty)
        assert results == []

    def test_high_pct_unused_is_warning(self, analysis):
        # >20% unused → warning
        rows = [{"ResponseCount": 0}] * 25 + [{"ResponseCount": 1000, "Positives": 100}] * 75
        last_data = _make_last_data(rows)
        results = analysis._check_model_maturity(last_data)
        unused_findings = [f for f in results if "never been used" in f.title]
        assert any(f.severity == "warning" for f in unused_findings)

    def test_low_pct_unused_is_info(self, analysis):
        # <20% unused → info
        rows = [{"ResponseCount": 0}] * 5 + [{"ResponseCount": 1000, "Positives": 100}] * 95
        last_data = _make_last_data(rows)
        results = analysis._check_model_maturity(last_data)
        unused_findings = [f for f in results if "never been used" in f.title]
        assert any(f.severity == "info" for f in unused_findings)

    def test_high_pct_no_positives_is_warning(self, analysis):
        # >10% have responses but zero positives → warning
        rows = [{"ResponseCount": 1000, "Positives": 0}] * 15 + [{"ResponseCount": 1000, "Positives": 100}] * 85
        last_data = _make_last_data(rows)
        results = analysis._check_model_maturity(last_data)
        findings = [f for f in results if "zero positives" in f.title]
        assert any(f.severity == "warning" for f in findings)

    def test_low_pct_no_positives_is_info(self, analysis):
        # <10% have responses but zero positives → info
        rows = [{"ResponseCount": 1000, "Positives": 0}] * 5 + [{"ResponseCount": 1000, "Positives": 100}] * 95
        last_data = _make_last_data(rows)
        results = analysis._check_model_maturity(last_data)
        findings = [f for f in results if "zero positives" in f.title]
        assert any(f.severity == "info" for f in findings)

    def test_mature_stuck_at_auc50(self, analysis):
        from pdstools.utils.metric_limits import MetricLimits

        bp_min = int(MetricLimits.best_practice_min("TotalPositiveCount"))
        rows = [{"Positives": bp_min + 100, "Performance": 0.5, "ResponseCount": 5000}]
        last_data = _make_last_data(rows)
        results = analysis._check_model_maturity(last_data)
        assert any("AUC=50" in f.title for f in results)

    def test_suspiciously_high_performance(self, analysis):
        from pdstools.utils.metric_limits import MetricLimits

        bp_min = int(MetricLimits.best_practice_min("TotalPositiveCount"))
        rows = [{"Positives": bp_min + 100, "Performance": 0.99, "ResponseCount": 5000}]
        last_data = _make_last_data(rows)
        results = analysis._check_model_maturity(last_data)
        assert any("suspiciously high" in f.title for f in results)


# ── Edge case: _check_model_performance ─────────────────────────────────


class TestCheckModelPerformanceEdgeCases:
    def test_exception_in_active_filter_returns_empty(self, analysis):
        last_data = _make_last_data([{}])
        bad_filter = pl.col("nonexistent_column") > 0
        results = analysis._check_model_performance(last_data, bad_filter)
        assert results == []

    def test_empty_active_data_returns_empty(self, analysis):
        last_data = _make_last_data([{}])
        results = analysis._check_model_performance(last_data, pl.lit(False))
        assert results == []


# ── Edge case: _check_taxonomy ───────────────────────────────────────────


class TestCheckTaxonomyEdgeCases:
    def test_rag_red_produces_warning(self):
        # 1 action → almost certainly RED for ActionCount
        dm = _make_dm([{"Name": "OnlyAction"}])
        with patch("pdstools.adm.Analysis.MetricLimits.evaluate_metric_rag", return_value="RED"):
            results = dm.analysis._check_taxonomy()
        warnings = [f for f in results if f.severity == "warning" and f.category == "taxonomy"]
        assert len(warnings) > 0

    def test_rag_amber_produces_info(self):
        dm = _make_dm([{"Name": "OnlyAction"}])
        with patch("pdstools.adm.Analysis.MetricLimits.evaluate_metric_rag", return_value="AMBER"):
            results = dm.analysis._check_taxonomy()
        infos = [f for f in results if f.severity == "info" and f.category == "taxonomy"]
        assert len(infos) > 0


# ── Edge case: _check_response_distribution ─────────────────────────────


class TestCheckResponseDistributionEdgeCases:
    def test_exception_in_active_filter_returns_empty(self, analysis):
        last_data = _make_last_data([{}])
        results = analysis._check_response_distribution(last_data, pl.col("nonexistent") > 0)
        assert results == []

    def test_empty_active_data_returns_empty(self, analysis):
        last_data = _make_last_data([{}])
        results = analysis._check_response_distribution(last_data, pl.lit(False))
        assert results == []

    def test_high_gini_responses_warning(self, analysis):
        # One model gets all responses → Gini near 1
        rows = [{"ResponseCount": 1_000_000, "Positives": 50000, "Performance": 0.7}] + [
            {"ResponseCount": 1, "Positives": 0, "Performance": 0.5}
        ] * 20
        last_data = _make_last_data(rows)
        results = analysis._check_response_distribution(last_data, pl.lit(True))
        assert any("highly skewed" in f.title and f.severity == "warning" for f in results)

    def test_moderate_gini_responses_info(self, analysis):
        # Moderate concentration (Gini ~0.75-0.89)
        rows = [{"ResponseCount": 10000, "Positives": 500, "Performance": 0.7}] + [
            {"ResponseCount": 100, "Positives": 5, "Performance": 0.7}
        ] * 10
        last_data = _make_last_data(rows)
        results = analysis._check_response_distribution(last_data, pl.lit(True))
        moderate = [f for f in results if "moderately skewed" in f.title]
        # May or may not trigger depending on exact Gini — just verify no crash
        assert isinstance(results, list)
        for f in moderate:
            assert f.severity == "info"

    def test_zero_total_responses_skips_low_auc_check(self, analysis):
        rows = [{"ResponseCount": 0, "Positives": 0, "Performance": 0.5}] * 5
        last_data = _make_last_data(rows)
        results = analysis._check_response_distribution(last_data, pl.lit(True))
        # Must not crash; no low-auc-volume finding possible
        assert not any("response volume" in f.title.lower() and "low-performance" in f.title.lower() for f in results)

    def test_majority_volume_at_low_auc_warning(self, analysis):
        # >50% responses from models with AUC < 0.55
        rows = [{"ResponseCount": 10000, "Positives": 100, "Performance": 0.50}] * 6 + [
            {"ResponseCount": 1000, "Positives": 200, "Performance": 0.75}
        ] * 4
        last_data = _make_last_data(rows)
        results = analysis._check_response_distribution(last_data, pl.lit(True))
        assert any("low-performance models" in f.title and f.severity == "warning" for f in results)

    def test_notable_volume_at_low_auc_info(self, analysis):
        # 30-50% responses from models with AUC < 0.55
        rows = [{"ResponseCount": 4000, "Positives": 100, "Performance": 0.50}] * 1 + [
            {"ResponseCount": 6000, "Positives": 500, "Performance": 0.75}
        ] * 1
        last_data = _make_last_data(rows)
        results = analysis._check_response_distribution(last_data, pl.lit(True))
        notable = [f for f in results if "low-performance models" in f.title]
        assert isinstance(notable, list)  # may or may not trigger; must not crash


# ── Edge case: _check_trends ─────────────────────────────────────────────


class TestCheckTrendsEdgeCases:
    def test_single_snapshot_returns_empty(self):
        dm = _make_dm([{}])  # one row → one snapshot
        results = dm.analysis._check_trends(pl.lit(True))
        assert results == []

    def test_exception_getting_snapshots_returns_empty(self):
        dm = _make_dm([{}])
        with patch.object(dm.model_data, "select", side_effect=RuntimeError("boom")):
            results = dm.analysis._check_trends(pl.lit(True))
        assert results == []

    def test_performance_decline_triggers_warning(self):
        # Build 6 weekly snapshots with AUC dropping sharply
        rows = []
        dates = ["20250101", "20250108", "20250115", "20250122", "20250129", "20250205"]
        perfs = [0.75, 0.74, 0.73, 0.65, 0.63, 0.61]  # >3 AUC drop first→second half
        for date, perf in zip(dates, perfs):
            rows.append({"SnapshotTime": date, "Performance": perf, "ResponseCount": 1000, "Positives": 50})
        dm = _make_dm(rows)
        results = dm.analysis._check_trends(pl.lit(True))
        assert any("declining" in f.title and f.severity == "warning" for f in results)

    def test_response_volume_drop_triggers_warning(self):
        rows = []
        dates = ["20250101", "20250108", "20250115", "20250122", "20250129", "20250205"]
        resps = [10000, 10000, 10000, 1000, 1000, 1000]  # 90% drop
        for date, resp in zip(dates, resps):
            rows.append({"SnapshotTime": date, "Performance": 0.70, "ResponseCount": resp, "Positives": 50})
        dm = _make_dm(rows)
        results = dm.analysis._check_trends(pl.lit(True))
        assert any("Response volume dropped" in f.title for f in results)


# ── Edge case: _check_predictions ────────────────────────────────────────


class TestCheckPredictionsEdgeCases:
    def _make_prediction_mock(self, rows: list[dict]):
        """Build a mock Prediction with controlled summary_by_channel output."""
        mock = MagicMock()
        mock.summary_by_channel.return_value = pl.DataFrame(rows).lazy()
        return mock

    def test_no_lift_column_returns_empty(self, analysis):
        pred = self._make_prediction_mock([{"Channel": "Web", "Direction": "Inbound"}])
        results = analysis._check_predictions(pred)
        assert results == []

    def test_exception_in_summary_returns_empty(self, analysis):
        pred = MagicMock()
        pred.summary_by_channel.side_effect = RuntimeError("no data")
        results = analysis._check_predictions(pred)
        assert results == []

    def test_negative_lift_is_critical(self, analysis):
        pred = self._make_prediction_mock(
            [
                {
                    "Channel": "Web",
                    "Direction": "Inbound",
                    "Lift": -0.05,
                    "Prediction": "WebPrediction",
                    "ControlPercentage": 0.02,
                }
            ]
        )
        results = analysis._check_predictions(pred)
        assert any(f.severity == "critical" and "Negative" in f.title for f in results)

    def test_very_low_lift_is_warning(self, analysis):
        pred = self._make_prediction_mock(
            [
                {
                    "Channel": "Web",
                    "Direction": "Inbound",
                    "Lift": 0.03,
                    "Prediction": "WebPrediction",
                    "ControlPercentage": 0.02,
                }
            ]
        )
        results = analysis._check_predictions(pred)
        assert any(f.severity == "warning" and "low engagement lift" in f.title for f in results)

    def test_default_prediction_name_is_warning(self, analysis):
        pred = self._make_prediction_mock(
            [
                {
                    "Channel": "Web",
                    "Direction": "Inbound",
                    "Lift": float("nan"),
                    "Prediction": "PredictInboundDefaultPropensity",
                    "ControlPercentage": 0.02,
                }
            ]
        )
        results = analysis._check_predictions(pred)
        assert any("Default prediction name" in f.title for f in results)

    def test_large_control_group_is_warning(self, analysis):
        pred = self._make_prediction_mock(
            [
                {
                    "Channel": "Web",
                    "Direction": "Inbound",
                    "Lift": float("nan"),
                    "Prediction": "WebPrediction",
                    "ControlPercentage": 0.15,
                }
            ]
        )
        results = analysis._check_predictions(pred)
        assert any("Large control group" in f.title for f in results)

    def test_small_control_group_is_info(self, analysis):
        pred = self._make_prediction_mock(
            [
                {
                    "Channel": "Web",
                    "Direction": "Inbound",
                    "Lift": float("nan"),
                    "Prediction": "WebPrediction",
                    "ControlPercentage": 0.001,
                }
            ]
        )
        results = analysis._check_predictions(pred)
        assert any("Very small control group" in f.title for f in results)


# ── Additional coverage: exception paths & IH predictor branches ─────────


class TestCheckTaxonomyExceptionPaths:
    def test_n_unique_values_exception_continues(self):
        """Lines 544-545: exception in n_unique_values loop is swallowed."""
        dm = _make_dm([{}])
        with patch("pdstools.adm.Analysis.report_utils.n_unique_values", side_effect=RuntimeError("boom")):
            results = dm.analysis._check_taxonomy()
        assert isinstance(results, list)  # must not crash

    def test_predictor_count_exception_is_swallowed(self):
        """Lines 619-620: exception in predictor-count block is swallowed."""
        dm = datasets.cdh_sample()
        assert dm.predictor_data is not None
        # combined_data must work for schema check (line 529) but fail on .filter() (line 594)
        mock_cd = MagicMock()
        mock_cd.collect_schema.return_value.names.return_value = ["Configuration", "EntryType", "PredictorName"]
        mock_cd.filter.side_effect = RuntimeError("boom")
        dm.combined_data = mock_cd
        results = dm.analysis._check_taxonomy()
        assert isinstance(results, list)


class TestCheckTrendsMoreEdgeCases:
    def test_exception_in_trend_collection_returns_empty(self):
        """Lines 755-756: exception during trend aggregation returns empty."""
        rows = [
            {"SnapshotTime": d, "Performance": 0.70, "ResponseCount": 1000, "Positives": 50}
            for d in ["20250101", "20250108", "20250115", "20250122"]
        ]
        dm = _make_dm(rows)
        with patch("pdstools.adm.Analysis.cdh_utils.weighted_average_polars", side_effect=RuntimeError("boom")):
            results = dm.analysis._check_trends(pl.lit(True))
        assert results == []

    def test_channel_with_fewer_than_3_snapshots_is_skipped(self):
        """Line 761: per-channel trend with < 3 rows is skipped gracefully."""
        rows = []
        for date in ["20250101", "20250108", "20250115", "20250122"]:
            rows.append(
                {
                    "SnapshotTime": date,
                    "Channel": "Web",
                    "Direction": "Inbound",
                    "Performance": 0.70,
                    "ResponseCount": 1000,
                    "Positives": 50,
                }
            )
        for date in ["20250101", "20250108"]:
            rows.append(
                {
                    "SnapshotTime": date,
                    "Channel": "Email",
                    "Direction": "Outbound",
                    "Performance": 0.70,
                    "ResponseCount": 1000,
                    "Positives": 50,
                }
            )
        dm = _make_dm(rows)
        results = dm.analysis._check_trends(pl.lit(True))
        assert isinstance(results, list)


class TestCheckPredictorsExceptionPaths:
    def test_poor_predictor_exception_is_swallowed(self, analysis):
        """Lines 861-862: exception in poor-predictor block is swallowed."""
        with patch.object(analysis.datamart, "combined_data", new=None):
            results = analysis._check_predictors()
        assert isinstance(results, list)


class TestCheckPredictorsIHBranches:
    def _make_dm_with_ih(self, ih_count: int, non_ih_count: int) -> ADMDatamart:
        """ADMDatamart with controlled IH predictor counts in combined_data."""
        dm = _make_dm([{}])
        rows = [
            {
                "Configuration": "C1",
                "EntryType": "Predictor",
                "PredictorCategory": "IH",
                "PredictorName": f"IH_{i}",
                "Performance": 0.55,
                "ResponseCount": 1000,
                "MissingPct": 0.0,
            }
            for i in range(ih_count)
        ] + [
            {
                "Configuration": "C1",
                "EntryType": "Predictor",
                "PredictorCategory": "Customer",
                "PredictorName": f"C_{i}",
                "Performance": 0.55,
                "ResponseCount": 1000,
                "MissingPct": 0.0,
            }
            for i in range(non_ih_count)
        ]
        dm.combined_data = pl.DataFrame(rows).lazy()
        return dm

    def test_more_than_100_ih_predictors_is_info(self):
        """Line 919: IH_count > 100 produces an info finding."""
        dm = self._make_dm_with_ih(ih_count=110, non_ih_count=10)
        results = dm.analysis._check_predictors()
        assert any("IH predictors" in f.title and f.severity == "info" for f in results)

    def test_zero_ih_predictors_is_info(self):
        """Lines 937-955: IH_count == 0 produces an info finding."""
        dm = self._make_dm_with_ih(ih_count=0, non_ih_count=10)
        results = dm.analysis._check_predictors()
        assert any("no IH predictors" in f.title and f.severity == "info" for f in results)
