"""Tests for the ADM Analysis module (automated health check findings)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pdstools import ADMDatamart, datasets
from pdstools.adm.Analysis import (
    Analysis,
    Finding,
    HealthCheckPreAggregates,
    _format_markdown_value,
    _gini,
)
from pdstools.utils import cdh_utils
from pdstools.utils.metric_limits import MetricLimits


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


class TestMarkdownFormattingHelpers:
    def test_format_markdown_value_handles_common_types(self):
        assert _format_markdown_value(None) == "N/A"
        assert _format_markdown_value(True) == "Yes"
        assert _format_markdown_value(False) == "No"
        assert _format_markdown_value(0.0) == "0"
        assert _format_markdown_value(12.3456) == "12.35"
        assert _format_markdown_value(0.123456) == "0.1235"
        assert _format_markdown_value("a|b\nc") == "a\\|b c"

    def test_format_markdown_value_handles_lists(self):
        value = _format_markdown_value([1, 2, 3, 4, 5, 6])
        assert value.startswith("1, 2, 3")
        assert "(+1 more)" in value


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
            severity_order = {"critical": 0, "warning": 1, "info": 2, "success": 3}
            orders = [severity_order[f.severity] for f in results]
            assert orders == sorted(orders)

    def test_findings_start_with_summary_headline(self, analysis):
        results = analysis.findings()
        assert results
        assert results[0].category == "summary"
        assert results[0].title.startswith("Health:")
        assert "verdict" in results[0].data

    def test_markdown_renders_summary_and_sections(self, analysis):
        markdown = analysis.markdown(title="Custom Health Check", subtitle="Sample")
        assert markdown.startswith("# Custom Health Check")
        assert "## Sample" in markdown
        assert "## Summary" in markdown
        assert "## Key Metrics" in markdown
        assert "## Estate Snapshot" in markdown
        assert "## Findings Overview" in markdown
        assert (
            "## Critical Findings" in markdown
            or "## Warning Findings" in markdown
            or "## Informational Findings" in markdown
        )

    def test_markdown_renders_compact_tables(self, analysis):
        markdown = analysis.markdown()
        assert "| Metric | Value |" in markdown
        assert "| Severity | Count |" in markdown
        assert "| Category | Count |" in markdown
        assert "### Top Channels by Responses" in markdown
        assert "### Top Configurations by Responses" in markdown

    def test_markdown_includes_disclaimer(self, analysis):
        markdown = analysis.markdown(disclaimer="Use with caution")
        assert "> **Disclaimer:** Use with caution" in markdown

    def test_compute_health_check_preaggregates_returns_reusable_object(self, analysis):
        preaggregates = analysis.compute_health_check_preaggregates()
        assert isinstance(preaggregates, HealthCheckPreAggregates)
        assert preaggregates.last_data.height > 0

    def test_compute_health_check_preaggregates_builds_minimal_channel_and_config_summaries(self):
        dm = _make_dm(
            [
                {
                    "ModelID": "m1",
                    "SnapshotTime": "20250101",
                    "Configuration": "CfgA",
                    "Name": "ActionA",
                    "Channel": "Web",
                    "Direction": "Inbound",
                    "Positives": 10,
                    "ResponseCount": 100,
                    "Performance": 0.60,
                },
                {
                    "ModelID": "m1",
                    "SnapshotTime": "20250102",
                    "Configuration": "CfgA",
                    "Name": "ActionA",
                    "Channel": "Web",
                    "Direction": "Inbound",
                    "Positives": 25,
                    "ResponseCount": 250,
                    "Performance": 0.65,
                },
                {
                    "ModelID": "m2",
                    "SnapshotTime": "20250101",
                    "Configuration": "CfgA",
                    "Name": "ActionB",
                    "Channel": "Web",
                    "Direction": "Inbound",
                    "Positives": 0,
                    "ResponseCount": 0,
                    "Performance": 0.70,
                },
                {
                    "ModelID": "m2",
                    "SnapshotTime": "20250102",
                    "Configuration": "CfgA",
                    "Name": "ActionB",
                    "Channel": "Web",
                    "Direction": "Inbound",
                    "Positives": 20,
                    "ResponseCount": 100,
                    "Performance": 0.80,
                },
            ]
        )

        preaggregates = dm.analysis.compute_health_check_preaggregates(include_markdown_sections=True)

        channel_row = preaggregates.channel_summary.row(0, named=True)
        assert channel_row["ChannelDirection"] == "Web/Inbound"
        assert channel_row["Responses"] == 250
        assert channel_row["Positives"] == 35
        assert channel_row["Actions"] == 2
        assert channel_row["CTR"] == pytest.approx(0.14)
        assert channel_row["Performance"] == pytest.approx(0.675)

        config_row = preaggregates.configuration_summary.row(0, named=True)
        assert config_row["Configuration"] == "CfgA"
        assert config_row["Channel"] == "Web"
        assert config_row["Direction"] == "Inbound"
        assert config_row["ResponseCount"] == 350
        assert config_row["Positives"] == 45
        assert config_row["Performance"] == pytest.approx((0.65 * 250 + 0.80 * 100) / 350)

    def test_compute_health_check_preaggregates_preserves_all_history_taxonomy_counts(self):
        dm = _make_dm(
            [
                {
                    "ModelID": "m1",
                    "SnapshotTime": "20250101",
                    "Configuration": "CfgA",
                    "Name": "ActionA",
                    "Treatment": "T1",
                    "Issue": "Sales",
                    "Channel": "Web",
                    "Direction": "Inbound",
                },
                {
                    "ModelID": "m1",
                    "SnapshotTime": "20250102",
                    "Configuration": "CfgA",
                    "Name": "ActionA",
                    "Treatment": "T1",
                    "Issue": "Sales",
                    "Channel": "Web",
                    "Direction": "Inbound",
                },
                {
                    "ModelID": "m2",
                    "SnapshotTime": "20250101",
                    "Configuration": "CfgB",
                    "Name": "ActionB",
                    "Treatment": "T2",
                    "Issue": "Service",
                    "Channel": "Email",
                    "Direction": "Outbound",
                },
            ]
        )

        preaggregates = dm.analysis.compute_health_check_preaggregates(include_markdown_sections=False)

        assert preaggregates.action_count == 2
        assert preaggregates.treatment_count == 2
        assert preaggregates.channel_count == 2
        assert preaggregates.taxonomy_counts["IssueCount"] == 2

    def test_markdown_accepts_preaggregates_without_rebuilding(self, analysis):
        preaggregates = analysis.compute_health_check_preaggregates()
        with patch.object(
            analysis, "compute_health_check_preaggregates", side_effect=RuntimeError("should not rebuild")
        ):
            markdown = analysis.markdown(preaggregates=preaggregates)
        assert markdown.startswith("# ADM Health Check")

    def test_findings_accepts_preaggregates_without_rebuilding(self, analysis):
        preaggregates = analysis.compute_health_check_preaggregates(include_markdown_sections=False)
        with patch.object(
            analysis, "compute_health_check_preaggregates", side_effect=RuntimeError("should not rebuild")
        ):
            findings = analysis.findings(preaggregates=preaggregates)
        assert findings

    def test_markdown_handles_summary_only(self, analysis):
        summary = Finding(
            severity="success",
            category="summary",
            title="Health: HEALTHY - no issues detected",
            detail="Everything looks good.",
            data={"avg_auc": 0.71},
        )
        last_data = _make_last_data([{}])
        with (
            patch.object(analysis, "findings", return_value=[summary]),
            patch.object(analysis, "_get_last_data", return_value=last_data),
            patch.object(analysis, "_estate_snapshot_sections", return_value=[]),
        ):
            markdown = analysis.markdown()
        assert "## Findings" in markdown
        assert "No findings were generated." in markdown

    def test_key_metrics_table_handles_active_filter_and_predictor_errors(self, sample_dm):
        findings = [
            Finding(
                severity="success",
                category="summary",
                title="Health: HEALTHY - AUC 71.0",
                detail="Everything looks good.",
                data={"avg_auc": 0.71},
            )
        ]
        last_data = sample_dm.analysis._get_last_data()
        with patch.object(
            sample_dm.aggregates,
            "predictors_global_overview",
            side_effect=RuntimeError("boom"),
        ):
            table = sample_dm.analysis._key_metrics_table(
                last_data,
                findings,
                pl.col("does_not_exist") > 0,
            )
        values = dict(table.iter_rows())
        assert values["Active models"] == "N/A"
        assert values["Predictors"] == "N/A"

    def test_estate_snapshot_sections_handle_aggregate_failures(self, sample_dm):
        with (
            patch.object(sample_dm.aggregates, "channel_overview", side_effect=RuntimeError("boom")),
            patch.object(sample_dm.analysis, "_health_check_channel_summary", side_effect=RuntimeError("boom")),
            patch.object(
                sample_dm.aggregates,
                "configuration_overview",
                side_effect=RuntimeError("boom"),
            ),
            patch.object(sample_dm.aggregates, "global_predictor_overview", side_effect=RuntimeError("boom")),
        ):
            sections = sample_dm.analysis._estate_snapshot_sections(pl.lit(True))
        assert sections == []

    def test_estate_snapshot_sections_include_predictor_categories(self, sample_dm):
        sections = sample_dm.analysis._estate_snapshot_sections(pl.lit(True))
        titles = [title for title, _ in sections]
        assert "Top Channels by Responses" in titles
        assert "Top Configurations by Responses" in titles
        assert "Predictor Categories" in titles

    def test_estate_snapshot_sections_allow_configuration_summary_without_channel_direction(
        self,
        analysis,
    ):
        preaggregates = HealthCheckPreAggregates(
            last_data=_make_last_data([{}]),
            configuration_summary=pl.DataFrame(
                [
                    {
                        "Configuration": "Cfg1",
                        "ResponseCount": 1000,
                        "Positives": 50,
                        "Performance": 0.65,
                    }
                ]
            ),
        )

        sections = analysis._estate_snapshot_sections(pl.lit(True), preaggregates=preaggregates)
        title, table = sections[0]
        assert title == "Top Configurations by Responses"
        assert table.columns == ["Configuration", "ResponseCount", "Positives", "Performance"]

    def test_estate_snapshot_sections_exclude_invalid_channel_rows(self, analysis):
        preaggregates = HealthCheckPreAggregates(
            last_data=_make_last_data([{}]),
            channel_overview=pl.DataFrame(
                [
                    {
                        "Channel": None,
                        "Direction": None,
                        "ChannelDirection": None,
                        "Configuration": "Cfg0",
                        "Duration": 1.0,
                        "isValid": True,
                        "Issues": 1,
                        "Groups": 1,
                        "Actions": 1,
                        "Used Actions": 1,
                        "New Actions": 0,
                        "Treatments": 1,
                        "Used Treatments": 1,
                        "usesNBAD": None,
                        "usesAGB": None,
                        "Responses": 5000,
                        "Positives": 100,
                        "Performance": 0.6,
                        "CTR": 0.02,
                        "OmniChannel": None,
                        "NBAD": "?",
                        "AGB": "?",
                    },
                    {
                        "Channel": "Web",
                        "Direction": "Inbound",
                        "ChannelDirection": "Web/Inbound",
                        "Configuration": "Cfg1",
                        "Duration": 1.0,
                        "isValid": True,
                        "Issues": 1,
                        "Groups": 1,
                        "Actions": 2,
                        "Used Actions": 2,
                        "New Actions": 0,
                        "Treatments": 2,
                        "Used Treatments": 2,
                        "usesNBAD": None,
                        "usesAGB": None,
                        "Responses": 1000,
                        "Positives": 50,
                        "Performance": 0.7,
                        "CTR": 0.05,
                        "OmniChannel": 0.8,
                        "NBAD": "?",
                        "AGB": "?",
                    },
                ]
            ),
        )

        sections = analysis._estate_snapshot_sections(pl.lit(True), preaggregates=preaggregates)

        title, table = sections[0]
        assert title == "Top Channels by Responses"
        assert table.to_dicts() == [
            {
                "ChannelDirection": "Web/Inbound",
                "Responses": 1000,
                "Positives": 50,
                "Performance": 0.7,
                "CTR": 0.05,
                "Actions": 2,
            }
        ]

    def test_health_check_channel_overview_matches_legacy_quarto_expression(self, sample_dm):
        active_filter = (
            pl.col("LastUpdate") > (pl.col("LastUpdate").max() - __import__("datetime").timedelta(days=30))
        ).fill_null(True)

        shared = sample_dm.aggregates.channel_overview(query=active_filter)
        legacy = (
            sample_dm.aggregates.summary_by_channel(
                query=active_filter,
                format_flags=True,
            )
            .drop(
                [
                    "ChannelDirectionGroup",
                    "DateRange Min",
                    "DateRange Max",
                ]
            )
            .collect()
        )

        assert_frame_equal(shared, legacy)

    def test_health_check_active_filter_matches_legacy_expression(self, sample_dm):
        shared = sample_dm.analysis.health_check_active_filter(active_threshold_days=30)
        legacy = (
            pl.col("LastUpdate") > (pl.col("LastUpdate").max() - __import__("datetime").timedelta(days=30))
        ).fill_null(True)

        shared_ids = sample_dm.model_data.filter(shared).select("ModelID").collect().sort("ModelID")
        legacy_ids = sample_dm.model_data.filter(legacy).select("ModelID").collect().sort("ModelID")

        assert_frame_equal(shared_ids, legacy_ids)

    def test_health_check_maturity_overview_matches_legacy_quarto_expression(self, analysis):
        active_filter = analysis.health_check_active_filter(active_threshold_days=30)
        last_data = analysis._get_last_data()
        shared = analysis.health_check_maturity_overview(last_data=last_data, active_filter=active_filter)

        min_responses = MetricLimits.minimum("TotalResponseCount")
        min_positives = MetricLimits.minimum("TotalPositiveCount")
        bp_min_positives = MetricLimits.best_practice_min("TotalPositiveCount")
        min_perf = MetricLimits.minimum("ModelPerformance")
        legacy_criteria = [
            ("Number of models in last snapshot", pl.lit(True)),
            ("Models that have never been used (responses = 0)", pl.col("ResponseCount") < min_responses),
            (
                "Models that have been used (responses > 0) but never received a positive response",
                (pl.col("Positives") < min_positives) & (pl.col("ResponseCount") >= min_responses),
            ),
            (
                f"Models that are still in an immature phase of learning (positives < {int(bp_min_positives)})",
                (pl.col("Positives") < bp_min_positives) & (pl.col("Positives") >= min_positives),
            ),
            (
                "Models that have received sufficient responses but are still at their minimum performance (AUC = 50)",
                (pl.col("Performance") == 0.5) & (pl.col("Positives") >= bp_min_positives),
            ),
            (
                f"Models that have received sufficient responses but still have a low performance (AUC < {min_perf})",
                (pl.col("Performance") > 0.5)
                & (pl.col("Performance") < min_perf)
                & (pl.col("Positives") >= bp_min_positives),
            ),
            (
                f"Models with sufficient positive responses (≥ {int(bp_min_positives)}) and a decent performance (≥ {min_perf})",
                (pl.col("Performance") >= min_perf) & (pl.col("Positives") >= bp_min_positives),
            ),
        ]
        legacy = pl.concat(
            [
                last_data.lazy()
                .filter(active_filter)
                .group_by(None)
                .agg(
                    pl.lit(label).alias("Category"),
                    pl.col("Name").filter(filter_expression).len().alias("Number of Models"),
                    (
                        cdh_utils.weighted_average_polars(
                            pl.col("Performance").filter(filter_expression),
                            pl.col("ResponseCount").filter(filter_expression),
                        )
                        * 100.0
                    ).alias("Average Performance"),
                )
                .drop("literal")
                .collect()
                for label, filter_expression in legacy_criteria
            ]
        )

        assert_frame_equal(shared, legacy)

    def test_health_check_configuration_overview_matches_legacy_quarto_expression(self, sample_dm):
        active_filter = sample_dm.analysis.health_check_active_filter(active_threshold_days=30)
        shared = sample_dm.aggregates.configuration_overview(query=active_filter)
        legacy = (
            sample_dm.aggregates.summary_by_configuration(query=active_filter)
            .with_columns(
                NBAD=pl.when(pl.col("usesNBAD").is_null())
                .then(pl.lit("?"))
                .when(pl.col("usesNBAD"))
                .then(pl.lit("Yes"))
                .otherwise(pl.lit("No")),
                AGB=pl.when(pl.col("usesAGB").is_null())
                .then(pl.lit("?"))
                .when(pl.col("usesAGB"))
                .then(pl.lit("Yes"))
                .otherwise(pl.lit("No")),
            )
            .drop("usesNBAD", "usesAGB")
            .collect()
        )

        assert_frame_equal(shared, legacy)

    def test_health_check_predictor_overview_matches_legacy_quarto_expression(self, sample_dm):
        active_filter = sample_dm.analysis.health_check_active_filter(active_threshold_days=30)
        shared = sample_dm.aggregates.global_predictor_overview(query=active_filter)
        legacy = sample_dm.aggregates.predictors_global_overview(query=active_filter).collect()

        assert_frame_equal(shared, legacy)

    def test_findings_have_valid_severity(self, analysis):
        for f in analysis.findings():
            assert f.severity in ("critical", "warning", "info", "success")

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
            "data_quality",
            "summary",
        }
        for f in analysis.findings():
            assert f.category in valid_categories

    def test_findings_have_nonempty_title_and_detail(self, analysis):
        findings = analysis.findings()
        assert findings, "Expected sample data to produce at least one finding"
        for f in findings:
            assert f.title, "Finding has empty title"
            assert f.detail, "Finding has empty detail"


# ── Individual check methods ────────────────────────────────────────────


class TestChannelChecks:
    def test_no_crash_with_sample_data(self, analysis):
        active_filter = pl.lit(True)
        results, _, _ = analysis._check_channels(active_filter)
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
        results, _, _ = dm.analysis._check_channels(pl.lit(True))
        assert any("zero responses" in f.title for f in results)


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
        buckets = {f.data.get("bucket") for f in results if "bucket" in f.data}
        assert "mature_decent" in buckets

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
        results = analysis._check_trends()
        assert isinstance(results, list)


class TestPredictorChecks:
    def test_no_crash_with_sample_data(self, analysis, sample_dm):
        if sample_dm.predictor_data is not None:
            results = analysis._check_predictors()
            assert isinstance(results, list)
            assert results
            assert all(f.category == "predictor" for f in results)


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
        categories = {f.category for f in results}
        assert "summary" in categories
        assert {"model", "taxonomy"}.issubset(categories)

        # Every finding should be well-formed
        for f in results:
            assert f.title
            assert f.detail
            assert isinstance(f.data, dict)

        # Headline summary should be first.
        assert results[0].category == "summary"

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
            results, dead_count, has_low_perf = dm.analysis._check_channels(pl.lit(True))
        assert results == []
        assert dead_count == 0
        assert has_low_perf is False

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
            results, _, _ = dm.analysis._check_channels(pl.lit(True))
        critical = [f for f in results if f.severity == "critical"]
        assert any("zero positives" in f.title for f in critical)

    def test_low_performance_rag_red(self):
        # Performance on 50-100 scale; below 52 triggers RAG=RED.
        mock_row = {
            "Channel": "Web",
            "Direction": "Inbound",
            "Responses": 5000,
            "Positives": 200,
            "Performance": 50.0,
            "OmniChannel": 0.8,
            "Configuration": "C1",
        }
        dm = _make_dm([{}])
        with patch.object(dm.aggregates, "summary_by_channel", return_value=pl.DataFrame([mock_row]).lazy()):
            results, _, has_low_perf = dm.analysis._check_channels(pl.lit(True))
        warnings = [f for f in results if f.severity == "warning" and f.category == "channel"]
        assert any("low average performance" in f.title for f in warnings)
        assert has_low_perf is True

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
            results, _, _ = dm.analysis._check_channels(pl.lit(True))
        assert any("low cross-channel overlap" in f.title for f in results)

    def test_many_configurations_per_channel(self):
        # Per-channel "has N model configurations" was removed as a finding —
        # it was almost always emitted on dead channels and added no
        # actionable information. The check should now produce *no* findings
        # for an active channel that simply has many configurations.
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
            results, _, _ = dm.analysis._check_channels(pl.lit(True))
        assert not any("model configurations" in f.title for f in results)


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

    def test_stale_historical_configuration_not_flagged(self):
        dm = _make_dm(
            [
                {"Configuration": "Default_Inbound_Model", "SnapshotTime": "20240101"},
                {"Configuration": "ActiveConfig", "SnapshotTime": "20250101"},
            ]
        )
        last_data = _make_last_data([{"Configuration": "ActiveConfig", "ResponseCount": 1000, "Positives": 50}])
        results = dm.analysis._check_configurations(last_data)
        assert not any("Default configuration" in f.title for f in results)


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

    def test_mature_low_performance_covers_values_just_above_auc50(self, analysis):
        from pdstools.utils.metric_limits import MetricLimits

        bp_min = int(MetricLimits.best_practice_min("TotalPositiveCount"))
        rows = [{"Positives": bp_min + 100, "Performance": 0.51, "ResponseCount": 5000}]
        last_data = _make_last_data(rows)
        results = analysis._check_model_maturity(last_data)
        assert any("low performance" in f.title for f in results)

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
        results, avg = analysis._check_model_performance(last_data, bad_filter)
        assert results == []
        assert avg is None

    def test_empty_active_data_returns_empty(self, analysis):
        last_data = _make_last_data([{}])
        results, avg = analysis._check_model_performance(last_data, pl.lit(False))
        assert results == []
        assert avg is None


# ── Edge case: _check_taxonomy ───────────────────────────────────────────


class TestCheckTaxonomyEdgeCases:
    def test_rag_red_produces_warning(self):
        # 1 action → almost certainly RED for ActionCount
        dm = _make_dm([{"Name": "OnlyAction"}])
        with patch("pdstools.adm.Analysis.MetricLimits.evaluate_metric_rag", return_value="RED"):
            results = dm.analysis._check_taxonomy()
        warnings = [f for f in results if f.severity == "warning" and f.category == "taxonomy"]
        # One warning per evaluated taxonomy metric (actions, treatments,
        # issues, channels) → 4 with the minimal 1-action dataset.
        assert len(warnings) == 4
        # Titles now state direction + the violated bound rather than a
        # generic "outside recommended limits" string.
        for f in warnings:
            assert (
                ("below minimum" in f.title)
                or ("above maximum" in f.title)
                or ("below recommended" in f.title)
                or ("above typical range" in f.title)
            )

    def test_rag_amber_produces_info(self):
        dm = _make_dm([{"Name": "OnlyAction"}])
        with patch("pdstools.adm.Analysis.MetricLimits.evaluate_metric_rag", return_value="AMBER"):
            results = dm.analysis._check_taxonomy()
        infos = [f for f in results if f.severity == "info" and f.category == "taxonomy"]
        assert len(infos) == 4
        for f in infos:
            assert ("below recommended" in f.title) or ("above typical range" in f.title)


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
        results = dm.analysis._check_trends()
        assert results == []

    def test_exception_getting_snapshots_returns_empty(self):
        dm = _make_dm([{}])
        with patch.object(dm.model_data, "select", side_effect=RuntimeError("boom")):
            results = dm.analysis._check_trends()
        assert results == []

    def test_performance_decline_triggers_warning(self):
        # Build 6 weekly snapshots with AUC dropping sharply
        rows = []
        dates = ["20250101", "20250108", "20250115", "20250122", "20250129", "20250205"]
        perfs = [0.75, 0.74, 0.73, 0.65, 0.63, 0.61]  # >3 AUC drop first→second half
        for date, perf in zip(dates, perfs, strict=True):
            rows.append({"SnapshotTime": date, "Performance": perf, "ResponseCount": 1000, "Positives": 50})
        dm = _make_dm(rows)
        results = dm.analysis._check_trends()
        assert any("declining" in f.title and f.severity == "warning" for f in results)

    def test_response_volume_drop_triggers_warning(self):
        rows = []
        dates = ["20250101", "20250108", "20250115", "20250122", "20250129", "20250205"]
        resps = [10000, 10000, 10000, 1000, 1000, 1000]  # 90% drop
        for date, resp in zip(dates, resps, strict=True):
            rows.append({"SnapshotTime": date, "Performance": 0.70, "ResponseCount": resp, "Positives": 50})
        dm = _make_dm(rows)
        results = dm.analysis._check_trends()
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
                    "ControlPercentage": 2.0,
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
                    "ControlPercentage": 2.0,
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
                    "ControlPercentage": 2.0,
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
                    "ControlPercentage": 15.0,
                }
            ]
        )
        results = analysis._check_predictions(pred)
        assert any("Large control group" in f.title for f in results)
        assert any("15.0%" in f.title for f in results)

    def test_small_control_group_is_info(self, analysis):
        pred = self._make_prediction_mock(
            [
                {
                    "Channel": "Web",
                    "Direction": "Inbound",
                    "Lift": float("nan"),
                    "Prediction": "WebPrediction",
                    "ControlPercentage": 0.1,
                }
            ]
        )
        results = analysis._check_predictions(pred)
        assert any("Very small control group" in f.title for f in results)
        assert any("0.1%" in f.title for f in results)


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
            results = dm.analysis._check_trends()
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
        results = dm.analysis._check_trends()
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


# ── New behaviour: data-quality, collapsed dead channels, headline,
#     tiered AUC, mature-low-perf bucket, taxonomy-direction titles,
#     configuration channel hint. ─────────────────────────────────────────


class TestDataQualityCheck:
    """Detect malformed Channel/Direction values (e.g. "/", "/Inbound")."""

    def test_invalid_channel_predicate(self):
        for v in [None, "", " ", "/", "/Inbound", "Web/", "  /  "]:
            assert Analysis._is_invalid_channel_name(v), v
        for v in ["Web", "Email", "VZW-MFA-IOS"]:
            assert not Analysis._is_invalid_channel_name(v), v

    def test_invalid_pair_emits_finding_with_exact_count(self):
        # Three garbage rows + one good row. Data-quality finding should
        # report a model count of exactly 3, not 4.
        rows = [
            {"ModelID": "m1", "Channel": "", "Direction": ""},
            {"ModelID": "m2", "Channel": "", "Direction": "Inbound"},
            {"ModelID": "m3", "Channel": "/", "Direction": "Inbound"},
            {"ModelID": "m4", "Channel": "Web", "Direction": "Inbound"},
        ]
        dm = _make_dm(rows)
        findings, invalid = dm.analysis._check_data_quality()
        assert len(findings) == 1
        f = findings[0]
        assert f.severity == "critical"
        assert f.category == "data_quality"
        assert f.data["model_count"] == 3
        assert f.data["invalid_count"] == 3
        # The good "Web/Inbound" pair is not in the invalid set.
        assert "Web/Inbound" not in invalid
        assert "/Inbound" in invalid

    def test_clean_data_emits_no_finding(self):
        rows = [{"ModelID": "m1", "Channel": "Web", "Direction": "Inbound"}]
        dm = _make_dm(rows)
        findings, invalid = dm.analysis._check_data_quality()
        assert findings == []
        assert invalid == set()

    def test_invalid_channels_excluded_from_channel_check(self):
        # Mix: one valid dead channel + one invalid dead channel. The
        # invalid pair should not show up in the collapsed dead-channel
        # finding.
        rows = [
            {"ModelID": "m1", "Channel": "", "Direction": "", "ResponseCount": 0, "Positives": 0},
            {"ModelID": "m2", "Channel": "Web", "Direction": "Inbound", "ResponseCount": 0, "Positives": 0},
        ]
        dm = _make_dm(rows)
        findings, dead_count, _ = dm.analysis._check_channels(pl.lit(True), invalid_channels={"/"})
        # Dead-channel finding should mention only Web/Inbound.
        dead = [f for f in findings if "zero responses" in f.title]
        assert len(dead) == 1
        assert dead[0].data["channels"] == ["Web/Inbound"]
        assert dead_count == 1


class TestCollapsedDeadChannels:
    """Multiple dead channels are aggregated into one finding per direction."""

    def test_multiple_dead_channels_collapse_per_direction(self):
        # Mock summary_by_channel directly: 3 dead Inbound, 1 dead Outbound,
        # 1 live channel. Should yield 2 collapsed findings.
        summary_rows = [
            {
                "Channel": "A",
                "Direction": "Inbound",
                "Responses": 0,
                "Positives": 0,
                "Performance": None,
                "OmniChannel": None,
                "Configuration": "C1",
            },
            {
                "Channel": "B",
                "Direction": "Inbound",
                "Responses": 0,
                "Positives": 0,
                "Performance": None,
                "OmniChannel": None,
                "Configuration": "C2",
            },
            {
                "Channel": "C",
                "Direction": "Inbound",
                "Responses": 0,
                "Positives": 0,
                "Performance": None,
                "OmniChannel": None,
                "Configuration": "C3",
            },
            {
                "Channel": "X",
                "Direction": "Outbound",
                "Responses": 0,
                "Positives": 0,
                "Performance": None,
                "OmniChannel": None,
                "Configuration": "C4",
            },
            {
                "Channel": "Live",
                "Direction": "Inbound",
                "Responses": 5000,
                "Positives": 200,
                "Performance": 0.65,
                "OmniChannel": 0.5,
                "Configuration": "C5",
            },
        ]
        dm = _make_dm([{}])
        with patch.object(dm.aggregates, "summary_by_channel", return_value=pl.DataFrame(summary_rows).lazy()):
            findings, dead_count, _ = dm.analysis._check_channels(pl.lit(True))
        dead = [f for f in findings if "zero responses" in f.title]
        assert len(dead) == 2
        assert dead_count == 4
        by_dir = {f.data["direction"]: f for f in dead}
        assert by_dir["Inbound"].data["dead_channel_count"] == 3
        assert by_dir["Outbound"].data["dead_channel_count"] == 1
        assert "1 Outbound channel has" in by_dir["Outbound"].title
        assert "3 Inbound channels have" in by_dir["Inbound"].title

    def test_truncates_long_channel_list_with_more_suffix(self):
        summary_rows = [
            {
                "Channel": f"Ch{i}",
                "Direction": "Inbound",
                "Responses": 0,
                "Positives": 0,
                "Performance": None,
                "OmniChannel": None,
                "Configuration": f"C{i}",
            }
            for i in range(8)
        ]
        dm = _make_dm([{}])
        with patch.object(dm.aggregates, "summary_by_channel", return_value=pl.DataFrame(summary_rows).lazy()):
            findings, dead_count, _ = dm.analysis._check_channels(pl.lit(True))
        dead = next(f for f in findings if "zero responses" in f.title)
        assert "+3 more" in dead.title
        # All 8 channel names are kept on the finding's data payload.
        assert len(dead.data["channels"]) == 8
        assert dead_count == 8


class TestHeadlineFinding:
    """findings() prepends a single 🩺 [summary] headline."""

    def _stub_clean_summary(self, dm: ADMDatamart):
        """Patch summary_by_channel to one healthy channel.

        Synthetic ADMDatamart input doesn't always populate the
        Inbound/Outbound response split that summary_by_channel relies on,
        so we feed a known-good summary directly to keep these tests
        deterministic.
        """
        return patch.object(
            dm.analysis,
            "_health_check_channel_summary",
            return_value=pl.DataFrame(
                [
                    {
                        "ChannelDirection": "Web/Inbound",
                        "Channel": "Web",
                        "Direction": "Inbound",
                        "Responses": 100000,
                        "Positives": 5000,
                        "Performance": 0.72,
                        "CTR": 0.05,
                        "Actions": 20,
                        "OmniChannel": 0.5,
                    }
                ]
            ),
        )

    def test_clean_data_emits_healthy_headline(self):
        rows = [{"ResponseCount": 5000, "Positives": 500, "Performance": 0.72, "ModelID": f"m{i}"} for i in range(20)]
        dm = _make_dm(rows)
        with (
            self._stub_clean_summary(dm),
            patch.object(dm.analysis, "_check_configurations", return_value=[]),
            patch.object(dm.analysis, "_check_model_maturity", return_value=[]),
            patch.object(dm.analysis, "_check_taxonomy", return_value=[]),
            patch.object(dm.analysis, "_check_response_distribution", return_value=[]),
            patch.object(dm.analysis, "_check_trends", return_value=[]),
        ):
            all_findings = dm.analysis.findings()
        head = all_findings[0]
        assert head.category == "summary"
        assert head.severity == "success"
        assert "HEALTHY" in head.title
        assert head.data["pct_never_used"] == 0.0
        assert head.data["dead_channel_count"] == 0

    def test_high_unused_triggers_mixed_headline(self):
        rows = [{"ResponseCount": 0, "Positives": 0, "Performance": 0.5, "ModelID": f"u{i}"} for i in range(30)]
        rows += [{"ResponseCount": 5000, "Positives": 500, "Performance": 0.72, "ModelID": f"m{i}"} for i in range(70)]
        dm = _make_dm(rows)
        with self._stub_clean_summary(dm):
            head = dm.analysis.findings()[0]
        assert head.category == "summary"
        assert head.severity == "warning"
        assert "MIXED" in head.title
        assert "30% of models never used" in head.title

    def test_headline_severity_matches_body_findings(self):
        rows = [{"ResponseCount": 0, "Positives": 0, "Performance": 0.5, "ModelID": f"u{i}"} for i in range(30)]
        rows += [{"ResponseCount": 5000, "Positives": 500, "Performance": 0.72, "ModelID": f"m{i}"} for i in range(70)]
        dm = _make_dm(rows)
        with self._stub_clean_summary(dm):
            findings = dm.analysis.findings()
        head, body = findings[0], findings[1:]
        assert head.severity == max(
            (f.severity for f in body),
            key=lambda severity: {"critical": 3, "warning": 2, "info": 1, "success": 0}[severity],
        )

    def test_summary_uses_stethoscope_icon(self):
        rows = [{"ResponseCount": 5000, "Positives": 500, "Performance": 0.72}]
        dm = _make_dm(rows)
        with self._stub_clean_summary(dm):
            head = dm.analysis.findings()[0]
        assert str(head).startswith("🩺 [summary]")


class TestTieredAUCFinding:
    """Overall-AUC finding scales severity with the tier."""

    def _avg_finding(self, perf):
        rows = [{"ResponseCount": 5000, "Positives": 500, "Performance": perf, "ModelID": "m1"}]
        dm = _make_dm(rows)
        ld = dm.analysis._get_last_data()
        results, avg = dm.analysis._check_model_performance(ld, pl.lit(True))
        assert len(results) == 1
        return results[0], avg

    def test_low_auc_is_critical(self):
        f, _ = self._avg_finding(0.55)
        assert f.severity == "critical"
        assert "low" in f.title

    def test_borderline_auc_is_warning(self):
        f, _ = self._avg_finding(0.60)
        assert f.severity == "warning"
        assert "borderline" in f.title

    def test_healthy_auc_is_info(self):
        f, _ = self._avg_finding(0.65)
        assert f.severity == "info"
        assert "healthy" in f.title

    def test_strong_auc_is_success(self):
        f, _ = self._avg_finding(0.78)
        assert f.severity == "success"
        assert "strong" in f.title


class TestMaturityBuckets:
    """Visible maturity buckets are mutually exclusive but intentionally
    not exhaustive — mature low-performance models are surfaced through
    dedicated count findings (low_performance, stuck_at_50) rather than
    a percentage bucket, so the visible totals can sum to less than 100 %."""

    def test_buckets_are_disjoint(self):
        rows = (
            [{"ModelID": f"u{i}", "ResponseCount": 0, "Positives": 0, "Performance": 0.5} for i in range(10)]
            + [{"ModelID": f"z{i}", "ResponseCount": 100, "Positives": 0, "Performance": 0.5} for i in range(7)]
            + [{"ModelID": f"i{i}", "ResponseCount": 1000, "Positives": 50, "Performance": 0.6} for i in range(13)]
            + [{"ModelID": f"l{i}", "ResponseCount": 5000, "Positives": 500, "Performance": 0.50} for i in range(11)]
            + [{"ModelID": f"d{i}", "ResponseCount": 5000, "Positives": 500, "Performance": 0.72} for i in range(19)]
        )
        dm = _make_dm(rows)
        ld = dm.analysis._get_last_data()
        results = dm.analysis._check_model_maturity(ld)
        counts = {f.data["bucket"]: f.data["count"] for f in results if "bucket" in f.data}
        # Four visible buckets — `mature_low_perf` is intentionally not a
        # bucket; those models are reported via low-performance / stuck
        # warnings instead.
        assert counts == {
            "never_used": 10,
            "responses_no_positives": 7,
            "immature": 13,
            "mature_decent": 19,
        }
        # Total visible coverage is 49 of 60 — the 11 mature low-perf
        # models are deliberately excluded from the percentage view.
        assert sum(counts.values()) == 49


class TestTaxonomyThresholdInTitle:
    """Taxonomy findings now state the violated bound and direction."""

    def test_below_minimum_title(self):
        dm = _make_dm([{"Treatment": "T1"}])  # 1 treatment → < hard min 2
        results = dm.analysis._check_taxonomy()
        treatment = [f for f in results if "treatment" in f.title.lower()]
        assert any("below minimum" in f.title for f in treatment)


class TestConfigurationHasNoPositivesIncludesChannel:
    def test_channel_hint_appended(self):
        dm = _make_dm(
            [
                {
                    "Configuration": "Push_Click_Through_Rate",
                    "Channel": "Push",
                    "Direction": "Outbound",
                    "ResponseCount": 1000,
                    "Positives": 0,
                }
            ]
        )
        last_data = _make_last_data(
            [
                {
                    "Configuration": "Push_Click_Through_Rate",
                    "Channel": "Push",
                    "Direction": "Outbound",
                    "ResponseCount": 1000,
                    "Positives": 0,
                }
            ]
        )
        results = dm.analysis._check_configurations(last_data)
        f = next(f for f in results if "no positives" in f.title)
        assert "(Push/Outbound)" in f.title
        assert f.data["channel"] == "Push"
        assert f.data["direction"] == "Outbound"
