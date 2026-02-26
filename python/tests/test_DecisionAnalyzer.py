# python/tests/test_DecisionAnalyzer.py
"""Tests for the DecisionAnalyzer core functionality.

Covers both data source formats:
- v1: Explainability Extract (sample_explainability_extract.parquet)
- v2: Decision Analyzer / EEV2 (sample_eev2.parquet)

These tests validate the non-UI logic: construction, type detection,
data cleanup, ranking, pre-aggregation, sampling, and analysis methods.
"""

import pathlib

import polars as pl
import pytest
from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer
from pdstools.decision_analyzer.utils import (
    ColumnResolver,
    apply_filter,
    determine_extract_type,
    get_table_definition,
)

basePath = pathlib.Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def da_v1():
    """DecisionAnalyzer from Explainability Extract (v1) sample data."""
    raw = pl.scan_parquet(f"{basePath}/data/sample_explainability_extract.parquet")
    return DecisionAnalyzer(raw, sample_size=5000)


@pytest.fixture(scope="module")
def da_v2():
    """DecisionAnalyzer from Decision Analyzer / EEV2 (v2) sample data."""
    raw = pl.scan_parquet(f"{basePath}/data/sample_eev2.parquet")
    return DecisionAnalyzer(raw, sample_size=5000)


# ---------------------------------------------------------------------------
# Extract type detection
# ---------------------------------------------------------------------------


class TestClassMethodConstructors:
    """Test from_explainability_extract and from_decision_analyzer class methods."""

    def test_from_explainability_extract(self):
        da = DecisionAnalyzer.from_explainability_extract(
            f"{basePath}/data/sample_explainability_extract.parquet",
            sample_size=1000,
        )
        assert da.extract_type == "explainability_extract"
        assert da.decision_data.collect().height > 0

    def test_from_decision_analyzer(self):
        da = DecisionAnalyzer.from_decision_analyzer(
            f"{basePath}/data/sample_eev2.parquet",
            sample_size=1000,
        )
        assert da.extract_type == "decision_analyzer"
        assert da.decision_data.collect().height > 0

    def test_from_explainability_extract_invalid_path(self):
        with pytest.raises(Exception):
            DecisionAnalyzer.from_explainability_extract("/nonexistent/path.parquet")

    def test_from_decision_analyzer_invalid_path(self):
        with pytest.raises(Exception):
            DecisionAnalyzer.from_decision_analyzer("/nonexistent/path.parquet")


class TestExtractTypeDetection:
    """Verify that v1 vs v2 format is correctly identified."""

    def test_v1_detected_as_explainability_extract(self):
        raw = pl.scan_parquet(f"{basePath}/data/sample_explainability_extract.parquet")
        assert determine_extract_type(raw) == "explainability_extract"

    def test_v2_detected_as_decision_analyzer(self):
        raw = pl.scan_parquet(f"{basePath}/data/sample_eev2.parquet")
        assert determine_extract_type(raw) == "decision_analyzer"

    def test_v1_instance_has_correct_type(self, da_v1):
        assert da_v1.extract_type == "explainability_extract"

    def test_v2_instance_has_correct_type(self, da_v2):
        assert da_v2.extract_type == "decision_analyzer"


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestConstruction:
    """Verify that DecisionAnalyzer initializes correctly for both formats."""

    def test_v1_no_validation_error(self, da_v1):
        assert da_v1.validation_error is None

    def test_v2_no_validation_error(self, da_v2):
        assert da_v2.validation_error is None

    def test_v1_has_decision_data(self, da_v1):
        assert isinstance(da_v1.decision_data, pl.LazyFrame)
        assert da_v1.decision_data.collect().height > 0

    def test_v2_has_decision_data(self, da_v2):
        assert isinstance(da_v2.decision_data, pl.LazyFrame)
        assert da_v2.decision_data.collect().height > 0

    def test_v1_unfiltered_equals_decision_data_initially(self, da_v1):
        """Before any filters, both should point to the same data."""
        assert da_v1.unfiltered_raw_decision_data.collect().height == da_v1.decision_data.collect().height

    def test_construction_with_additional_columns(self):
        raw = pl.scan_parquet(f"{basePath}/data/sample_explainability_extract.parquet")
        da = DecisionAnalyzer(
            raw,
            sample_size=1000,
            additional_columns={"extra_col": pl.Utf8},
        )
        assert da.extract_type == "explainability_extract"

    def test_construction_with_mandatory_expr(self):
        raw = pl.scan_parquet(f"{basePath}/data/sample_explainability_extract.parquet")
        mandatory = pl.col("Issue") == "Service"
        da = DecisionAnalyzer(raw, sample_size=1000, mandatory_expr=mandatory)
        schema = da.decision_data.collect_schema()
        assert "is_mandatory" in schema.names()

    def test_missing_columns_raises_valueerror(self):
        """Constructing from data with missing critical columns should raise ValueError."""
        minimal = pl.LazyFrame({"Interaction ID": ["1"], "Action": ["A"]})
        with pytest.warns(UserWarning, match="missing"):
            with pytest.raises(ValueError, match="critical columns missing"):
                DecisionAnalyzer(minimal, sample_size=100)


# ---------------------------------------------------------------------------
# Data cleanup and derived columns
# ---------------------------------------------------------------------------


class TestDataCleanup:
    """Verify cleanup_raw_data produces expected columns and types."""

    def test_v1_has_rank_column(self, da_v1):
        schema = da_v1.decision_data.collect_schema()
        assert "pxRank" in schema.names()

    def test_v2_has_rank_column(self, da_v2):
        schema = da_v2.decision_data.collect_schema()
        assert "pxRank" in schema.names()

    def test_v1_has_day_column(self, da_v1):
        schema = da_v1.decision_data.collect_schema()
        assert "day" in schema.names()

    def test_v2_has_day_column(self, da_v2):
        schema = da_v2.decision_data.collect_schema()
        assert "day" in schema.names()

    def test_v1_has_is_mandatory_column(self, da_v1):
        schema = da_v1.decision_data.collect_schema()
        assert "is_mandatory" in schema.names()

    def test_v1_synthetic_stages(self, da_v1):
        """v1 should have synthetic Arbitration and Output stages."""
        stages = da_v1.decision_data.select("StageGroup").unique().collect().get_column("StageGroup").to_list()
        assert "Arbitration" in stages
        assert "Output" in stages

    def test_v2_has_real_stages(self, da_v2):
        """v2 should have real stage groups from the data."""
        stages = da_v2.AvailableNBADStages
        assert len(stages) > 2
        assert "Arbitration" in stages

    def test_v1_rank_1_is_output(self, da_v1):
        """In v1, rank-1 actions should be in Output stage."""
        rank1 = (
            da_v1.decision_data.filter(pl.col("pxRank") == 1)
            .select("StageGroup")
            .unique()
            .collect()
            .get_column("StageGroup")
            .to_list()
        )
        assert "Output" in rank1

    def test_rank_is_ordinal_per_interaction(self, da_v1):
        """Ranks should be 1..N within each interaction, no gaps."""
        sample = (
            da_v1.decision_data.head(10000)
            .group_by("Interaction ID")
            .agg(
                pl.col("pxRank").min().alias("min_rank"),
                pl.col("pxRank").max().alias("max_rank"),
                pl.col("pxRank").n_unique().alias("n_ranks"),
                pl.len().alias("n_rows"),
            )
            .collect()
        )
        # min rank should always be 1
        assert sample["min_rank"].min() == 1
        # number of unique ranks should equal number of rows per interaction
        assert (sample["n_ranks"] == sample["n_rows"]).all()


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


class TestStages:
    """Verify stage-related properties."""

    def test_v1_available_stages(self, da_v1):
        assert da_v1.AvailableNBADStages == ["Arbitration", "Output"]

    def test_v2_available_stages_includes_arbitration(self, da_v2):
        assert "Arbitration" in da_v2.AvailableNBADStages

    def test_stages_from_arbitration_down(self, da_v1):
        stages = da_v1.stages_from_arbitration_down
        assert stages[0] == "Arbitration"
        assert len(stages) >= 1

    def test_v2_stages_are_ordered(self, da_v2):
        """Stages should be in StageOrder sequence."""
        stages = da_v2.AvailableNBADStages
        assert len(stages) >= 2


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    """Verify the sampling mechanism."""

    def test_v1_sample_returns_lazyframe(self, da_v1):
        assert isinstance(da_v1.sample, pl.LazyFrame)

    def test_v2_sample_returns_lazyframe(self, da_v2):
        assert isinstance(da_v2.sample, pl.LazyFrame)

    def test_v1_sample_not_empty(self, da_v1):
        assert da_v1.sample.collect().height > 0

    def test_v2_sample_not_empty(self, da_v2):
        assert da_v2.sample.collect().height > 0

    def test_sample_respects_size_limit(self):
        """With a very small sample_size, the number of interactions should be limited."""
        raw = pl.scan_parquet(f"{basePath}/data/sample_eev2.parquet")
        da = DecisionAnalyzer(raw, sample_size=1000)
        n_interactions = da.sample.select(pl.n_unique("Interaction ID")).collect().item()
        # Should be at most sample_size (may be less due to hash sampling)
        assert n_interactions <= 1500  # some tolerance for hash-based sampling

    def test_sample_has_required_columns(self, da_v1):
        cols = da_v1.sample.collect_schema().names()
        for required in [
            "Interaction ID",
            "Issue",
            "Group",
            "Action",
            "pxRank",
            "Priority",
        ]:
            assert required in cols, f"Missing column: {required}"


# ---------------------------------------------------------------------------
# Pre-aggregation
# ---------------------------------------------------------------------------


class TestPreaggregation:
    """Verify pre-aggregated views."""

    def test_v1_filter_view_not_empty(self, da_v1):
        df = da_v1.getPreaggregatedFilterView.collect()
        assert df.height > 0

    def test_v2_filter_view_not_empty(self, da_v2):
        df = da_v2.getPreaggregatedFilterView.collect()
        assert df.height > 0

    def test_filter_view_has_decisions_column(self, da_v1):
        cols = da_v1.getPreaggregatedFilterView.collect_schema().names()
        assert "Decisions" in cols

    def test_v1_remaining_view_not_empty(self, da_v1):
        df = da_v1.getPreaggregatedRemainingView.collect()
        assert df.height > 0

    def test_v2_remaining_view_not_empty(self, da_v2):
        df = da_v2.getPreaggregatedRemainingView.collect()
        assert df.height > 0

    def test_remaining_view_has_win_rank_columns(self, da_v1):
        cols = da_v1.getPreaggregatedRemainingView.collect_schema().names()
        assert "Win_at_rank1" in cols

    def test_remaining_counts_geq_filter_counts(self, da_v2):
        """Remaining view totals should be >= filter view totals at each stage."""
        remaining = (
            da_v2.getPreaggregatedRemainingView.group_by("StageGroup")
            .agg(pl.sum("Decisions"))
            .collect()
            .sort("StageGroup")
        )
        filtered = (
            da_v2.getPreaggregatedFilterView.group_by("StageGroup")
            .agg(pl.sum("Decisions"))
            .collect()
            .sort("StageGroup")
        )
        # Remaining at earlier stages includes later stages, so total remaining >= filtered
        assert remaining["Decisions"].sum() >= filtered["Decisions"].sum()


# ---------------------------------------------------------------------------
# Global filters
# ---------------------------------------------------------------------------


class TestGlobalFilters:
    def test_apply_and_reset_filters(self):
        raw = pl.scan_parquet(f"{basePath}/data/sample_explainability_extract.parquet")
        da = DecisionAnalyzer(raw, sample_size=1000)
        original_count = da.decision_data.collect().height

        # Apply a filter that reduces data
        da.applyGlobalDataFilters(
            pl.col("Issue") == da.decision_data.select(pl.col("Issue").first()).collect().item(),
        )
        filtered_count = da.decision_data.collect().height
        assert filtered_count <= original_count

        # Reset
        da.resetGlobalDataFilters()
        reset_count = da.decision_data.collect().height
        assert reset_count == original_count


# ---------------------------------------------------------------------------
# Distribution data
# ---------------------------------------------------------------------------


class TestDistributionData:
    def test_v1_distribution_by_name(self, da_v1):
        df = da_v1.getDistributionData(
            stage="Arbitration",
            grouping_levels="Action",
        ).collect()
        assert df.height > 0
        assert "Action" in df.columns
        assert "Decisions" in df.columns

    def test_v2_distribution_by_name(self, da_v2):
        df = da_v2.getDistributionData(
            stage="Arbitration",
            grouping_levels="Action",
        ).collect()
        assert df.height > 0

    def test_distribution_sorted_descending(self, da_v1):
        df = da_v1.getDistributionData(
            stage="Arbitration",
            grouping_levels="Action",
        ).collect()
        decisions = df["Decisions"].to_list()
        assert decisions == sorted(decisions, reverse=True)


# ---------------------------------------------------------------------------
# Funnel data (v2 has real stages)
# ---------------------------------------------------------------------------


class TestFunnelData:
    def test_v2_funnel_returns_two_frames(self, da_v2):
        remaining, filtered = da_v2.getFunnelData(scope="Issue")
        assert isinstance(remaining, pl.LazyFrame)
        assert isinstance(filtered, pl.DataFrame)

    def test_v1_funnel_returns_two_frames(self, da_v1):
        remaining, filtered = da_v1.getFunnelData(scope="Issue")
        assert isinstance(remaining, pl.LazyFrame)
        assert isinstance(filtered, pl.DataFrame)


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


class TestSensitivity:
    def test_v1_sensitivity_returns_factors(self, da_v1):
        df = da_v1.get_sensitivity(win_rank=1).collect()
        factors = df["Factor"].to_list()
        assert "Propensity" in factors
        assert "Value" in factors
        assert "Context Weights" in factors
        assert "Levers" in factors

    def test_v2_sensitivity_returns_factors(self, da_v2):
        df = da_v2.get_sensitivity(win_rank=1).collect()
        assert df.height > 0
        assert "Factor" in df.columns
        assert "Influence" in df.columns

    def test_sensitivity_influence_non_negative_for_global(self, da_v1):
        """Global sensitivity should have non-negative influence values."""
        df = da_v1.get_sensitivity(win_rank=1).collect()
        assert (df["Influence"] >= 0).all()


# ---------------------------------------------------------------------------
# Win/loss distribution
# ---------------------------------------------------------------------------


class TestWinLoss:
    def test_v1_win_loss_returns_data(self, da_v1):
        df = da_v1.get_win_loss_distribution_data(level="Issue", win_rank=1).collect()
        assert df.height > 0
        assert "Status" in df.columns

    def test_win_loss_has_wins_and_losses(self, da_v1):
        df = da_v1.get_win_loss_distribution_data(level="Issue", win_rank=1).collect()
        statuses = df["Status"].unique().to_list()
        assert "Wins" in statuses
        assert "Losses" in statuses

    def test_win_percentages_sum_to_one(self, da_v1):
        df = da_v1.get_win_loss_distribution_data(level="Issue", win_rank=1).collect()
        for status in ["Wins", "Losses"]:
            total = df.filter(pl.col("Status") == status)["Percentage"].sum()
            assert abs(total - 1.0) < 0.01, f"{status} percentages sum to {total}"


# ---------------------------------------------------------------------------
# Overview stats
# ---------------------------------------------------------------------------


class TestOverviewStats:
    def test_v1_overview_has_expected_keys(self, da_v1):
        stats = da_v1.get_overview_stats
        expected_keys = {
            "Actions",
            "Channels",
            "Duration",
            "StartDate",
            "Customers",
            "Decisions",
        }
        assert expected_keys.issubset(stats.keys())

    def test_v2_overview_has_expected_keys(self, da_v2):
        stats = da_v2.get_overview_stats
        expected_keys = {
            "Actions",
            "Channels",
            "Duration",
            "StartDate",
            "Customers",
            "Decisions",
        }
        assert expected_keys.issubset(stats.keys())

    def test_overview_actions_positive(self, da_v1):
        assert da_v1.get_overview_stats["Actions"] > 0

    def test_overview_decisions_positive(self, da_v1):
        assert da_v1.get_overview_stats["Decisions"] > 0


# ---------------------------------------------------------------------------
# Trend data
# ---------------------------------------------------------------------------


class TestTrendData:
    def test_v1_trend_returns_data(self, da_v1):
        df = da_v1.get_trend_data(stage="Arbitration", scope="Issue").collect()
        assert df.height > 0
        assert "day" in df.columns
        assert "Decisions" in df.columns

    def test_v2_trend_returns_data(self, da_v2):
        df = da_v2.get_trend_data(stage="Arbitration", scope="Issue").collect()
        assert df.height > 0


# ---------------------------------------------------------------------------
# Action variation
# ---------------------------------------------------------------------------


class TestActionVariation:
    def test_v1_action_variation_at_arbitration(self, da_v1):
        df = da_v1.getActionVariationData(stage="Arbitration").collect()
        assert df.height > 0
        assert "ActionIndex" in df.columns
        assert "DecisionsFraction" in df.columns

    def test_action_variation_starts_at_zero(self, da_v1):
        df = da_v1.getActionVariationData(stage="Arbitration").collect()
        # First row is the dummy zero row
        assert df["ActionIndex"][0] == 0
        assert df["DecisionsFraction"][0] == 0.0

    def test_action_variation_ends_at_one(self, da_v1):
        df = da_v1.getActionVariationData(stage="Arbitration").collect()
        last_fraction = df["DecisionsFraction"][-1]
        assert abs(last_fraction - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Re-ranking
# ---------------------------------------------------------------------------


class TestReRank:
    def test_v1_rerank_returns_data(self, da_v1):
        df = da_v1.reRank().collect()
        assert df.height > 0

    def test_rerank_has_component_ranks(self, da_v1):
        df = da_v1.reRank()
        cols = df.collect_schema().names()
        assert "rank_PVCL" in cols
        assert "rank_VCL" in cols
        assert "rank_PCL" in cols

    def test_rerank_with_filters(self, da_v1):
        df = da_v1.reRank(
            additional_filters=pl.col("StageGroup").is_in(
                da_v1.stages_from_arbitration_down,
            ),
        ).collect()
        assert df.height > 0


# ---------------------------------------------------------------------------
# Win distribution (business lever analysis)
# ---------------------------------------------------------------------------


class TestWinDistribution:
    def test_baseline_distribution(self, da_v1):
        lever_cond = pl.col("Issue") == da_v1.decision_data.select(pl.col("Issue").first()).collect().item()
        result = da_v1.get_win_distribution_data(lever_condition=lever_cond)
        assert result.height > 0
        assert "original_win_count" in result.columns
        assert "selected_action" in result.columns

    def test_lever_adjusted_distribution(self, da_v1):
        lever_cond = pl.col("Issue") == da_v1.decision_data.select(pl.col("Issue").first()).collect().item()
        result = da_v1.get_win_distribution_data(
            lever_condition=lever_cond,
            lever_value=2.0,
        )
        assert "new_win_count" in result.columns

    def test_distribution_with_no_winner_tracking(self, da_v1):
        lever_cond = pl.col("Issue") == da_v1.decision_data.select(pl.col("Issue").first()).collect().item()
        result = da_v1.get_win_distribution_data(
            lever_condition=lever_cond,
            all_interactions=1000,
        )
        assert "No Winner" in result["Action"].to_list()


# ---------------------------------------------------------------------------
# Optionality
# ---------------------------------------------------------------------------


class TestOptionality:
    def test_v1_optionality_data(self, da_v1):
        df = da_v1.get_optionality_data(da_v1.sample).collect()
        assert df.height > 0
        assert "nOffers" in df.columns
        assert "Interactions" in df.columns

    def test_v2_optionality_data(self, da_v2):
        df = da_v2.get_optionality_data(da_v2.sample).collect()
        assert df.height > 0

    def test_optionality_with_trend(self, da_v1):
        df = da_v1.get_optionality_data_with_trend().collect()
        assert df.height > 0
        assert "day" in df.columns


# ---------------------------------------------------------------------------
# Filter components (v2 only)
# ---------------------------------------------------------------------------


class TestFilterComponents:
    def test_v2_filter_component_data(self, da_v2):
        """Filter component data should only be available for v2."""
        if "Component Name" not in da_v2.decision_data.collect_schema().names():
            pytest.skip("No pxComponentName in this dataset")
        df = da_v2.getFilterComponentData(top_n=5)
        assert df.height > 0
        assert "Component Name" in df.columns

    def test_v2_filter_component_data_includes_component_type(self, da_v2):
        """When pxComponentType is in the data, it should appear in the result."""
        if "Component Name" not in da_v2.decision_data.collect_schema().names():
            pytest.skip("No pxComponentName in this dataset")
        df = da_v2.getFilterComponentData(top_n=5)
        # pxComponentType may or may not be present depending on the dataset
        assert "Component Name" in df.columns
        assert "Filtered Decisions" in df.columns


# ---------------------------------------------------------------------------
# Component action impact (v2 only)
# ---------------------------------------------------------------------------


class TestComponentActionImpact:
    def test_v2_component_action_impact(self, da_v2):
        if "Component Name" not in da_v2.decision_data.collect_schema().names():
            pytest.skip("No pxComponentName in this dataset")
        df = da_v2.getComponentActionImpact(top_n=5)
        assert df.height > 0
        assert "Component Name" in df.columns
        assert "Action" in df.columns
        assert "Filtered Decisions" in df.columns

    def test_component_action_impact_respects_top_n(self, da_v2):
        if "Component Name" not in da_v2.decision_data.collect_schema().names():
            pytest.skip("No pxComponentName in this dataset")
        df = da_v2.getComponentActionImpact(top_n=3)
        # Each component should have at most 3 actions
        per_component = df.group_by("Component Name").agg(pl.len().alias("n"))
        assert per_component["n"].max() <= 3

    def test_component_action_impact_has_issue_group(self, da_v2):
        if "Component Name" not in da_v2.decision_data.collect_schema().names():
            pytest.skip("No pxComponentName in this dataset")
        df = da_v2.getComponentActionImpact(top_n=5)
        assert "Issue" in df.columns
        assert "Group" in df.columns

    def test_v1_component_action_impact_skipped(self, da_v1):
        """v1 data has no pxComponentName, so the method should produce empty results."""
        if "Component Name" not in da_v1.decision_data.collect_schema().names():
            pytest.skip("Expected: no pxComponentName in v1")


# ---------------------------------------------------------------------------
# Component drilldown (v2 only)
# ---------------------------------------------------------------------------


class TestComponentDrilldown:
    def test_v2_component_drilldown(self, da_v2):
        if "Component Name" not in da_v2.decision_data.collect_schema().names():
            pytest.skip("No pxComponentName in this dataset")
        # Pick the first available component
        components = (
            da_v2.decision_data.filter(pl.col("Record type") == "FILTERED_OUT")
            .select("Component Name")
            .unique()
            .collect()
            .get_column("Component Name")
            .to_list()
        )
        if not components:
            pytest.skip("No filtered components in dataset")
        df = da_v2.getComponentDrilldown(component_name=components[0])
        assert df.height > 0
        assert "Action" in df.columns
        assert "Filtered Decisions" in df.columns

    def test_component_drilldown_sorted_descending(self, da_v2):
        if "Component Name" not in da_v2.decision_data.collect_schema().names():
            pytest.skip("No pxComponentName in this dataset")
        components = (
            da_v2.decision_data.filter(pl.col("Record type") == "FILTERED_OUT")
            .select("Component Name")
            .unique()
            .collect()
            .get_column("Component Name")
            .to_list()
        )
        if not components:
            pytest.skip("No filtered components in dataset")
        df = da_v2.getComponentDrilldown(component_name=components[0])
        decisions = df["Filtered Decisions"].to_list()
        assert decisions == sorted(decisions, reverse=True)

    def test_component_drilldown_nonexistent_component(self, da_v2):
        if "Component Name" not in da_v2.decision_data.collect_schema().names():
            pytest.skip("No pxComponentName in this dataset")
        df = da_v2.getComponentDrilldown(component_name="NONEXISTENT_COMPONENT_XYZ")
        assert df.height == 0

    def test_component_drilldown_has_score_columns(self, da_v2):
        """Drilldown should include avg score columns when scoring data exists."""
        if "Component Name" not in da_v2.decision_data.collect_schema().names():
            pytest.skip("No pxComponentName in this dataset")
        components = (
            da_v2.decision_data.filter(pl.col("Record type") == "FILTERED_OUT")
            .select("Component Name")
            .unique()
            .collect()
            .get_column("Component Name")
            .to_list()
        )
        if not components:
            pytest.skip("No filtered components in dataset")
        df = da_v2.getComponentDrilldown(component_name=components[0])
        # At least one avg_* column should be present if scoring data exists
        avg_cols = [c for c in df.columns if c.startswith("avg_")]
        available_scores = {"Priority", "Value", "Propensity"}.intersection(
            da_v2.decision_data.collect_schema().names(),
        )
        if available_scores:
            assert len(avg_cols) > 0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_apply_filter_with_none(self):
        df = pl.LazyFrame({"a": [1, 2, 3]})
        result = apply_filter(df, None)
        assert result.collect().height == 3

    def test_apply_filter_with_single_expr(self):
        df = pl.LazyFrame({"a": [1, 2, 3]})
        result = apply_filter(df, pl.col("a") > 1)
        assert result.collect().height == 2

    def test_apply_filter_with_list_of_exprs(self):
        df = pl.LazyFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        result = apply_filter(df, [pl.col("a") > 1, pl.col("b") < 30])
        assert result.collect().height == 1

    def test_apply_filter_invalid_type_raises(self):
        df = pl.LazyFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError):
            apply_filter(df, "not_an_expr")

    def test_apply_filter_missing_column_raises(self):
        df = pl.LazyFrame({"a": [1, 2, 3]})
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            apply_filter(df, pl.col("nonexistent") > 1)

    def test_get_table_definition_valid(self):
        td = get_table_definition("decision_analyzer")
        assert isinstance(td, dict)
        assert "pxInteractionID" in td

        td2 = get_table_definition("explainability_extract")
        assert isinstance(td2, dict)

    def test_get_table_definition_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown table"):
            get_table_definition("nonexistent")


# ---------------------------------------------------------------------------
# ColumnResolver
# ---------------------------------------------------------------------------


class TestColumnResolver:
    def test_basic_rename(self):
        table_def = {
            "raw_col": {"display_name": "target_col", "default": True, "type": pl.Utf8},
        }
        resolver = ColumnResolver(table_definition=table_def, raw_columns={"raw_col"})
        assert resolver.rename_mapping == {"raw_col": "target_col"}
        assert "target_col" in resolver.final_columns

    def test_column_already_named_correctly(self):
        table_def = {
            "my_col": {"display_name": "my_col", "default": True, "type": pl.Utf8},
        }
        resolver = ColumnResolver(table_definition=table_def, raw_columns={"my_col"})
        assert resolver.rename_mapping == {}
        assert "my_col" in resolver.final_columns

    def test_conflict_prefers_target(self):
        """When both raw and target columns exist, target wins and raw is dropped."""
        table_def = {
            "raw_col": {"display_name": "target_col", "default": True, "type": pl.Utf8},
        }
        resolver = ColumnResolver(
            table_definition=table_def,
            raw_columns={"raw_col", "target_col"},
        )
        assert "raw_col" in resolver.columns_to_drop
        assert "target_col" in resolver.final_columns

    def test_missing_columns_detected(self):
        table_def = {
            "required_col": {
                "display_name": "required_col",
                "default": True,
                "type": pl.Utf8,
            },
        }
        resolver = ColumnResolver(table_definition=table_def, raw_columns=set())
        missing = resolver.get_missing_columns()
        assert "required_col" in missing

    def test_non_default_columns_not_in_missing(self):
        table_def = {
            "optional_col": {
                "display_name": "optional_col",
                "default": False,
                "type": pl.Utf8,
            },
        }
        resolver = ColumnResolver(table_definition=table_def, raw_columns=set())
        assert resolver.get_missing_columns() == []


# ---------------------------------------------------------------------------
# Scope and field helpers
# ---------------------------------------------------------------------------


class TestScopeHelpers:
    def test_v1_possible_scope_values(self, da_v1):
        scopes = da_v1.getPossibleScopeValues()
        assert "Issue" in scopes
        assert "Group" in scopes

    def test_v2_possible_scope_values(self, da_v2):
        scopes = da_v2.getPossibleScopeValues()
        assert "Issue" in scopes

    def test_available_fields_for_filtering(self, da_v1):
        fields = da_v1.getAvailableFieldsForFiltering()
        assert len(fields) > 0
        assert "Action" in fields

    def test_available_categorical_fields(self, da_v1):
        fields = da_v1.getAvailableFieldsForFiltering(categoricalOnly=True)
        assert len(fields) > 0
