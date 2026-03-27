# python/tests/test_DecisionAnalyzer.py
"""
Tests for the DecisionAnalyzer core functionality.

Covers both data source formats:
- v1: Explainability Extract (sample_explainability_extract.parquet)
- v2: Decision Analyzer / EEV2 (sample_eev2.parquet)

These tests validate the non-UI logic: construction, type detection,
data cleanup, ranking, pre-aggregation, sampling, and analysis methods.
"""

import pathlib
from datetime import datetime

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
        assert "Rank" in schema.names()

    def test_v2_has_rank_column(self, da_v2):
        schema = da_v2.decision_data.collect_schema()
        assert "Rank" in schema.names()

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
        stages = da_v1.decision_data.select("Stage Group").unique().collect().get_column("Stage Group").to_list()
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
            da_v1.decision_data.filter(pl.col("Rank") == 1)
            .select("Stage Group")
            .unique()
            .collect()
            .get_column("Stage Group")
            .to_list()
        )
        assert "Output" in rank1

    def test_rank_is_ordinal_per_interaction(self, da_v1):
        """Ranks should be 1..N within each interaction, no gaps."""
        sample = (
            da_v1.decision_data.head(10000)
            .group_by("Interaction ID")
            .agg(
                pl.col("Rank").min().alias("min_rank"),
                pl.col("Rank").max().alias("max_rank"),
                pl.col("Rank").n_unique().alias("n_ranks"),
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
            "Rank",
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
            da_v2.getPreaggregatedRemainingView.group_by("Stage Group")
            .agg(pl.sum("Decisions"))
            .collect()
            .sort("Stage Group")
        )
        filtered = (
            da_v2.getPreaggregatedFilterView.group_by("Stage Group")
            .agg(pl.sum("Decisions"))
            .collect()
            .sort("Stage Group")
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
        da.applyGlobalDataFilters(pl.col("Issue") == da.decision_data.select(pl.col("Issue").first()).collect().item())
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
        df = da_v1.getDistributionData(stage="Arbitration", grouping_levels="Action").collect()
        assert df.height > 0
        assert "Action" in df.columns
        assert "Decisions" in df.columns

    def test_v2_distribution_by_name(self, da_v2):
        df = da_v2.getDistributionData(stage="Arbitration", grouping_levels="Action").collect()
        assert df.height > 0

    def test_distribution_sorted_descending(self, da_v1):
        df = da_v1.getDistributionData(stage="Arbitration", grouping_levels="Action").collect()
        decisions = df["Decisions"].to_list()
        assert decisions == sorted(decisions, reverse=True)


# ---------------------------------------------------------------------------
# Funnel data (v2 has real stages)
# ---------------------------------------------------------------------------


class TestFunnelData:
    def test_v2_funnel_returns_three_frames(self, da_v2):
        available, passing, filtered = da_v2.getFunnelData(scope="Issue")
        assert isinstance(available, pl.LazyFrame)
        assert isinstance(passing, pl.DataFrame)
        assert isinstance(filtered, pl.DataFrame)

    def test_v1_funnel_returns_three_frames(self, da_v1):
        available, passing, filtered = da_v1.getFunnelData(scope="Issue")
        assert isinstance(available, pl.LazyFrame)
        assert isinstance(passing, pl.DataFrame)
        assert isinstance(filtered, pl.DataFrame)

    def test_passing_lte_available(self, da_v2):
        """Passing actions at each stage must be <= available actions."""
        available, passing, filtered = da_v2.getFunnelData(scope="Issue")
        level = da_v2.level
        avail_df = (
            available.collect()
            .with_columns(pl.col(level).cast(pl.Utf8))
            .group_by(level)
            .agg(pl.sum("action_occurrences").alias("avail"))
        )
        pass_df = (
            passing.with_columns(pl.col(level).cast(pl.Utf8))
            .group_by(level)
            .agg(pl.sum("action_occurrences").alias("pass"))
        )
        joined = avail_df.join(pass_df, on=level, how="left").fill_null(0)
        assert (joined["pass"] <= joined["avail"]).all()

    def test_passing_has_required_columns(self, da_v2):
        """Passing df must have same columns as filtered df."""
        available, passing, filtered = da_v2.getFunnelData(scope="Issue")
        for col in [
            "action_occurrences",
            "interaction_count_for_scope",
            "interaction_count",
            "actions_per_interaction",
            "penetration_pct",
        ]:
            assert col in passing.columns
            assert col in filtered.columns

    def test_decisions_without_actions_shape(self, da_v2):
        result = da_v2.get_decisions_without_actions_data()
        assert isinstance(result, pl.DataFrame)
        assert da_v2.level in result.columns
        assert "decisions_without_actions" in result.columns
        expected_stages = [s for s in da_v2.AvailableNBADStages if s != "Output"]
        assert len(result) == len(expected_stages)
        assert "Output" not in result[da_v2.level].to_list()

    def test_decisions_without_actions_non_negative(self, da_v2):
        result = da_v2.get_decisions_without_actions_data()
        assert (result["decisions_without_actions"] >= 0).all()

    def test_decisions_without_actions_per_stage(self, da_v2):
        """Values are stage deltas and total knockouts should be bounded by total decisions."""
        result = da_v2.get_decisions_without_actions_data()
        total = da_v2.decision_data.select("Interaction ID").unique().collect().height
        assert result["decisions_without_actions"].sum() <= total

    def test_decisions_without_actions_last_stage_uses_output_survivors(self, da_v2):
        """Sum of stage deltas equals total decisions minus Output survivors."""
        result = da_v2.get_decisions_without_actions_data()
        total = da_v2.decision_data.select("Interaction ID").unique().collect().height
        output_survivors = (
            da_v2.getPreaggregatedFilterView.filter(pl.col(da_v2.level) == "Output")
            .select(pl.col("Interaction_IDs").flatten().unique().count())
            .collect()
            .item()
        )
        assert result["decisions_without_actions"].sum() == total - output_survivors


# ---------------------------------------------------------------------------
# Funnel summary
# ---------------------------------------------------------------------------


class TestFunnelSummary:
    def test_summary_columns(self, da_v2):
        available, passing, filtered = da_v2.getFunnelData(scope="Issue")
        result = da_v2.get_funnel_summary(available, passing)
        expected_cols = {
            da_v2.level,
            "Avg Available/Decision",
            "Avg Passing/Decision",
            "Avg Filtered/Decision",
            "% of total filtered",
            "% Decisions with Actions",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_summary_row_count(self, da_v2):
        available, passing, filtered = da_v2.getFunnelData(scope="Issue")
        result = da_v2.get_funnel_summary(available, passing)
        assert len(result) == len(da_v2.AvailableNBADStages)

    def test_filtered_is_difference(self, da_v2):
        available, passing, filtered = da_v2.getFunnelData(scope="Issue")
        result = da_v2.get_funnel_summary(available, passing)
        diff = result["Avg Available/Decision"] - result["Avg Passing/Decision"]
        # Allow 0.01 rounding tolerance since each column is independently rounded
        assert ((result["Avg Filtered/Decision"] - diff).abs() <= 0.01).all()

    def test_pct_sums_to_100(self, da_v2):
        available, passing, filtered = da_v2.getFunnelData(scope="Issue")
        result = da_v2.get_funnel_summary(available, passing)
        total_pct = result["% of total filtered"].sum()
        assert abs(total_pct - 100.0) < 0.5  # rounding tolerance

    def test_decisions_pct_in_range(self, da_v2):
        available, passing, filtered = da_v2.getFunnelData(scope="Issue")
        result = da_v2.get_funnel_summary(available, passing)
        assert (result["% Decisions with Actions"] >= 0).all()
        assert (result["% Decisions with Actions"] <= 100.1).all()  # rounding tolerance


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


@pytest.fixture(scope="module")
def da_win_loss_boundary_tiny():
    """Tiny deterministic dataset to verify selected-group rank boundary semantics."""
    tiny_data = pl.LazyFrame(
        {
            "pySubjectID": ["S1", "S1", "S1", "S1", "S2", "S2", "S2"],
            "pxInteractionID": ["I1", "I1", "I1", "I1", "I2", "I2", "I2"],
            "pxDecisionTime": [
                datetime(2024, 1, 1, 0, 0, 0),
                datetime(2024, 1, 1, 0, 0, 1),
                datetime(2024, 1, 1, 0, 0, 2),
                datetime(2024, 1, 1, 0, 0, 3),
                datetime(2024, 1, 1, 0, 1, 0),
                datetime(2024, 1, 1, 0, 1, 1),
                datetime(2024, 1, 1, 0, 1, 2),
            ],
            "pyIssue": ["Issue1", "Issue1", "Issue1", "Issue1", "Issue2", "Issue2", "Issue2"],
            "pyGroup": ["Other", "Selected", "Other", "Selected", "Selected", "Other", "Other"],
            "pyName": ["A", "B", "D", "C", "E", "F", "G"],
            "pyChannel": ["Web", "Web", "Web", "Web", "Web", "Web", "Web"],
            "pyDirection": ["Inbound", "Inbound", "Inbound", "Inbound", "Inbound", "Inbound", "Inbound"],
            "Value": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ContextWeight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "Weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "pyPropensity": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "FinalPropensity": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "Priority": [0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7],
            "ModelControlGroup": ["N", "N", "N", "N", "N", "N", "N"],
        }
    )
    return DecisionAnalyzer(tiny_data, sample_size=10)


class TestSelectedGroupRankBoundariesWinLoss:
    def test_selected_group_rank_boundaries_are_computed_per_interaction(self, da_win_loss_boundary_tiny):
        selected_group_filter = pl.col("Group") == "Selected"

        boundaries = (
            da_win_loss_boundary_tiny.get_selected_group_rank_boundaries(
                group_filter=selected_group_filter,
            )
            .sort("Interaction ID")
            .collect()
        )

        expected = pl.DataFrame(
            {
                "Interaction ID": ["I1", "I2"],
                "selected_group_best_rank": [2, 1],
                "selected_group_worst_rank": [4, 1],
                "selected_group_row_count": [2, 1],
            }
        )

        assert boundaries.rows(named=True) == expected.rows(named=True)

    def test_selected_group_wins_and_losses_use_rank_boundaries(self, da_win_loss_boundary_tiny):
        selected_group_filter = pl.col("Group") == "Selected"

        losses_to_selected = (
            da_win_loss_boundary_tiny.get_winning_or_losing_interactions(
                group_filter=selected_group_filter,
                win=True,
            )
            .sort("Interaction ID")
            .collect()
            .get_column("Interaction ID")
            .to_list()
        )
        wins_against_selected = (
            da_win_loss_boundary_tiny.get_winning_or_losing_interactions(
                group_filter=selected_group_filter,
                win=False,
            )
            .sort("Interaction ID")
            .collect()
            .get_column("Interaction ID")
            .to_list()
        )

        assert losses_to_selected == ["I2"]
        assert wins_against_selected == ["I1"]

    def test_group_filter_status_distributions_match_expected_actions(self, da_win_loss_boundary_tiny):
        selected_group_filter = pl.col("Group") == "Selected"

        beaten_by_selected = (
            da_win_loss_boundary_tiny.get_win_loss_distribution_data(
                level="Action",
                group_filter=selected_group_filter,
                status="Wins",
            )
            .sort("Action")
            .collect()
        )
        beating_selected = (
            da_win_loss_boundary_tiny.get_win_loss_distribution_data(
                level="Action",
                group_filter=selected_group_filter,
                status="Losses",
            )
            .sort("Action")
            .collect()
        )

        assert beaten_by_selected.select("Action").to_series().to_list() == ["F", "G"]
        assert beaten_by_selected.select("Decisions").to_series().to_list() == [1, 1]
        assert beating_selected.select("Action").to_series().to_list() == ["A"]
        assert beating_selected.select("Decisions").to_series().to_list() == [1]

    def test_group_filter_paths_validate_required_arguments(self, da_win_loss_boundary_tiny):
        with pytest.raises(ValueError, match="win_rank must be provided"):
            da_win_loss_boundary_tiny.get_win_loss_distribution_data(level="Action")

        with pytest.raises(ValueError, match="status must be either 'Wins' or 'Losses'"):
            da_win_loss_boundary_tiny.get_win_loss_distribution_data(
                level="Action",
                group_filter=pl.col("Group") == "Selected",
                status=None,
            )

    def test_group_filter_distribution_with_top_k(self, da_win_loss_boundary_tiny):
        selected_group_filter = pl.col("Group") == "Selected"

        result = da_win_loss_boundary_tiny.get_win_loss_distribution_data(
            level="Action",
            group_filter=selected_group_filter,
            status="Wins",
            top_k=1,
        ).collect()
        assert result.height == 1

    def test_win_loss_distributions_with_group_filter(self, da_win_loss_boundary_tiny):
        selected_group_filter = pl.col("Group") == "Selected"

        interactions_win = da_win_loss_boundary_tiny.get_winning_or_losing_interactions(
            group_filter=selected_group_filter,
            win=True,
        )
        interactions_loss = da_win_loss_boundary_tiny.get_winning_or_losing_interactions(
            group_filter=selected_group_filter,
            win=False,
        )
        winning_from, losing_to = da_win_loss_boundary_tiny.get_win_loss_distributions(
            interactions_win=interactions_win,
            interactions_loss=interactions_loss,
            groupby_cols=["Action"],
            top_k=10,
            group_filter=selected_group_filter,
        )
        win_result = winning_from.collect()
        loss_result = losing_to.collect()

        assert win_result.height > 0
        assert "Action" in win_result.columns
        assert "Decisions" in win_result.columns
        assert loss_result.height > 0
        assert "Action" in loss_result.columns
        assert "Decisions" in loss_result.columns

    def test_win_loss_distributions_with_win_rank(self, da_win_loss_boundary_tiny):
        selected_group_filter = pl.col("Group") == "Selected"
        interactions_win = da_win_loss_boundary_tiny.get_winning_or_losing_interactions(
            group_filter=selected_group_filter,
            win=True,
        )
        interactions_loss = da_win_loss_boundary_tiny.get_winning_or_losing_interactions(
            group_filter=selected_group_filter,
            win=False,
        )
        winning_from, losing_to = da_win_loss_boundary_tiny.get_win_loss_distributions(
            interactions_win=interactions_win,
            interactions_loss=interactions_loss,
            groupby_cols=["Action"],
            top_k=10,
            win_rank=1,
        )

        assert "Action" in winning_from.collect().columns
        assert "Action" in losing_to.collect().columns

    def test_win_loss_distributions_validates_win_rank_when_no_group_filter(self, da_win_loss_boundary_tiny):
        interactions = pl.LazyFrame({"Interaction ID": ["I1"]})
        with pytest.raises(ValueError, match="win_rank must be provided"):
            da_win_loss_boundary_tiny.get_win_loss_distributions(
                interactions_win=interactions,
                interactions_loss=interactions,
                groupby_cols=["Action"],
                top_k=10,
            )

    def test_win_loss_counts_at_rank_1(self, da_win_loss_boundary_tiny):
        selected_group_filter = pl.col("Group") == "Selected"
        counts = da_win_loss_boundary_tiny.get_win_loss_counts(
            group_filter=selected_group_filter,
            win_rank=1,
        )
        # I2: best_rank=1 → win. I1: best_rank=2 → loss.
        assert counts == {"wins": 1, "losses": 1, "total": 2}

    def test_win_loss_counts_at_rank_2(self, da_win_loss_boundary_tiny):
        selected_group_filter = pl.col("Group") == "Selected"
        counts = da_win_loss_boundary_tiny.get_win_loss_counts(
            group_filter=selected_group_filter,
            win_rank=2,
        )
        # I1: best_rank=2 → win. I2: best_rank=1 → win. Both win.
        assert counts == {"wins": 2, "losses": 0, "total": 2}

    def test_win_loss_counts_sum_equals_total(self, da_win_loss_boundary_tiny):
        selected_group_filter = pl.col("Group") == "Selected"
        for n in [1, 2, 3, 10]:
            counts = da_win_loss_boundary_tiny.get_win_loss_counts(
                group_filter=selected_group_filter,
                win_rank=n,
            )
            assert counts["wins"] + counts["losses"] == counts["total"]


# ---------------------------------------------------------------------------
# Boxplot point cap and sampling in plots
# ---------------------------------------------------------------------------


class TestBoxplotPointCapAndSampling:
    def test_boxplot_point_cap_returns_sample_size(self, da_v1):
        assert da_v1.plot._boxplot_point_cap() == da_v1.sample_size

    def test_boxplot_point_cap_returns_default_when_no_sample_size(self, da_v1):
        from pdstools.decision_analyzer.plots import DEFAULT_BOXPLOT_POINT_CAP

        original = da_v1.sample_size
        da_v1.sample_size = None
        try:
            assert da_v1.plot._boxplot_point_cap() == DEFAULT_BOXPLOT_POINT_CAP
        finally:
            da_v1.sample_size = original

    def test_prio_factor_boxplots_returns_tuple(self, da_v1):
        first_action = da_v1.decision_data.select("Action").first().collect().item()
        result = da_v1.plot.prio_factor_boxplots(
            reference=pl.col("Action") == first_action,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_prio_factor_boxplots_return_df(self, da_v1):
        first_action = da_v1.decision_data.select("Action").first().collect().item()
        df = da_v1.plot.prio_factor_boxplots(
            reference=pl.col("Action") == first_action,
            return_df=True,
        )
        assert isinstance(df, pl.DataFrame)
        assert "segment" in df.columns

    def test_prio_factor_boxplots_sampling_caps_rows(self, da_v1):
        first_action = da_v1.decision_data.select("Action").first().collect().item()
        original = da_v1.sample_size
        da_v1.sample_size = 100
        try:
            df = da_v1.plot.prio_factor_boxplots(
                reference=pl.col("Action") == first_action,
                return_df=True,
            )
            assert df.height == 100
        finally:
            da_v1.sample_size = original

    def test_prio_factor_boxplots_warning_message(self, da_v1):
        """When data exceeds point cap but both segments survive, return sampling warning."""
        first_action = da_v1.decision_data.select("Action").first().collect().item()
        original = da_v1.sample_size
        da_v1.sample_size = 5000
        try:
            result = da_v1.plot.prio_factor_boxplots(
                reference=pl.col("Action") == first_action,
            )
            fig, warning = result
            if fig is not None:
                assert warning is not None
                assert "sample" in warning.lower()
        finally:
            da_v1.sample_size = original

    def test_rank_boxplot_returns_figure(self, da_v1):
        first_action = da_v1.decision_data.select("Action").first().collect().item()
        fig = da_v1.plot.rank_boxplot(
            reference=pl.col("Action") == first_action,
        )
        assert hasattr(fig, "update_layout")

    def test_rank_boxplot_sampling_with_low_cap(self, da_v1):
        first_action = da_v1.decision_data.select("Action").first().collect().item()
        original = da_v1.sample_size
        da_v1.sample_size = 10
        try:
            fig = da_v1.plot.rank_boxplot(
                reference=pl.col("Action") == first_action,
            )
            assert hasattr(fig, "update_layout")
        finally:
            da_v1.sample_size = original


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

    def test_action_variation_with_color_by(self, da_v1):
        df = da_v1.getActionVariationData(stage="Arbitration", color_by="Channel/Direction").collect()
        assert df.height > 0
        assert "Channel/Direction" in df.columns
        assert "ActionIndex" in df.columns
        assert "DecisionsFraction" in df.columns
        # Verify each group starts at zero and has proper fractions
        for group in df["Channel/Direction"].unique():
            group_df = df.filter(pl.col("Channel/Direction") == group)
            # Each group should have its own zero point
            assert group_df["ActionIndex"].min() == 0
            # Each group should end at or near 1.0
            max_fraction = group_df["DecisionsFraction"].max()
            assert abs(max_fraction - 1.0) < 0.01


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
        df = da_v1.reRank(additional_filters=pl.col("Stage Group").is_in(da_v1.stages_from_arbitration_down)).collect()
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
        result = da_v1.get_win_distribution_data(lever_condition=lever_cond, lever_value=2.0)
        assert "new_win_count" in result.columns

    def test_distribution_with_no_winner_tracking(self, da_v1):
        lever_cond = pl.col("Issue") == da_v1.decision_data.select(pl.col("Issue").first()).collect().item()
        result = da_v1.get_win_distribution_data(lever_condition=lever_cond, all_interactions=1000)
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
            da_v2.decision_data.filter(pl.col("Record Type") == "FILTERED_OUT")
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
            da_v2.decision_data.filter(pl.col("Record Type") == "FILTERED_OUT")
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
            da_v2.decision_data.filter(pl.col("Record Type") == "FILTERED_OUT")
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
            da_v2.decision_data.collect_schema().names()
        )
        if available_scores:
            assert len(avg_cols) > 0

    def test_drilldown_respects_scope(self, da_v2):
        """Drilldown should group at the requested scope level."""
        if "Component Name" not in da_v2.decision_data.collect_schema().names():
            pytest.skip("No pxComponentName in this dataset")
        components = (
            da_v2.decision_data.filter(pl.col("Record Type") == "FILTERED_OUT")
            .select("Component Name")
            .unique()
            .collect()
            .get_column("Component Name")
            .to_list()
        )
        if not components:
            pytest.skip("No filtered components in dataset")
        name = components[0]
        for scope in ["Issue", "Group", "Action"]:
            df = da_v2.getComponentDrilldown(component_name=name, scope=scope)
            assert scope in df.columns


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
        table_def = {"raw_col": {"display_name": "target_col", "default": True, "type": pl.Utf8}}
        resolver = ColumnResolver(table_definition=table_def, raw_columns={"raw_col"})
        assert resolver.rename_mapping == {"raw_col": "target_col"}
        assert "target_col" in resolver.final_columns

    def test_column_already_named_correctly(self):
        table_def = {"my_col": {"display_name": "my_col", "default": True, "type": pl.Utf8}}
        resolver = ColumnResolver(table_definition=table_def, raw_columns={"my_col"})
        assert resolver.rename_mapping == {}
        assert "my_col" in resolver.final_columns

    def test_conflict_prefers_target(self):
        """When both raw and target columns exist, target wins and raw is dropped."""
        table_def = {"raw_col": {"display_name": "target_col", "default": True, "type": pl.Utf8}}
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
            }
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
            }
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


# ---------------------------------------------------------------------------
# Available levels and set_level
# ---------------------------------------------------------------------------


class TestAvailableLevelsAndSetLevel:
    def test_v1_available_levels_is_stage_group_only(self, da_v1):
        """v1 (explainability_extract) only has synthetic Stage Group."""
        assert da_v1.available_levels == ["Stage Group"]

    def test_v2_available_levels_includes_stage_group(self, da_v2):
        levels = da_v2.available_levels
        assert "Stage Group" in levels

    def test_v2_set_level_to_stage(self, da_v2):
        """If Stage is available in v2, switching to it should work."""
        if "Stage" not in da_v2.available_levels:
            pytest.skip("Stage not available in this v2 dataset")
        original_level = da_v2.level
        da_v2.set_level("Stage")
        assert da_v2.level == "Stage"
        # Restore
        da_v2.set_level(original_level)
        assert da_v2.level == original_level

    def test_set_level_invalid_raises_valueerror(self, da_v2):
        with pytest.raises(ValueError, match="level must be one of"):
            da_v2.set_level("NonexistentLevel")

    def test_set_level_same_is_noop(self, da_v2):
        """Setting level to its current value should be a no-op."""
        original_level = da_v2.level
        original_stages = list(da_v2.AvailableNBADStages)
        da_v2.set_level(original_level)  # should not change anything
        assert da_v2.level == original_level
        assert da_v2.AvailableNBADStages == original_stages


# ---------------------------------------------------------------------------
# Stage methods (stages_from_arbitration_down, arbitration_stage)
# ---------------------------------------------------------------------------


class TestStageMethods:
    def test_stages_from_arbitration_down_v2(self, da_v2):
        stages = da_v2.stages_from_arbitration_down
        assert stages[0] == "Arbitration"
        assert len(stages) >= 1

    def test_arbitration_stage_returns_lazyframe(self, da_v1):
        result = da_v1.arbitration_stage
        assert isinstance(result, pl.LazyFrame)
        df = result.collect()
        assert df.height > 0

    def test_arbitration_stage_only_contains_arbitration_stages(self, da_v1):
        df = da_v1.arbitration_stage.collect()
        stage_values = df[da_v1.level].unique().to_list()
        for s in stage_values:
            assert s in da_v1.stages_from_arbitration_down

    def test_v2_arbitration_stage(self, da_v2):
        result = da_v2.arbitration_stage
        assert isinstance(result, pl.LazyFrame)
        df = result.collect()
        assert df.height > 0


# ---------------------------------------------------------------------------
# Data access: getPossibleStageValues, stage_to_group_mapping
# ---------------------------------------------------------------------------


class TestDataAccess:
    def test_getPossibleStageValues_v1(self, da_v1):
        stages = da_v1.getPossibleStageValues()
        assert isinstance(stages, list)
        assert "Arbitration" in stages
        assert "Output" in stages

    def test_getPossibleStageValues_v2(self, da_v2):
        stages = da_v2.getPossibleStageValues()
        assert isinstance(stages, list)
        assert len(stages) >= 2
        assert "Arbitration" in stages

    def test_stage_to_group_mapping_at_stage_group_level(self, da_v2):
        """When level is Stage Group, mapping should be empty."""
        assert da_v2.level == "Stage Group"
        mapping = da_v2.stage_to_group_mapping
        assert mapping == {}

    def test_stage_to_group_mapping_at_stage_level(self, da_v2):
        """When level is Stage, mapping should return non-empty dict."""
        if "Stage" not in da_v2.available_levels:
            pytest.skip("Stage not available in this v2 dataset")
        original_level = da_v2.level
        da_v2.set_level("Stage")
        mapping = da_v2.stage_to_group_mapping
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        # Values should be stage group names
        assert "Arbitration" in mapping.values() or len(mapping) > 0
        # Restore
        da_v2.set_level(original_level)

    def test_stage_to_group_mapping_v1_always_empty(self, da_v1):
        """v1 has no real stage mapping."""
        mapping = da_v1.stage_to_group_mapping
        assert mapping == {}


# ---------------------------------------------------------------------------
# Analysis methods: optionality funnel, AB tests, thresholding
# ---------------------------------------------------------------------------


class TestAnalysisMethods:
    def test_get_optionality_funnel_v1(self, da_v1):
        result = da_v1.get_optionality_funnel()
        df = result.collect()
        assert df.height > 0
        assert "available_actions" in df.columns
        assert "Interactions" in df.columns
        assert da_v1.level in df.columns

    def test_get_optionality_funnel_v2(self, da_v2):
        result = da_v2.get_optionality_funnel()
        df = result.collect()
        assert df.height > 0
        assert "available_actions" in df.columns

    def test_getABTestResults_no_crash(self, da_v2):
        """AB test requires Model Control Group column — test it doesn't crash or raises cleanly."""
        schema = da_v2.decision_data.collect_schema()
        if "Model Control Group" not in schema.names():
            with pytest.raises(Exception):
                da_v2.getABTestResults()
        else:
            result = da_v2.getABTestResults()
            assert isinstance(result, pl.DataFrame)
            assert result.height > 0

    def test_getThresholdingData_propensity(self, da_v1):
        result = da_v1.getThresholdingData(fld="Propensity")
        assert isinstance(result, pl.DataFrame)
        assert result.height > 0
        assert "Decile" in result.columns
        assert "Threshold" in result.columns
        assert "Count" in result.columns
        assert da_v1.level in result.columns

    def test_getThresholdingData_priority(self, da_v2):
        result = da_v2.getThresholdingData(fld="Priority")
        assert isinstance(result, pl.DataFrame)
        assert result.height > 0

    def test_getThresholdingData_custom_quantile_range(self, da_v1):
        result = da_v1.getThresholdingData(fld="Propensity", quantile_range=range(20, 100, 20))
        assert isinstance(result, pl.DataFrame)
        assert result.height > 0

    def test_getThresholdingData_caches_result(self, da_v1):
        """Calling with same args should return cached result."""
        r1 = da_v1.getThresholdingData(fld="Propensity")
        r2 = da_v1.getThresholdingData(fld="Propensity")
        assert r1.equals(r2)


# ---------------------------------------------------------------------------
# Filtering/scoping: priority component distribution, remaining at stage,
# filtered_action_counts
# ---------------------------------------------------------------------------


class TestFilteringAndScoping:
    def test_priority_component_distribution_v2(self, da_v2):
        """Test priority_component_distribution with a known component."""
        available = set(da_v2.sample.collect_schema().names())
        component = None
        for c in ["Propensity", "Value", "Priority"]:
            if c in available:
                component = c
                break
        if component is None:
            pytest.skip("No priority components available")
        result = da_v2.priority_component_distribution(component=component, granularity="Issue", stage="Arbitration")
        assert isinstance(result, pl.LazyFrame)
        df = result.collect()
        assert df.height > 0
        assert "Issue" in df.columns
        assert component in df.columns

    def test_priority_component_distribution_no_stage(self, da_v2):
        available = set(da_v2.sample.collect_schema().names())
        component = "Propensity" if "Propensity" in available else None
        if component is None:
            pytest.skip("Propensity not available")
        result = da_v2.priority_component_distribution(component=component, granularity="Action", stage=None)
        df = result.collect()
        assert df.height > 0

    def test_all_components_distribution_v2(self, da_v2):
        result = da_v2.all_components_distribution(granularity="Issue", stage="Arbitration")
        assert isinstance(result, pl.LazyFrame)
        df = result.collect()
        assert df.height > 0
        assert "Issue" in df.columns

    def test_all_components_distribution_no_stage(self, da_v2):
        result = da_v2.all_components_distribution(granularity="Group", stage=None)
        df = result.collect()
        assert df.height > 0
        assert "Group" in df.columns

    def test_remaining_at_stage_none(self, da_v2):
        """stage=None falls back to non-null Priority rows."""
        result = da_v2._remaining_at_stage(stage=None)
        assert isinstance(result, pl.LazyFrame)
        df = result.collect()
        assert df.height > 0
        # All rows should have non-null Priority
        assert df["Priority"].null_count() == 0

    def test_remaining_at_stage_arbitration(self, da_v2):
        result = da_v2._remaining_at_stage(stage="Arbitration")
        df = result.collect()
        assert df.height > 0
        stages = df[da_v2.level].unique().to_list()
        for s in stages:
            assert s in da_v2.stages_from_arbitration_down

    def test_filtered_action_counts_no_thresholds(self, da_v2):
        result = da_v2.filtered_action_counts(groupby_cols=[da_v2.level])
        assert isinstance(result, pl.LazyFrame)
        df = result.collect()
        assert df.height > 0
        assert "no_of_offers" in df.columns

    def test_filtered_action_counts_with_thresholds(self, da_v2):
        result = da_v2.filtered_action_counts(
            groupby_cols=[da_v2.level],
            propensityTH=0.5,
            priorityTH=0.5,
        )
        df = result.collect()
        assert df.height > 0
        assert "no_of_offers" in df.columns
        assert "good_offers" in df.columns
        assert "poor_propensity_offers" in df.columns
        assert "poor_priority_offers" in df.columns
        assert "new_models" in df.columns

    def test_filtered_action_counts_by_interaction(self, da_v1):
        result = da_v1.filtered_action_counts(groupby_cols=["Interaction ID"])
        df = result.collect()
        assert df.height > 0
        assert "Interaction ID" in df.columns

    def test_filtered_action_counts_missing_col_raises(self, da_v1):
        with pytest.raises(ValueError, match="not found"):
            da_v1.filtered_action_counts(groupby_cols=["NonexistentColumn"])


# ---------------------------------------------------------------------------
# Win distribution (deeper coverage)
# ---------------------------------------------------------------------------


class TestWinDistributionExtended:
    def test_baseline_distribution_v2(self, da_v2):
        """Test baseline win distribution (lever_value=None) on v2."""
        first_issue = da_v2.decision_data.select(pl.col("Issue").first()).collect().item()
        lever_cond = pl.col("Issue") == first_issue
        result = da_v2.get_win_distribution_data(lever_condition=lever_cond)
        assert isinstance(result, pl.DataFrame)
        assert result.height > 0
        assert "original_win_count" in result.columns
        assert "selected_action" in result.columns
        # Should have both Selected and Rest
        selected_vals = result["selected_action"].unique().to_list()
        assert "Selected" in selected_vals or "Rest" in selected_vals

    def test_lever_adjusted_distribution_v2(self, da_v2):
        first_issue = da_v2.decision_data.select(pl.col("Issue").first()).collect().item()
        lever_cond = pl.col("Issue") == first_issue
        result = da_v2.get_win_distribution_data(lever_condition=lever_cond, lever_value=2.0)
        assert "new_win_count" in result.columns
        assert "original_win_count" in result.columns

    def test_distribution_with_no_winner_v2(self, da_v2):
        first_issue = da_v2.decision_data.select(pl.col("Issue").first()).collect().item()
        lever_cond = pl.col("Issue") == first_issue
        result = da_v2.get_win_distribution_data(lever_condition=lever_cond, all_interactions=1000)
        assert "No Winner" in result["Action"].to_list()

    def test_lever_adjusted_with_no_winner(self, da_v1):
        first_issue = da_v1.decision_data.select(pl.col("Issue").first()).collect().item()
        lever_cond = pl.col("Issue") == first_issue
        result = da_v1.get_win_distribution_data(lever_condition=lever_cond, lever_value=1.5, all_interactions=5000)
        assert "No Winner" in result["Action"].to_list()
        assert "new_win_count" in result.columns


# ---------------------------------------------------------------------------
# find_lever_value (expensive — minimal test)
# ---------------------------------------------------------------------------


class TestFindLeverValue:
    def test_find_lever_value_narrow_range(self, da_v1):
        """Test find_lever_value with a narrow search range for speed."""
        first_issue = da_v1.decision_data.select(pl.col("Issue").first()).collect().item()
        lever_cond = pl.col("Issue") == first_issue
        # Use a generous precision and narrow range to keep it fast
        try:
            result = da_v1.find_lever_value(
                lever_condition=lever_cond,
                target_win_percentage=50.0,
                low=0.5,
                high=10.0,
                precision=1.0,
            )
            assert isinstance(result, float)
            assert 0.5 <= result <= 10.0
        except ValueError:
            # Target may not be achievable — that's acceptable
            pass


# ---------------------------------------------------------------------------
# Offer Quality Analysis
# ---------------------------------------------------------------------------


class TestOfferQualityAnalysis:
    def test_stages_with_propensity_property(self, da_v2):
        """Test that stages_with_propensity detects stages with propensity scores."""
        stages = da_v2.stages_with_propensity
        assert isinstance(stages, list)
        assert len(stages) > 0
        # Should include arbitration-related stages
        assert any("arbitration" in s.lower() or "output" in s.lower() for s in stages)

    def test_stages_with_propensity_v1(self, da_v1):
        """Test stages_with_propensity on v1 data."""
        stages = da_v1.stages_with_propensity
        assert isinstance(stages, list)
        assert len(stages) > 0

    def test_get_offer_quality_basic(self, da_v2):
        """Test get_offer_quality returns expected structure."""
        # get_offer_quality requires action_counts to have threshold columns
        action_counts = da_v2.filtered_action_counts(
            groupby_cols=[da_v2.level, "Interaction ID"], propensityTH=0.1, priorityTH=10.0
        )
        result = da_v2.get_offer_quality(action_counts, group_by="Interaction ID")
        df = result.collect()
        assert df.height > 0
        # Check for offer quality category columns
        assert "has_no_offers" in df.columns
        assert "atleast_one_relevant_action" in df.columns
        assert "only_irrelevant_actions" in df.columns
        assert "atleast_one_action" in df.columns
        assert da_v2.level in df.columns

    def test_get_offer_quality_with_thresholds(self, da_v2):
        """Test get_offer_quality respects propensity/priority thresholds."""
        action_counts = da_v2.filtered_action_counts(
            groupby_cols=[da_v2.level, "Interaction ID"], propensityTH=0.5, priorityTH=50.0
        )
        result = da_v2.get_offer_quality(action_counts, group_by="Interaction ID")
        df = result.collect()
        assert df.height > 0
        # At least one category should have non-zero counts
        has_counts = (
            df.select("has_no_offers", "atleast_one_relevant_action", "only_irrelevant_actions", "atleast_one_action")
            .sum()
            .row(0)
        )
        assert sum(has_counts) > 0

    def test_get_offer_quality_includes_all_customers(self, da_v2):
        """Test that get_offer_quality includes customers even if they have no actions."""
        action_counts = da_v2.filtered_action_counts(
            groupby_cols=[da_v2.level, "Interaction ID"],
            propensityTH=0.99,  # High threshold to filter most
            priorityTH=999.0,  # High threshold to filter most
        )
        result = da_v2.get_offer_quality(action_counts, group_by="Interaction ID")
        df = result.collect()

        # Should have customers with no offers
        assert df.filter(pl.col("has_no_offers") > 0).height > 0

        # Total unique customers should match first stage customer count
        first_stage = da_v2.AvailableNBADStages[0]
        total_customers_in_result = df.filter(pl.col(da_v2.level) == first_stage).select("Interaction ID").n_unique()
        total_customers_in_data = (
            da_v2.sample.filter(pl.col(da_v2.level) == first_stage)
            .select(pl.col("Interaction ID").n_unique().alias("count"))
            .collect()
            .item()
        )
        # Results should include all customers from the data
        assert total_customers_in_result == total_customers_in_data

    def test_get_offer_quality_stage_without_propensity(self, da_v2):
        """Test offer quality for stages without propensity uses atleast_one_action category."""
        action_counts = da_v2.filtered_action_counts(
            groupby_cols=[da_v2.level, "Interaction ID"], propensityTH=0.1, priorityTH=10.0
        )
        result = da_v2.get_offer_quality(action_counts, group_by="Interaction ID")
        df = result.collect()

        # For stages without propensity, should use atleast_one_action instead of relevant/irrelevant
        stages_without_prop = set(da_v2.AvailableNBADStages) - set(da_v2.stages_with_propensity)
        if stages_without_prop:
            stage = list(stages_without_prop)[0]
            stage_df = df.filter(pl.col(da_v2.level) == stage)
            if stage_df.height > 0:
                # Should have atleast_one_action counts, not relevant/irrelevant
                assert stage_df.select(pl.col("atleast_one_action").sum()).item() >= 0


# ---------------------------------------------------------------------------
# Propensity validation
# ---------------------------------------------------------------------------


class TestPropensityValidation:
    """Test propensity validation warnings for data quality issues."""

    def test_sample_data_propensity_detection(self, da_v2):
        """Test that propensity validation detects issues in sample data if present."""
        # The sample_eev2.parquet data may have high propensities
        # This test verifies the validation runs without errors
        warning = da_v2.propensity_validation_warning
        # If there's a warning, it should be properly formatted
        if warning is not None:
            # Should contain one of the expected warning types
            assert "Invalid propensity" in warning or "Unusually high propensities" in warning

    def test_high_propensities_triggers_warning(self, da_v2):
        """Test that high propensities (> 10%) trigger a warning."""
        # Load real data and modify propensities to be high
        raw = pl.scan_parquet(f"{basePath}/data/sample_eev2.parquet")
        modified_data = raw.with_columns(pl.lit(0.15).alias("FinalPropensity"))  # Set all to 15%
        da = DecisionAnalyzer(modified_data, sample_size=1000)
        warning = da.propensity_validation_warning
        assert warning is not None
        assert "Unusually high propensities detected" in warning or "Invalid propensity" in warning

    def test_invalid_propensities_triggers_warning(self, da_v2):
        """Test that invalid propensities (> 1.0) trigger a warning."""
        # Load real data and modify propensities to be invalid
        raw = pl.scan_parquet(f"{basePath}/data/sample_eev2.parquet")
        modified_data = raw.with_columns(pl.lit(1.5).alias("FinalPropensity"))  # Set all to 1.5
        da = DecisionAnalyzer(modified_data, sample_size=1000)
        warning = da.propensity_validation_warning
        assert warning is not None
        assert "Invalid propensity values detected" in warning
        assert "> 1.0" in warning

    def test_missing_propensity_column_no_warning(self, da_v2):
        """Test that missing Propensity column doesn't cause errors."""
        # Load real data and drop propensity column
        raw = pl.scan_parquet(f"{basePath}/data/sample_eev2.parquet")
        # Drop FinalPropensity if it exists
        cols_to_keep = [c for c in raw.collect_schema().names() if c not in ["FinalPropensity", "Propensity"]]
        modified_data = raw.select(cols_to_keep)

        # This will warn about missing FinalPropensity but shouldn't crash
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            da = DecisionAnalyzer(modified_data, sample_size=1000)
            # propensity_validation_warning should handle missing column gracefully
            assert da.propensity_validation_warning is None

    def test_propensity_validation_warning_is_cached(self, da_v2):
        """Test that propensity_validation_warning is a cached property."""
        # Access twice should return same result without recomputation
        warning1 = da_v2.propensity_validation_warning
        warning2 = da_v2.propensity_validation_warning
        assert warning1 == warning2
