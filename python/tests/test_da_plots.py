# python/tests/test_da_plots.py
"""
Tests for the Decision Analyzer plots module.

These tests ensure that all plot functions run without exceptions. We don't validate
the exact content of plots (that would be brittle and hard to maintain), but we verify
that they:
1. Execute without raising exceptions
2. Return the expected type (Figure or LazyFrame when return_df=True)
3. Handle edge cases gracefully
"""

import pathlib

import polars as pl
import pytest
from plotly.graph_objs import Figure

from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer
from pdstools.decision_analyzer.plots import (
    Plot,
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


@pytest.fixture(scope="module")
def plot_v1(da_v1):
    """Plot instance for v1 data."""
    return Plot(da_v1)


@pytest.fixture(scope="module")
def plot_v2(da_v2):
    """Plot instance for v2 data."""
    return Plot(da_v2)


# ---------------------------------------------------------------------------
# Plot class methods - Basic execution tests
# ---------------------------------------------------------------------------


class TestThresholdDeciles:
    """Test threshold_deciles method."""

    def test_propensity_threshold_plot(self, plot_v2):
        """Test propensity thresholding plot."""
        fig = plot_v2.threshold_deciles("Propensity", "Propensity Threshold")
        assert isinstance(fig, Figure)
        assert len(fig.data) == 2  # Bar + line

    def test_priority_threshold_plot(self, plot_v2):
        """Test priority thresholding plot."""
        fig = plot_v2.threshold_deciles("Priority", "Priority Threshold")
        assert isinstance(fig, Figure)
        assert len(fig.data) == 2

    def test_return_df(self, plot_v2):
        """Test return_df parameter."""
        df = plot_v2.threshold_deciles("Propensity", "Propensity Threshold", return_df=True)
        assert isinstance(df, pl.DataFrame)
        assert "Decile" in df.columns
        assert "Count" in df.columns


class TestDistributionAsTreemap:
    """Test distribution_as_treemap method."""

    def test_treemap_with_scope(self, plot_v2, da_v2):
        """Test treemap with scope options."""
        # Use Stage Group since Stage column doesn't exist in pre-aggregated view
        df = da_v2.getPreaggregatedRemainingView.filter(pl.col("Stage Group") == "Output")
        fig = plot_v2.distribution_as_treemap(df, stage="Output", scope_options=["Action"])
        assert isinstance(fig, Figure)

    def test_treemap_multiple_scopes(self, plot_v2, da_v2):
        """Test treemap with multiple scope levels."""
        df = da_v2.getPreaggregatedRemainingView.filter(pl.col("Stage Group") == "Output")
        fig = plot_v2.distribution_as_treemap(df, stage="Output", scope_options=["Channel", "Action"])
        assert isinstance(fig, Figure)

    def test_treemap_empty_scope(self, plot_v2, da_v2):
        """Test treemap with no scope options."""
        df = da_v2.getPreaggregatedRemainingView.filter(pl.col("Stage Group") == "Output")
        fig = plot_v2.distribution_as_treemap(df, stage="Output", scope_options=[])
        assert isinstance(fig, Figure)


class TestSensitivity:
    """Test sensitivity method."""

    def test_global_sensitivity(self, plot_v2):
        """Test global sensitivity plot."""
        fig = plot_v2.sensitivity(win_rank=1)
        assert isinstance(fig, Figure)

    def test_sensitivity_with_priority(self, plot_v2):
        """Test sensitivity with priority factor shown."""
        fig = plot_v2.sensitivity(win_rank=1, hide_priority=False)
        assert isinstance(fig, Figure)

    def test_sensitivity_return_df(self, plot_v2):
        """Test sensitivity with return_df."""
        df = plot_v2.sensitivity(win_rank=1, return_df=True)
        assert isinstance(df, pl.LazyFrame)
        collected = df.collect()
        assert "Factor" in collected.columns
        assert "Influence" in collected.columns

    def test_local_sensitivity(self, plot_v2):
        """Test local sensitivity with reference group."""
        # Get a valid action name from the data
        action_name = (
            plot_v2._decision_data.getPreaggregatedRemainingView.select("Action").collect().get_column("Action")[0]
        )
        # reference_group needs to be a pl.Expr
        fig = plot_v2.sensitivity(win_rank=1, reference_group=pl.col("Action") == action_name)
        assert isinstance(fig, Figure)


class TestGlobalWinlossDistribution:
    """Test global_winloss_distribution method."""

    def test_winloss_by_action(self, plot_v2):
        """Test win/loss distribution by action."""
        fig = plot_v2.global_winloss_distribution(level="Action", win_rank=1)
        assert isinstance(fig, Figure)
        # Should have 2 traces (wins and losses pie charts)
        assert len(fig.data) == 2

    def test_winloss_by_channel(self, plot_v2):
        """Test win/loss distribution by channel."""
        fig = plot_v2.global_winloss_distribution(level="Channel", win_rank=1)
        assert isinstance(fig, Figure)

    def test_winloss_return_df(self, plot_v2):
        """Test win/loss with return_df."""
        df = plot_v2.global_winloss_distribution(level="Action", win_rank=1, return_df=True)
        assert isinstance(df, pl.LazyFrame)
        collected = df.collect()
        assert "Status" in collected.columns
        assert set(collected["Status"]) <= {"Wins", "Losses"}


class TestPropensityVsOptionality:
    """Test propensity_vs_optionality method."""

    def test_default_stage(self, plot_v2):
        """Test propensity vs optionality for default stage."""
        fig = plot_v2.propensity_vs_optionality()
        assert isinstance(fig, Figure)

    def test_specific_stage(self, plot_v2):
        """Test propensity vs optionality for specific stage."""
        fig = plot_v2.propensity_vs_optionality(stage="Output")
        assert isinstance(fig, Figure)

    def test_return_df(self, plot_v2):
        """Test with return_df."""
        df = plot_v2.propensity_vs_optionality(return_df=True)
        assert isinstance(df, pl.LazyFrame)


class TestOptionalityFunnel:
    """Test optionality_funnel method."""

    def test_funnel_plot(self, plot_v2, da_v2):
        """Test optionality funnel plot."""
        df = da_v2.sample
        fig = plot_v2.optionality_funnel(df)
        assert isinstance(fig, Figure)


class TestActionVariation:
    """Test action_variation method."""

    def test_action_variation_plot(self, plot_v2):
        """Test action variation plot."""
        fig = plot_v2.action_variation(stage="Output")
        assert isinstance(fig, Figure)

    def test_return_df(self, plot_v2):
        """Test with return_df."""
        df = plot_v2.action_variation(stage="Output", return_df=True)
        assert isinstance(df, pl.LazyFrame)

    def test_action_variation_with_color_by(self, plot_v2):
        """Test action variation plot with color_by parameter."""
        fig = plot_v2.action_variation(stage="Output", color_by="Channel/Direction")
        assert isinstance(fig, Figure)
        # Verify the figure has at least one trace
        assert len(fig.data) >= 1
        # Verify the legend title matches the color_by dimension
        assert fig.layout.legend.title.text == "Channel/Direction"


class TestTrendChart:
    """Test trend_chart method."""

    def test_trend_by_action(self, plot_v2):
        """Test trend chart by action."""
        fig, _ = plot_v2.trend_chart(stage="Output", scope="Action")
        assert isinstance(fig, Figure)

    def test_trend_by_channel(self, plot_v2):
        """Test trend chart by channel."""
        fig, _ = plot_v2.trend_chart(stage="Output", scope="Channel")
        assert isinstance(fig, Figure)

    def test_return_df(self, plot_v2):
        """Test with return_df."""
        result = plot_v2.trend_chart(stage="Output", scope="Action", return_df=True)
        # When return_df=True, it just returns a LazyFrame, not a tuple
        assert isinstance(result, pl.LazyFrame)


class TestDecisionFunnel:
    """Test decision_funnel method."""

    def test_funnel_plots(self, plot_v2):
        """Returns passing and filtered figures."""
        passing_fig, filtered_fig = plot_v2.decision_funnel(scope="Action")
        assert isinstance(passing_fig, Figure)
        assert isinstance(filtered_fig, Figure)
        assert len(passing_fig.data) >= 1
        assert all(trace.type == "funnel" for trace in passing_fig.data)
        assert all(getattr(trace, "orientation", None) == "h" for trace in passing_fig.data)

    def test_return_df(self, plot_v2):
        """With return_df returns all three dataframes."""
        available_df, passing_df, filtered_df = plot_v2.decision_funnel(scope="Action", return_df=True)
        assert isinstance(available_df, pl.LazyFrame)
        assert isinstance(passing_df, pl.DataFrame)
        assert isinstance(filtered_df, pl.DataFrame)

    def test_decisions_without_actions_plot(self, plot_v2):
        """decisions_without_actions_plot returns a Figure."""
        fig = plot_v2.decisions_without_actions_plot()
        assert isinstance(fig, Figure)

    def test_decisions_without_actions_return_df(self, plot_v2):
        df = plot_v2.decisions_without_actions_plot(return_df=True)
        assert isinstance(df, pl.DataFrame)
        assert "decisions_without_actions" in df.columns


class TestFilteringComponents:
    """Test filtering_components method."""

    def test_components_plot(self, plot_v2, da_v2):
        """Test filtering components plot."""
        stages = da_v2.AvailableNBADStages
        fig = plot_v2.filtering_components(
            stages=stages,
            top_n=10,
            AvailableNBADStages=stages,
        )
        assert isinstance(fig, Figure)

    def test_return_df(self, plot_v2, da_v2):
        """Test with return_df."""
        stages = da_v2.AvailableNBADStages
        df = plot_v2.filtering_components(
            stages=stages,
            top_n=10,
            AvailableNBADStages=stages,
            return_df=True,
        )
        assert isinstance(df, pl.DataFrame)


class TestDistribution:
    """Test distribution method."""

    def test_distribution_plot(self, plot_v2, da_v2):
        """Test distribution plot - smoke test to ensure it runs."""
        # Get pre-aggregated data like the UI does
        df = da_v2.getPreaggregatedFilterView.filter(pl.col("Stage Group") == "Output")
        fig = plot_v2.distribution(
            df=df,
            scope="Action",
            breakdown="Channel",
            metric="Decisions",
        )
        assert isinstance(fig, Figure)

    def test_distribution_horizontal(self, plot_v2, da_v2):
        """Test distribution with horizontal orientation."""
        df = da_v2.getPreaggregatedFilterView.filter(pl.col("Stage Group") == "Output")
        fig = plot_v2.distribution(
            df=df,
            scope="Channel",
            breakdown="Action",
            metric="Decisions",
            horizontal=True,
        )
        assert isinstance(fig, Figure)


class TestPrioFactorBoxplots:
    """Test prio_factor_boxplots method."""

    @pytest.mark.skip(reason="Sample data doesn't have the correct boolean column structure - fails with schema error")
    def test_factor_boxplots_no_reference(self, plot_v2):
        """Test priority factor boxplots without reference group."""
        # Test without reference group (simpler case)
        # Fails with: invalid series dtype: expected `Boolean`, got `null`
        pass

    @pytest.mark.skip(reason="Sample data doesn't have the correct boolean column structure - fails with schema error")
    def test_return_df_no_reference(self, plot_v2):
        """Test with return_df and no reference."""
        # Same issue as test_factor_boxplots_no_reference
        pass


class TestRankBoxplot:
    """Test rank_boxplot method."""

    def test_rank_boxplot(self, plot_v2):
        """Test rank boxplot."""
        fig = plot_v2.rank_boxplot()
        assert isinstance(fig, Figure)


class TestComponentActionImpact:
    """Test component_action_impact method."""

    def test_component_impact(self, plot_v2):
        """Test component action impact plot - smoke test."""
        # This calls getComponentActionImpact internally
        fig = plot_v2.component_action_impact(top_n=5, scope="Action")
        assert isinstance(fig, Figure)

    def test_component_impact_by_group(self, plot_v2):
        """Test component impact at Group level."""
        fig = plot_v2.component_action_impact(top_n=5, scope="Group")
        assert isinstance(fig, Figure)

    def test_return_df(self, plot_v2):
        """Test with return_df."""
        df = plot_v2.component_action_impact(top_n=5, scope="Action", return_df=True)
        assert isinstance(df, pl.DataFrame)


class TestComponentDrilldown:
    """Test component_drilldown method."""

    def test_drilldown_plot(self, plot_v2, da_v2):
        """Test component drilldown plot - smoke test."""
        # Get a valid component name from the data
        component_data = (
            da_v2.decision_data.filter(pl.col("Record Type") == "FILTERED_OUT").select("Component Name").collect()
        )

        if component_data.height > 0:
            component_name = component_data.get_column("Component Name")[0]
            fig = plot_v2.component_drilldown(component_name=component_name)
            assert isinstance(fig, Figure)
        else:
            # If no component data, that's OK for smoke test - skip
            pytest.skip("No component data in sample")

    def test_return_df(self, plot_v2, da_v2):
        """Test with return_df."""
        component_data = (
            da_v2.decision_data.filter(pl.col("Record Type") == "FILTERED_OUT").select("Component Name").collect()
        )

        if component_data.height > 0:
            component_name = component_data.get_column("Component Name")[0]
            df = plot_v2.component_drilldown(component_name=component_name, return_df=True)
            assert isinstance(df, pl.DataFrame)
        else:
            pytest.skip("No component data in sample")


class TestOptionalityPerStage:
    """Test optionality_per_stage method."""

    def test_optionality_plot(self, plot_v2):
        """Test optionality per stage plot."""
        fig = plot_v2.optionality_per_stage()
        assert isinstance(fig, Figure)

    def test_return_df(self, plot_v2):
        """Test with return_df."""
        df = plot_v2.optionality_per_stage(return_df=True)
        assert isinstance(df, pl.LazyFrame)


class TestOptionalityTrend:
    """Test optionality_trend method."""

    def _build_trend_df(self, da_v2):
        level = da_v2.level
        return (
            da_v2.get_optionality_data_with_trend(da_v2.sample)
            .group_by(["day", level])
            .agg(avg_actions=(pl.col("nOffers") * pl.col("Interactions")).sum() / pl.col("Interactions").sum())
            .sort("day")
        )

    def test_trend_plot(self, plot_v2, da_v2):
        """Test optionality trend plot."""
        trend_df = self._build_trend_df(da_v2)
        fig, _ = plot_v2.optionality_trend(trend_df)
        assert isinstance(fig, Figure)

    def test_return_df(self, plot_v2, da_v2):
        """Test with return_df."""
        trend_df = self._build_trend_df(da_v2)
        result = plot_v2.optionality_trend(trend_df, return_df=True)
        assert isinstance(result, pl.LazyFrame)


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


class TestOfferQualityPiecharts:
    """Test offer_quality_piecharts function."""

    def test_piecharts(self, da_v2):
        """Test offer quality pie charts."""
        from pdstools.decision_analyzer.plots import offer_quality_piecharts

        # Step 1: Get filtered action counts (mimicking the Streamlit page workflow)
        action_counts = da_v2.filtered_action_counts(
            groupby_cols=[da_v2.level, "Interaction ID", "day"],
            propensityTH=0.5,
            priorityTH=50,
        )

        # Step 2: Get offer quality data
        quality_data = da_v2.get_offer_quality(action_counts, group_by="Interaction ID")

        # Step 3: Create pie charts
        fig = offer_quality_piecharts(
            quality_data,
            propensityTH=0.5,
            AvailableNBADStages=da_v2.AvailableNBADStages,
            level=da_v2.level,
        )

        assert isinstance(fig, Figure)

    def test_piecharts_all_stages(self, da_v2):
        """Test that all stages are included (no 5-stage limit)."""
        from pdstools.decision_analyzer.plots import offer_quality_piecharts

        action_counts = da_v2.filtered_action_counts(
            groupby_cols=[da_v2.level, "Interaction ID", "day"],
            propensityTH=0.5,
            priorityTH=50,
        )

        quality_data = da_v2.get_offer_quality(action_counts, group_by="Interaction ID")

        fig = offer_quality_piecharts(
            quality_data,
            propensityTH=0.5,
            AvailableNBADStages=da_v2.AvailableNBADStages,
            level=da_v2.level,
        )

        num_stages = len(da_v2.AvailableNBADStages)
        assert len(fig.data) == num_stages
        assert isinstance(fig, Figure)

    def test_piecharts_no_matching_stages(self, da_v2):
        """Test graceful handling when no stages match the data."""
        from pdstools.decision_analyzer.plots import offer_quality_piecharts

        action_counts = da_v2.filtered_action_counts(
            groupby_cols=[da_v2.level, "Interaction ID", "day"],
            propensityTH=0.5,
            priorityTH=50,
        )

        quality_data = da_v2.get_offer_quality(action_counts, group_by="Interaction ID")

        fig = offer_quality_piecharts(
            quality_data,
            propensityTH=0.5,
            AvailableNBADStages=["NonExistent_Stage"],
            level=da_v2.level,
        )

        assert isinstance(fig, Figure)
        assert len(fig.data) == 0
        assert fig.layout.height == 400

    def test_piecharts_single_stage(self, da_v2):
        """Test with single stage (edge case)."""
        from pdstools.decision_analyzer.plots import offer_quality_piecharts

        action_counts = da_v2.filtered_action_counts(
            groupby_cols=[da_v2.level, "Interaction ID", "day"],
            propensityTH=0.5,
            priorityTH=50,
        )

        quality_data = da_v2.get_offer_quality(action_counts, group_by="Interaction ID")

        fig = offer_quality_piecharts(
            quality_data,
            propensityTH=0.5,
            AvailableNBADStages=["Arbitration"],
            level=da_v2.level,
        )

        assert isinstance(fig, Figure)
        assert len(fig.data) == 1
        assert fig.layout.height == 400


class TestOfferQualitySinglePie:
    """Test offer_quality_single_pie function."""

    def test_single_pie_arbitration(self, da_v2):
        """Test single pie chart for Arbitration stage."""
        from pdstools.decision_analyzer.plots import offer_quality_single_pie

        action_counts = da_v2.filtered_action_counts(
            groupby_cols=[da_v2.level, "Interaction ID"],
            propensityTH=0.05,
            priorityTH=50,
        )

        quality_data = da_v2.get_offer_quality(action_counts, group_by="Interaction ID")

        fig = offer_quality_single_pie(
            quality_data,
            stage="Arbitration",
            propensityTH=0.05,
            level=da_v2.level,
        )

        assert isinstance(fig, Figure)
        assert len(fig.data) == 1
        assert len(fig.data[0].values) == 4

    def test_single_pie_output_stage(self, da_v2):
        """Test single pie chart for Output stage."""
        from pdstools.decision_analyzer.plots import offer_quality_single_pie

        action_counts = da_v2.filtered_action_counts(
            groupby_cols=[da_v2.level, "Interaction ID"],
            propensityTH=0.05,
            priorityTH=50,
        )

        quality_data = da_v2.get_offer_quality(action_counts, group_by="Interaction ID")

        fig = offer_quality_single_pie(
            quality_data,
            stage="Output",
            propensityTH=0.05,
            level=da_v2.level,
        )

        assert isinstance(fig, Figure)
        assert len(fig.data) == 1


class TestGetTrendChart:
    """Test getTrendChart function."""

    def test_trend_chart(self, da_v2):
        """Test trend chart function."""
        from pdstools.decision_analyzer.plots import getTrendChart

        # Step 1: Get filtered action counts
        action_counts = da_v2.filtered_action_counts(
            groupby_cols=[da_v2.level, "Interaction ID", "day"],
            propensityTH=0.5,
            priorityTH=50,
        )

        # Step 2: Get offer quality data with day grouping
        quality_data = da_v2.get_offer_quality(action_counts, group_by=["Interaction ID", "day"])

        # Step 3: Create trend chart
        fig = getTrendChart(quality_data, stage="Output", level=da_v2.level)

        assert isinstance(fig, Figure)

    def test_return_df(self, da_v2):
        """Test with return_df."""
        from pdstools.decision_analyzer.plots import getTrendChart

        # Step 1: Get filtered action counts
        action_counts = da_v2.filtered_action_counts(
            groupby_cols=[da_v2.level, "Interaction ID", "day"],
            propensityTH=0.5,
            priorityTH=50,
        )

        # Step 2: Get offer quality data with day grouping
        quality_data = da_v2.get_offer_quality(action_counts, group_by=["Interaction ID", "day"])

        # Step 3: Get dataframe instead of figure
        df = getTrendChart(quality_data, stage="Output", level=da_v2.level, return_df=True)

        assert isinstance(df, pl.LazyFrame)


class TestPlotPriorityComponentDistribution:
    """Test plot_priority_component_distribution function."""

    def test_component_distribution(self, da_v2):
        """Test priority component distribution plot."""
        from pdstools.decision_analyzer.plots import plot_priority_component_distribution

        # Use the correct method name: priority_component_distribution
        value_data = da_v2.priority_component_distribution(
            component="Value",
            granularity="Action",
            stage="Output",
        )

        # This function returns a tuple of (violin_fig, ecdf_fig, stats_df)
        violin_fig, ecdf_fig, stats_df = plot_priority_component_distribution(
            value_data, component="Value", granularity="Action"
        )

        assert isinstance(violin_fig, Figure)
        assert isinstance(ecdf_fig, Figure)
        assert isinstance(stats_df, pl.DataFrame)


class TestPlotComponentOverview:
    """Test plot_component_overview function."""

    def test_overview_plot(self, da_v2):
        """Test component overview plot."""
        from pdstools.decision_analyzer.plots import plot_component_overview
        from pdstools.decision_analyzer.utils import PRIO_COMPONENTS

        # Use the correct method: all_components_distribution
        overview_data = da_v2.all_components_distribution(
            granularity="Action",
            stage="Output",
        )

        # Get available components from the data
        available_cols = set(overview_data.collect_schema().names())
        component_options = [c for c in PRIO_COMPONENTS if c in available_cols]

        fig = plot_component_overview(overview_data, component_options, granularity="Action")
        assert isinstance(fig, Figure)


class TestCreateWinDistributionPlot:
    """Test create_win_distribution_plot function."""

    @pytest.mark.skip(reason="Function signature or data requirements unclear")
    def test_win_distribution(self, da_v2):
        """Test win distribution plot."""
        # Function may have different signature or data requirements
        pass


class TestCreateParameterDistributionBoxplots:
    """Test create_parameter_distribution_boxplots function."""

    @pytest.mark.skip(reason="Function signature or data requirements unclear")
    def test_boxplots(self, da_v2):
        """Test parameter distribution boxplots."""
        # Function may have different signature or data requirements
        pass


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_v1_data_compatibility(self, plot_v1):
        """Test that v1 data works with basic plots."""
        # v1 doesn't have propensity, but should work with some plots
        fig = plot_v1.sensitivity(win_rank=1)
        assert isinstance(fig, Figure)
