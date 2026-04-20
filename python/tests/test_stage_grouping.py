"""Tests for stage_grouping utility."""

import polars as pl

from pdstools.decision_analyzer.DecisionAnalyzer import DecisionAnalyzer
from pdstools.decision_analyzer.stage_grouping import (
    NBAD_PIPELINE,
    get_display_name,
    get_stage_display_name,
    get_stage_group_display_name,
    get_stage_group_for_stage,
)


def test_nbad_pipeline_has_all_stage_groups():
    expected = {
        "AvailableActions",
        "ChannelAvailabilityProductAssociations",
        "EngagementPolicies",
        "JourneysContactPolicies",
        "ActionPropensity",
        "AvailableTreatments",
        "TreatmentEligibility",
        "TreatmentPropensity",
        "Arbitration",
        "Bundling",
        "FinalActionsProcessing",
        "Output",
    }
    assert set(NBAD_PIPELINE.keys()) == expected


def test_get_stage_group_display_name_known():
    assert get_stage_group_display_name("EngagementPolicies") == "Engagement Policies"
    assert get_stage_group_display_name("FinalActionsProcessing") == "Contact Policies and final Action processing"
    assert get_stage_group_display_name("Arbitration") == "Arbitration"


def test_get_stage_group_display_name_unknown_falls_back():
    assert get_stage_group_display_name("SomeUnknownGroup") == "SomeUnknownGroup"


def test_get_stage_display_name_known():
    assert get_stage_display_name("ActionAvailability") == "Action availability"
    assert get_stage_display_name("AllActionsEligibility") == "Issue/Group: All Actions - Eligibility"
    assert get_stage_display_name("BundlingValidity") == "Valid bundles"
    assert get_stage_display_name("ActionModelMaturity") == "Action - Model maturity"
    assert get_stage_display_name("TreatmentModelMaturity") == "Treatment - Model maturity"
    assert get_stage_display_name("ActionPropensityThresholds") == "Action - Propensity thresholds"
    assert get_stage_display_name("TreatmentPropensityThresholds") == "Treatment - Propensity thresholds"


def test_get_stage_display_name_unknown_falls_back():
    assert get_stage_display_name("SomeUnknownStage") == "SomeUnknownStage"


def test_get_display_name_resolves_group_first():
    # Stage groups take priority over stages in get_display_name
    assert get_display_name("EngagementPolicies") == "Engagement Policies"
    assert get_display_name("Arbitration") == "Arbitration"


def test_get_display_name_falls_through_to_stage():
    assert get_display_name("ActionAvailability") == "Action availability"
    assert get_display_name("BundlingValidity") == "Valid bundles"


def test_get_display_name_unknown_falls_back():
    assert get_display_name("Completely Unknown") == "Completely Unknown"


def test_get_stage_group_for_stage_known():
    assert get_stage_group_for_stage("ActionAvailability") == "AvailableActions"
    assert get_stage_group_for_stage("AllActionsEligibility") == "EngagementPolicies"
    assert get_stage_group_for_stage("TreatmentPlacements") == "FinalActionsProcessing"
    assert get_stage_group_for_stage("Output") == "Output"


def test_get_stage_group_for_stage_unknown_returns_none():
    assert get_stage_group_for_stage("SomeUnknownStage") is None


def test_no_duplicate_stage_names_across_groups():
    """Each stage internal name should appear in at most one group."""
    all_stages = []
    for group in NBAD_PIPELINE.values():
        all_stages.extend(group["stages"].keys())
    assert len(all_stages) == len(set(all_stages)), "Duplicate stage internal names found"


def test_display_names_not_empty():
    """Every entry must have a non-empty display name."""
    for group_name, group in NBAD_PIPELINE.items():
        assert group["display_name"], f"Empty display_name for group {group_name}"
        for stage_name, display in group["stages"].items():
            assert display, f"Empty display name for stage {stage_name}"


def _make_da_raw_data() -> pl.LazyFrame:
    """Minimal v2 data with stage columns using internal names."""
    return (
        pl.DataFrame(
            {
                "pxInteractionID": ["I1", "I2"],
                "pxDecisionTime": ["2024-01-01", "2024-01-02"],
                "pyIssue": ["Sales", "Sales"],
                "pyGroup": ["Cards", "Cards"],
                "pyName": ["Action1", "Action2"],
                "Primary_ContainerPayload_Channel": ["Web", "Email"],
                "Primary_ContainerPayload_Direction": ["Inbound", "Outbound"],
                "pxStrategyName": ["NBA", "NBA"],
                "Stage_pyName": ["AllActionsEligibility", "IssueGroupConstraints"],
                "Stage_pyStageGroup": ["EngagementPolicies", "EngagementPolicies"],
                "Stage_pyOrder": [700, 1900],
                "pxComponentName": ["comp1", "comp2"],
                "pxComponentType": ["AllActionsEligibility", "IssueGroupConstraints"],
                "Value": [1.0, 1.0],
                "ContextWeight": [1.0, 1.0],
                "Weight": [1.0, 1.0],
                "FinalPropensity": [0.05, 0.03],
                "Priority": [100.0, 90.0],
            }
        )
        .with_columns(pl.col("pxDecisionTime").str.strptime(pl.Datetime))
        .lazy()
    )


def test_stage_columns_transformed_to_display_names():
    """DecisionAnalyzer should replace internal stage names with display names."""
    da = DecisionAnalyzer(_make_da_raw_data())
    data = da.decision_data.collect()

    assert "Issue/Group: All Actions - Eligibility" in data["Stage"].to_list()
    assert "Issue/Group: Customer Actions - Constraints" in data["Stage"].to_list()
    assert "Engagement Policies" in data["Stage Group"].to_list()
    # Internal names should be gone
    assert "AllActionsEligibility" not in data["Stage"].to_list()
    assert "EngagementPolicies" not in data["Stage Group"].to_list()


def test_available_nbad_stages_uses_display_names():
    """AvailableNBADStages should contain display names, not internal names."""
    da = DecisionAnalyzer(_make_da_raw_data())
    assert "Engagement Policies" in da.AvailableNBADStages
    assert "EngagementPolicies" not in da.AvailableNBADStages


def test_unknown_stage_kept_as_is():
    """Stage values with no mapping should pass through unchanged."""
    raw = _make_da_raw_data().with_columns(
        pl.lit("SomeFutureStage").alias("Stage_pyName"),
        pl.lit("SomeFutureGroup").alias("Stage_pyStageGroup"),
    )
    da = DecisionAnalyzer(raw)
    data = da.decision_data.collect()
    assert "SomeFutureStage" in data["Stage"].to_list()
    assert "SomeFutureGroup" in data["Stage Group"].to_list()
