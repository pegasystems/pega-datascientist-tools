"""Canonical NBAD pipeline hierarchy with user-friendly display names.

Maps internal stage group names (``Stage_pyStageGroup``) and stage names
(``Stage_pyName``) to the display names shown in the Pega application.

If a value from the data has no entry here, ``get_display_name`` falls back
to the raw value so future pipeline stages are handled gracefully.
"""

from typing import TypedDict


class StageGroupDef(TypedDict):
    display_name: str
    stages: dict[str, str]  # internal Stage_pyName -> display name


NBAD_PIPELINE: dict[str, StageGroupDef] = {
    "AvailableActions": {
        "display_name": "Available Actions",
        "stages": {
            "ActionAvailability": "Action availability",
        },
    },
    "ChannelAvailabilityProductAssociations": {
        "display_name": "Channel availability and Product associations",
        "stages": {
            "InboundChannelAvailability": "Inbound channel availability",
            "OutboundChannelAvailability": "Outbound channel availability",
            "ActionEmergencyExclusion": "Action emergency exclusions",
            "InvalidUnsellableProducts": "Invalid and unsellable products",
            "ProductAssociationsExtension": "Product associations - Extension",
            "NBAPostActionImportExtension": "Post Action Import - Extension",
        },
    },
    "EngagementPolicies": {
        "display_name": "Engagement Policies",
        "stages": {
            "AllActionsEligibility": "Issue/Group: All Actions - Eligibility",
            "AllActionsApplicability": "Issue/Group: All Actions - Applicability",
            "AllActionsSuitability": "Issue/Group: All Actions - Suitability",
            "AllActionsConstraints": "Issue/Group: All Actions - Constraints",
            "AllIssuesGroupsEligibility": "All Issues/All Groups: Customer - Eligibility",
            "AllIssuesGroupsApplicability": "All Issues/All Groups: Customer - Applicability",
            "AllIssuesGroupsSuitability": "All Issues/All Groups: Customer - Suitability",
            "AllIssuesGroupsExtension": "All Issues/All Groups - Extension",
            "IssueGroupEligibility": "Issue/Group: Customer Actions - Eligibility",
            "IssueGroupApplicability": "Issue/Group: Customer Actions - Applicability",
            "IssueGroupSuitability": "Issue/Group: Customer Actions - Suitability",
            "IssueGroupConstraints": "Issue/Group: Customer Actions - Constraints",
            "IssueGroupExtension": "Issue/Group: Customer Actions - Extension",
        },
    },
    "JourneysContactPolicies": {
        "display_name": "Journeys and Contact Policies",
        "stages": {
            "NBAPreProcessExtension": "Pre-processing - Extension",
            "CustomerJourneysExclusions": "Customer journey exclusions",
            "ContactPoliciesActionSuppressions": "Contact Policies - Action suppressions",
            "ActiveOutboundChannels": "Active outbound channels",
            "ContactPoliciesOutboundChannelLimits": "Contact Policies - Outbound channel limits",
        },
    },
    "ActionPropensity": {
        "display_name": "Action Propensity",
        "stages": {
            "ActionModelMaturity": "Action - Model maturity",
            "ActionPropensityThresholds": "Action - Propensity thresholds",
        },
    },
    "AvailableTreatments": {
        "display_name": "Available Treatments",
        "stages": {
            "TreatmentJoins": "Treatment joins",
            "TreatmentJoinsChannels": "Treatment joins - Channels",
            "TreatmentAvailability": "Treatment availability",
        },
    },
    "TreatmentEligibility": {
        "display_name": "Treatment Eligibility",
        "stages": {
            "TreatmentEligibilityChannels": "Treatment Eligibility",
            "TreatmentSuppressions": "Contact Policies - Treatment suppressions",
            "TreatmentInboundChannelExtension": "Inbound channels - Extension",
            "TreatmentOutboundChannelExtension": "Outbound channels - Extension",
            "TreatmentEmergencyExclusionChannel": "Emergency exclusions by channel",
            "TreatmentEmergencyExclusionAllChannels": "Emergency exclusions for all channels",
        },
    },
    "TreatmentPropensity": {
        "display_name": "Treatment Propensity",
        "stages": {
            "TreatmentAIModelExtension": "AI models propensity - Extension",
            "TreatmentModelMaturity": "Treatment - Model maturity",
            "TreatmentPropensityThresholds": "Treatment - Propensity thresholds",
            "TreatmentPropensityExtension": "Treatment propensity - Extension",
        },
    },
    "Arbitration": {
        "display_name": "Arbitration",
        "stages": {
            "ArbitrationPriorityThresholds": "Priority thresholds",
            "ChannelProcessingExtension": "Channel Processing - Extension",
        },
    },
    "Bundling": {
        "display_name": "Bundling",
        "stages": {
            "NBAPostProcessingExtension": "Post-processing - Extension",
            "BundlingPreExtension": "Pre-bundling - Extension",
            "BundlingExclusions": "Bundle exclusions",
            "BundlingValidity": "Valid bundles",
        },
    },
    "FinalActionsProcessing": {
        "display_name": "Contact Policies and final Action processing",
        "stages": {
            "ContactPoliciesActionLimits": "Contact Policies - Action limits",
            "InvalidBundleMembers": "Invalid bundle members",
            "ContactPoliciesActionLimitsExtension": "Contact Policies - Action limits - Extension",
            "TreatmentPlacements": "Treatment placements",
            "FinalProcessingWebChannelExtension": "Web channel - Extension",
            "FinalProcessingMobileChannelExtension": "Mobile channel - Extension",
            "FinalProcessingAssistedInboundChannelExtension": "Assisted Inbound channel - Extension",
            "FinalProcessingOtherInboundChannelExtension": "Other channel - Extension",
            "FinalProcessingOutboundChannelExtension": "Outbound channel - Extension",
            "ContactPoliciesChannelLimits": "Contact Policies - Channel limits",
            "ContactPoliciesExtension": "Contact Policies - Extension",
            "FinalProcessingPostProcessExtension": "Post-processing - Extension",
        },
    },
    "Output": {
        "display_name": "Output",
        "stages": {
            "Output": "Output",
        },
    },
}

# Flat lookup: internal stage name -> display name (built once at import time)
_STAGE_DISPLAY_NAMES: dict[str, str] = {
    stage: display for group in NBAD_PIPELINE.values() for stage, display in group["stages"].items()
}

# Flat lookup: internal stage name -> internal group name (built once at import time)
_STAGE_TO_GROUP: dict[str, str] = {
    stage: group_name for group_name, group in NBAD_PIPELINE.items() for stage in group["stages"]
}


def get_stage_group_display_name(internal_name: str) -> str:
    """Return the display name for a stage group, or ``internal_name`` if not found."""
    group = NBAD_PIPELINE.get(internal_name)
    return group["display_name"] if group else internal_name


def get_stage_display_name(internal_name: str) -> str:
    """Return the display name for a stage, or ``internal_name`` if not found."""
    return _STAGE_DISPLAY_NAMES.get(internal_name, internal_name)


def get_display_name(internal_name: str) -> str:
    """Return the display name for a stage group or stage.

    Checks stage groups first, then stages. Returns ``internal_name`` if
    no mapping exists — ensuring forward compatibility with new pipeline stages.
    """
    group = NBAD_PIPELINE.get(internal_name)
    if group:
        return group["display_name"]
    return _STAGE_DISPLAY_NAMES.get(internal_name, internal_name)


def get_stage_group_for_stage(stage_name: str) -> str | None:
    """Return the internal stage group name for a given internal stage name.

    Returns ``None`` if the stage is not found. Both the input and the return
    value are internal names (pre-transformation). After data transformation
    the stage-to-group relationship is read directly from the data.
    """
    return _STAGE_TO_GROUP.get(stage_name)
