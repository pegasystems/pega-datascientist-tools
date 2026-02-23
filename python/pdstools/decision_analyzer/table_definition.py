# python/pdstools/decision_analyzer/table_definition.py
from typing import Dict, List, Type, TypedDict

from typing_extensions import NotRequired

import polars as pl


class TableConfig(TypedDict):
    display_name: str
    default: bool
    type: Type[pl.DataType]
    aliases: NotRequired[List[str]]


DecisionAnalyzer: Dict[str, TableConfig] = {
    "pxRecordType": {
        "display_name": "Record Type",
        "default": False,
        "type": pl.Categorical,
    },
    "Primary_pySubjectID": {
        "display_name": "Subject ID",
        "default": False,
        "type": pl.Categorical,
        "aliases": ["Subject ID", "SubjectID", "pySubjectID"],
    },
    "Primary_pySubjectType": {
        "display_name": "Subject Type",
        "default": False,
        "type": pl.Categorical,
        "aliases": ["Subject Type", "SubjectType", "pySubjectType"],
    },
    "pxInteractionID": {
        "display_name": "Interaction ID",
        "default": True,
        "type": pl.Utf8,
        "aliases": ["Interaction ID", "InteractionID"],
    },
    "pxDecisionTime": {
        "display_name": "Decision Time",
        "default": True,
        "type": pl.Datetime,
        "aliases": ["Decision Time", "DecisionTime"],
    },
    "pyIssue": {
        "display_name": "Issue",
        "default": True,
        "type": pl.Categorical,
        "aliases": ["Issue"],
    },
    "pyGroup": {
        "display_name": "Group",
        "default": True,
        "type": pl.Categorical,
        "aliases": ["Group"],
    },
    "pyName": {
        "display_name": "Action",
        "default": True,
        "type": pl.Utf8,
        "aliases": ["Name", "Action"],
    },
    "pyTreatment": {
        "display_name": "Treatment",
        "default": False,
        "type": pl.Utf8,
        "aliases": ["Treatment"],
    },
    "PlacementType": {
        "display_name": "Placement Type",
        "default": False,
        "type": pl.Categorical,
    },
    "pxStrategyName": {
        "display_name": "Strategy Name",
        "default": False,
        "type": pl.Categorical,
        "aliases": ["Strategy Name"],
    },
    "Primary_ContainerPayload_Channel": {
        "display_name": "Channel",
        "default": True,
        "type": pl.Categorical,
        "aliases": ["Channel", "pyChannel"],
    },
    "Primary_ContainerPayload_Direction": {
        "display_name": "Direction",
        "default": True,
        "type": pl.Categorical,
        "aliases": ["Direction", "pyDirection"],
    },
    "Stage_pyName": {
        "display_name": "Stage",
        "default": True,
        "type": pl.Categorical,
    },
    "Stage_pyStageGroup": {
        "display_name": "StageGroup",
        "default": True,
        "type": pl.Categorical,
    },
    "Stage_pyOrder": {
        "display_name": "StageOrder",
        "default": True,
        "type": pl.Int32,
    },
    "pxComponentName": {
        "display_name": "Component Name",
        "default": True,
        "type": pl.Utf8,
        "aliases": ["Component Name"],
    },
    "pxComponentType": {
        "display_name": "Component Type",
        "default": True,
        "type": pl.Utf8,
        "aliases": ["Component Type"],
    },
    "Value": {
        "display_name": "Value",
        "default": True,
        "type": pl.Float64,
    },
    "ContextWeight": {
        "display_name": "Context Weight",
        "default": True,
        "type": pl.Float64,
    },
    "Weight": {
        "display_name": "Levers",
        "default": True,
        "type": pl.Float64,
    },
    "FinalPropensity": {
        "display_name": "Propensity",
        "default": False,
        "type": pl.Float64,
    },
    "Priority": {
        "display_name": "Priority",
        "default": False,
        "type": pl.Float32,
    },
    "pyApplication": {
        "display_name": "Application",
        "default": False,
        "type": pl.Utf8,
        "aliases": ["Application"],
    },
    "pyApplicationVersion": {
        "display_name": "Application Version",
        "default": False,
        "type": pl.Utf8,
        "aliases": ["Application Version"],
    },
}


ExplainabilityExtract: Dict[str, TableConfig] = {
    "pySubjectID": {
        "display_name": "Subject ID",
        "default": True,
        "type": pl.Utf8,
        "aliases": ["Subject ID", "SubjectID"],
    },
    "pxInteractionID": {
        "display_name": "Interaction ID",
        "default": True,
        "type": pl.Utf8,
        "aliases": ["Interaction ID", "InteractionID"],
    },
    "pxDecisionTime": {
        "display_name": "Decision Time",
        "default": True,
        "type": pl.Datetime,
        "aliases": ["Decision Time", "DecisionTime"],
    },
    "pyIssue": {
        "display_name": "Issue",
        "default": True,
        "type": pl.Categorical,
        "aliases": ["Issue"],
    },
    "pyGroup": {
        "display_name": "Group",
        "default": True,
        "type": pl.Categorical,
        "aliases": ["Group"],
    },
    "pyName": {
        "display_name": "Action",
        "default": True,
        "type": pl.Utf8,
        "aliases": ["Name", "Action"],
    },
    # "pyTreatment": {"display_name": "Treatment", "default": False, "type": pl.Utf8},
    "pyChannel": {
        "display_name": "Channel",
        "default": True,
        "type": pl.Categorical,
        "aliases": ["Channel"],
    },
    "pyDirection": {
        "display_name": "Direction",
        "default": True,
        "type": pl.Categorical,
        "aliases": ["Direction"],
    },
    "Value": {"display_name": "Value", "default": True, "type": pl.Float64},
    "ContextWeight": {
        "display_name": "Context Weight",
        "default": True,
        "type": pl.Float64,
    },
    "Weight": {
        "display_name": "Levers",
        "default": True,
        "type": pl.Float64,
    },
    "pyPropensity": {
        "display_name": "pyPropensity",
        "default": True,
        "type": pl.Float64,
    },
    "FinalPropensity": {
        "display_name": "Propensity",
        "default": True,
        "type": pl.Float64,
    },
    "Priority": {
        "display_name": "Priority",
        "default": True,
        "type": pl.Float32,
    },
    "ModelControlGroup": {
        "display_name": "ModelControlGroup",
        "default": True,
        "type": pl.Utf8,
        "aliases": ["ModelControl"],
    },
}


audit_tag_mapping = {
    "AvailableActions": ["ActionAvailability"],
    "ChannelAvailabilityProductAssociations": [
        "InboundChannelAvailability",
        "OutboundChannelAvailability",
        "ActionEmergencyExclusion",
        "InvalidUnsellableProducts",
        "ProductAssociationsExtension",
        "NBAPostActionImportExtension",
    ],
    "EngagementPolicies": [
        "AllActionsEligibility",
        "AllActionsApplicability",
        "AllActionsSuitability",
        "AllIssuesGroupsEligibility",
        "AllIssuesGroupsApplicability",
        "AllIssuesGroupsSuitability",
        "AllIssuesGroupsExtension",
        "IssueGroupEligibility",
        "IssueGroupApplicability",
        "IssueGroupSuitability",
        "IssueGroupExtension",
    ],
    "JourneysContactPolicies": [
        "NBAPreProcessExtension",
        "CustomerJourneysExclusions",
        "ActiveOutboundChannels",
        "ContactPoliciesActionSuppressions",
        "ContactPoliciesOutboundChannelLimits",
    ],
    "ActionPropensity": ["ActionModelMaturity", "ActionPropensityThresholds"],
    "AvailableTreatments": [
        "TreatmentJoins",
        "TreatmentJoinsChannels",
        "TreatmentAvailability",
    ],
    "TreatmentEligibility": [
        "TreatmentEligibilityChannels",
        "TreatmentSuppressions",
        "TreatmentInboundChannelExtension",
        "TreatmentOutboundChannelExtension",
        "TreatmentEmergencyExclusionChannel",
        "TreatmentEmergencyExclusionAllChannels",
    ],
    "TreatmentPropensity": [
        "TreatmentAIModelExtension",
        "TreatmentModelMaturity",
        "TreatmentPropensityThresholds",
        "TreatmentPropensityExtension",
    ],
    "Arbitration": ["ArbitrationPriorityThresholds", "ChannelProcessingExtension"],
    "Bundling": [
        "NBAPostProcessingExtension",
        "BundlingPreExtension",
        "BundlingExclusions",
        "BundlingValidity",
    ],
    "FinalActionsProcessing": [
        "ContactPoliciesActionLimits",
        "InvalidBundleMembers",
        "ContactPoliciesActionLimitsExtension",
        "TreatmentPlacements",
        "FinalProcessingWebChannelExtension",
        "FinalProcessingMobileChannelExtension",
        "FinalProcessingAssistedInboundChannelExtension",
        "FinalProcessingOtherInboundChannelExtension",
        "FinalProcessingOutboundChannelExtension",
        "ContactPoliciesChannelLimits",
        "ContactPoliciesExtension",
        "FinalProcessingPostProcessExtension",
    ],
    "Output": ["Output"],
}
