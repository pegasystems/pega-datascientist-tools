from typing import Dict, Type, TypedDict
import polars as pl


class TableConfig(TypedDict):
    label: str
    default: bool
    type: Type[pl.DataType]
    required: bool


DecisionAnalyzer: Dict[str, TableConfig] = {
    "pySubjectID": {
        "label": "pySubjectID",
        "default": False,
        "type": pl.Categorical,
        "required": True,
    },  # should be optional
    "pySubjectType": {
        "label": "pySubjectType",
        "default": False,
        "type": pl.Categorical,
        "required": True,
    },
    # "StrategicSegment": {"label": "StrategicSegment", "default": False, "type": pl.Categorical},
    "pxInteractionID": {
        "label": "pxInteractionID",
        "default": True,
        "type": pl.Utf8,
        "required": True,
    },
    "pxDecisionTime": {
        "label": "pxDecisionTime",
        "default": True,
        "type": pl.Datetime,
        "required": True,
    },
    # "pxRank": {
    #     "label": "pxRank",
    #     "default": True,
    #     "type": pl.Int64,
    #     "required": True,
    # },
    "pyIssue": {
        "label": "pyIssue",
        "default": True,
        "type": pl.Categorical,
        "required": True,
    },
    "pyGroup": {
        "label": "pyGroup",
        "default": True,
        "type": pl.Categorical,
        "required": True,
    },
    "pyName": {
        "label": "pyName",
        "default": True,
        "type": pl.Utf8,
        "required": True,
    },
    "pyTreatment": {
        "label": "pyTreatment",
        "default": False,
        "type": pl.Utf8,
        "required": True,
    },
    "PlacementType": {
        "label": "PlacementType",
        "default": False,
        "type": pl.Categorical,
        "required": True,
    },
    "pxStrategyName": {
        "label": "pxStrategyName",
        "default": False,
        "type": pl.Categorical,
        "required": True,
    },
    # "ChannelSubGroup": {"label": "ChannelSubGroup", "default": False, "type": pl.Categorical},
    # "ChannelGroup": {"label": "ChannelGroup", "default": False, "type": pl.Categorical},
    "Primary_ContainerPayload_Channel": {
        "label": "pyChannel",
        "default": True,
        "type": pl.Categorical,
        "required": True,
    },
    "Primary_ContainerPayload_Direction": {
        "label": "pyDirection",
        "default": True,
        "type": pl.Categorical,
        "required": True,
    },
    "Stage_pyName": {
        "label": "Stage",
        "default": True,
        "type": pl.Categorical,
        "required": True,
    },
    "Stage_pyStageGroup": {
        "label": "StageGroup",
        "default": True,
        "type": pl.Categorical,
        "required": True,
    },
    "Stage_pyOrder": {
        "label": "StageOrder",
        "default": True,
        "type": pl.Int32,
        "required": True,
    },
    "pxComponentName": {
        "label": "pxComponentName",
        "default": True,
        "type": pl.Utf8,
        "required": True,
    },
    "pxComponentType": {
        "label": "pxComponentType",
        "default": True,
        "type": pl.Utf8,
        "required": True,
    },
    # "MktType": {"label": "MktType", "default": False, "type": pl.Categorical},
    # "MktValue": {"label": "MktValue", "default": False, "type": pl.Float64},
    # "pyReason": {"label": "pyReason", "default": False, "type": pl.Utf8},
    # "ModelControlGroup": {
    #     "label": "ModelControlGroup",
    #     "default": True,
    #     "type": pl.Categorical,
    #     "required": True,
    # },
    "Value": {
        "label": "Value",
        "default": True,
        "type": pl.Float64,
        "required": True,
    },
    "ContextWeight": {
        "label": "Context Weight",
        "default": True,
        "type": pl.Float64,
        "required": True,
    },
    "Weight": {
        "label": "Levers",
        "default": True,
        "type": pl.Float64,
        "required": True,
    },
    # "ActionWeight": {"label": "ActionWeight", "default": False, "type": pl.Float64},
    "pyModelPropensity": {
        "label": "pyModelPropensity",
        "default": False,
        "type": pl.Float64,
        "required": True,
    },
    "FinalPropensity": {
        "label": "Propensity",
        "default": False,
        "type": pl.Float64,
        "required": True,
    },
    # "pyPropensity": {
    #     "label": "pyPropensity",
    #     "default": False,
    #     "type": pl.Float64,
    #     "required": True,
    # },
    # "pyModelPositives": {
    #     "label": "pyModelPositives",
    #     "default": False,
    #     "type": pl.Float32,
    #     "required": True,
    # },
    # "pyModelPerformance": {
    #     "label": "pyModelPerformance",
    #     "default": False,
    #     "type": pl.Float32,
    #     "required": True,
    # },
    # "pyModelEvidence": {
    #     "label": "pyModelEvidence",
    #     "default": False,
    #     "type": pl.Float32,
    #     "required": True,
    # },
    "Priority": {
        "label": "Priority",
        "default": False,
        "type": pl.Float32,
        "required": True,
    },
    "pyApplication": {
        "label": "pyApplication",
        "default": False,
        "type": pl.Float64,
        "required": True,
    },
    "pyApplicationVersion": {
        "label": "pyApplicationVersion",
        "default": False,
        "type": pl.Float64,
        "required": True,
    },
}


ExplainabilityExtract: Dict[str, TableConfig] = {
    "pySubjectID": {
        "label": "pySubjectID",
        "default": True,
        "type": pl.Utf8,
        "required": False,
    },
    "pxInteractionID": {
        "label": "pxInteractionID",
        "default": True,
        "type": pl.Utf8,
        "required": True,
    },
    "pxDecisionTime": {
        "label": "pxDecisionTime",
        "default": True,
        "type": pl.Datetime,
        "required": True,
    },
    "pyIssue": {
        "label": "pyIssue",
        "default": True,
        "type": pl.Categorical,
        "required": True,
    },
    "pyGroup": {
        "label": "pyGroup",
        "default": True,
        "type": pl.Categorical,
        "required": True,
    },
    "pyName": {"label": "pyName", "default": True, "type": pl.Utf8, "required": True},
    # "pyTreatment": {"label": "pyTreatment", "default": False, "type": pl.Utf8, "required": False},
    "pyChannel": {
        "label": "pyChannel",
        "default": True,
        "type": pl.Categorical,
        "required": True,
    },
    "pyDirection": {
        "label": "pyDirection",
        "default": True,
        "type": pl.Categorical,
        "required": True,
    },
    "Value": {"label": "Value", "default": True, "type": pl.Float64, "required": True},
    "ContextWeight": {
        "label": "Context Weight",
        "default": True,
        "type": pl.Float64,
        "required": True,
    },
    "Weight": {
        "label": "Levers",
        "default": True,
        "type": pl.Float64,
        "required": True,
    },
    "pyModelPropensity": {
        "label": "pyModelPropensity",
        "default": True,
        "type": pl.Float64,
        "required": False,
    },
    "pyPropensity": {
        "label": "pyPropensity",
        "default": True,
        "type": pl.Float64,
        "required": False,
    },
    "FinalPropensity": {
        "label": "Propensity",
        "default": True,
        "type": pl.Float64,
        "required": True,
    },
    "Priority": {
        "label": "Priority",
        "default": True,
        "type": pl.Float32,
        "required": True,
    },
    "ModelControlGroup": {
        "label": "ModelControlGroup",
        "default": True,
        "type": pl.Utf8,
        "required": True,
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
