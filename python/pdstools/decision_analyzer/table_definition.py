from typing import Dict, Type, TypedDict
import polars as pl


class TableConfig(TypedDict):
    label: str
    default: bool
    type: Type[pl.DataType]


DecisionAnalyzer: Dict[str, TableConfig] = {
    "pxRecordType": {
        "label": "pxRecordType",
        "default": False,
        "type": pl.Categorical,
    },
    "Primary_pySubjectID": {
        "label": "pySubjectID",
        "default": False,
        "type": pl.Categorical,
    },  # should be optional
    "Primary_pySubjectType": {
        "label": "pySubjectType",
        "default": False,
        "type": pl.Categorical,
    },
    # "StrategicSegment": {"label": "StrategicSegment", "default": False, "type": pl.Categorical},
    "pxInteractionID": {
        "label": "pxInteractionID",
        "default": True,
        "type": pl.Utf8,
    },
    "pxDecisionTime": {
        "label": "pxDecisionTime",
        "default": True,
        "type": pl.Datetime,
    },
    # "pxRank": {
    #     "label": "pxRank",
    #     "default": True,
    #     "type": pl.Int64,
    # },
    "pyIssue": {
        "label": "pyIssue",
        "default": True,
        "type": pl.Categorical,
    },
    "pyGroup": {
        "label": "pyGroup",
        "default": True,
        "type": pl.Categorical,
    },
    "pyName": {
        "label": "pyName",
        "default": True,
        "type": pl.Utf8,
    },
    "pyTreatment": {
        "label": "pyTreatment",
        "default": False,
        "type": pl.Utf8,
    },
    "PlacementType": {
        "label": "PlacementType",
        "default": False,
        "type": pl.Categorical,
    },
    "pxStrategyName": {
        "label": "pxStrategyName",
        "default": False,
        "type": pl.Categorical,
    },
    # "ChannelSubGroup": {"label": "ChannelSubGroup", "default": False, "type": pl.Categorical},
    # "ChannelGroup": {"label": "ChannelGroup", "default": False, "type": pl.Categorical},
    "Primary_ContainerPayload_Channel": {
        "label": "pyChannel",
        "default": True,
        "type": pl.Categorical,
    },
    "Primary_ContainerPayload_Direction": {
        "label": "pyDirection",
        "default": True,
        "type": pl.Categorical,
    },
    "Stage_pyName": {
        "label": "Stage",
        "default": True,
        "type": pl.Categorical,
    },
    "Stage_pyStageGroup": {
        "label": "StageGroup",
        "default": True,
        "type": pl.Categorical,
    },
    "Stage_pyOrder": {
        "label": "StageOrder",
        "default": True,
        "type": pl.Int32,
    },
    "pxComponentName": {
        "label": "pxComponentName",
        "default": True,
        "type": pl.Utf8,
    },
    "pxComponentType": {
        "label": "pxComponentType",
        "default": True,
        "type": pl.Utf8,
    },
    # "MktType": {"label": "MktType", "default": False, "type": pl.Categorical},
    # "MktValue": {"label": "MktValue", "default": False, "type": pl.Float64},
    # "pyReason": {"label": "pyReason", "default": False, "type": pl.Utf8},
    # "ModelControlGroup": {
    #     "label": "ModelControlGroup",
    #     "default": True,
    #     "type": pl.Categorical,
    # },
    "Value": {
        "label": "Value",
        "default": True,
        "type": pl.Float64,
    },
    "ContextWeight": {
        "label": "Context Weight",
        "default": True,
        "type": pl.Float64,
    },
    "Weight": {
        "label": "Levers",
        "default": True,
        "type": pl.Float64,
    },
    # "ActionWeight": {"label": "ActionWeight", "default": False, "type": pl.Float64},
    # "pyModelPropensity": {
    #     "label": "pyModelPropensity",
    #     "default": False,
    #     "type": pl.Float64,
    # },
    "FinalPropensity": {
        "label": "Propensity",
        "default": False,
        "type": pl.Float64,
    },
    # "pyPropensity": {
    #     "label": "pyPropensity",
    #     "default": False,
    #     "type": pl.Float64,
    # },
    # "pyModelPositives": {
    #     "label": "pyModelPositives",
    #     "default": False,
    #     "type": pl.Float32,
    # },
    # "pyModelPerformance": {
    #     "label": "pyModelPerformance",
    #     "default": False,
    #     "type": pl.Float32,
    # },
    # "pyModelEvidence": {
    #     "label": "pyModelEvidence",
    #     "default": False,
    #     "type": pl.Float32,
    # },
    "Priority": {
        "label": "Priority",
        "default": False,
        "type": pl.Float32,
    },
    "pyApplication": {
        "label": "pyApplication",
        "default": False,
        "type": pl.Float64,
    },
    "pyApplicationVersion": {
        "label": "pyApplicationVersion",
        "default": False,
        "type": pl.Float64,
    },
}


ExplainabilityExtract: Dict[str, TableConfig] = {
    "pySubjectID": {
        "label": "pySubjectID",
        "default": True,
        "type": pl.Utf8,
    },
    "pxInteractionID": {
        "label": "pxInteractionID",
        "default": True,
        "type": pl.Utf8,
    },
    "pxDecisionTime": {
        "label": "pxDecisionTime",
        "default": True,
        "type": pl.Datetime,
    },
    "pyIssue": {
        "label": "pyIssue",
        "default": True,
        "type": pl.Categorical,
    },
    "pyGroup": {
        "label": "pyGroup",
        "default": True,
        "type": pl.Categorical,
    },
    "pyName": {"label": "pyName", "default": True, "type": pl.Utf8},
    # "pyTreatment": {"label": "pyTreatment", "default": False, "type": pl.Utf8},
    "pyChannel": {
        "label": "pyChannel",
        "default": True,
        "type": pl.Categorical,
    },
    "pyDirection": {
        "label": "pyDirection",
        "default": True,
        "type": pl.Categorical,
    },
    "Value": {"label": "Value", "default": True, "type": pl.Float64},
    "ContextWeight": {
        "label": "Context Weight",
        "default": True,
        "type": pl.Float64,
    },
    "Weight": {
        "label": "Levers",
        "default": True,
        "type": pl.Float64,
    },
    # "pyModelPropensity": {
    #     "label": "pyModelPropensity",
    #     "default": True,
    #     "type": pl.Float64,
    # },
    "pyPropensity": {
        "label": "pyPropensity",
        "default": True,
        "type": pl.Float64,
    },
    "FinalPropensity": {
        "label": "Propensity",
        "default": True,
        "type": pl.Float64,
    },
    "Priority": {
        "label": "Priority",
        "default": True,
        "type": pl.Float32,
    },
    "ModelControlGroup": {
        "label": "ModelControlGroup",
        "default": True,
        "type": pl.Utf8,
    },
    # "EvalautionCriteria": {
    #     "label": "EvalautionCriteria",
    #     "default": True,
    #     "type": pl.Utf8,
    # },
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
