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
    # "pySubjectType": {"label": "pySubjectType", "default": False, "type": pl.Categorical},
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
    "pxRank": {
        "label": "pxRank",
        "default": True,
        "type": pl.Int64,
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
    "pyName": {
        "label": "pyName",
        "default": True,
        "type": pl.Utf8,
        "required": True,
    },
    # "pyTreatment": {"label": "pyTreatment", "default": False, "type": pl.Utf8},
    # "PlacementType": {"label": "PlacementType", "default": False, "type": pl.Categorical},
    # "ChannelSubGroup": {"label": "ChannelSubGroup", "default": False, "type": pl.Categorical},
    # "ChannelGroup": {"label": "ChannelGroup", "default": False, "type": pl.Categorical},
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
    "pxEngagementStage": {
        "label": "pxEngagementStage",
        "default": True,
        "type": pl.Categorical,
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
    "ModelControlGroup": {
        "label": "ModelControlGroup",
        "default": True,
        "type": pl.Categorical,
        "required": True,
    },
    "Value": {
        "label": "Value",
        "default": True,
        "type": pl.Float64,
        "required": True,
    },
    "ContextWeight": {
        "label": "ContextWeight",
        "default": True,
        "type": pl.Float64,
        "required": True,
    },
    "Weight": {
        "label": "Weight",
        "default": True,
        "type": pl.Float64,
        "required": True,
    },
    # "ActionWeight": {"label": "ActionWeight", "default": False, "type": pl.Float64},
    # "pyModelPropensity": {"label": "pyModelPropensity", "default": False, "type": pl.Float64},
    "FinalPropensity": {
        "label": "FinalPropensity",
        "default": False,
        "type": pl.Float64,
        "required": True,
    },
    # "pyPropensity": {"label": "pyPropensity", "default": False, "type": pl.Float64},
    # "pyModelPositives":{"label": "pyModelPositives", "default": False, "type": pl.Float32},
    # "pyModelPerformance":{"label": "pyModelPerformance", "default": False, "type": pl.Float32},
    # "pyModelEvidence":{"label": "pyModelEvidence", "default": False, "type": pl.Float32},
    "Priority": {
        "label": "Priority",
        "default": False,
        "type": pl.Float32,
        "required": True,
    },
    # "pyApplication":{"label": "pyApplication", "default": False, "type": pl.Float64},
    # "pyApplicationVersion":{"label": "pyApplicationVersion", "default": False, "type": pl.Float64},
}


ExplainabilityExtract: Dict[str, TableConfig] = {
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
        "label": "ContextWeight",
        "default": True,
        "type": pl.Float64,
        "required": True,
    },
    "Weight": {
        "label": "Weight",
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
        "label": "FinalPropensity",
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
