from collections import namedtuple
from typing import List

NBAD_ModelConfiguration = namedtuple(
    "NBAD_ModelConfiguration",
    ["model_name", "channel", "direction", "standard", "multi_channel"],
)
standardNBADModelConfigurations: List[NBAD_ModelConfiguration] = [
    NBAD_ModelConfiguration("Web_Click_Through_Rate", "Web", "Inbound", True, False),
    NBAD_ModelConfiguration("WebTreatmentClickModel", "Web", "Inbound", True, False),
    NBAD_ModelConfiguration(
        "Mobile_Click_Through_Rate", "Mobile", "Inbound", True, False
    ),
    NBAD_ModelConfiguration(
        "Email_Click_Through_Rate", "E-mail", "Outbound", True, False
    ),
    NBAD_ModelConfiguration("Push_Click_Through_Rate", "Push", "Outbound", True, False),
    NBAD_ModelConfiguration("SMS_Click_Through_Rate", "SMS", "Outbound", True, False),
    NBAD_ModelConfiguration(
        "Retail_Click_Through_Rate", "Retail", "Inbound", True, False
    ),
    NBAD_ModelConfiguration(
        "Retail_Click_Through_Rate_Outbound", "Retail", "Outbound", True, False
    ),
    NBAD_ModelConfiguration(
        "CallCenter_Click_Through_Rate", "Call Center", "Inbound", True, False
    ),
    NBAD_ModelConfiguration(
        "CallCenterAcceptRateOutbound", "Call Center", "Outbound", True, False
    ),
    NBAD_ModelConfiguration(
        "Assisted_Click_Through_Rate",
        "Assisted",
        "Inbound",
        True,
        False,
    ),  # withdrawn record
    NBAD_ModelConfiguration(
        "Assisted_Click_Through_Rate_Outbound",
        "Assisted",
        "Outbound",
        True,
        False,
    ),  # withdrawn record
    NBAD_ModelConfiguration("Default_Inbound_Model", "Default", "Inbound", True, False),
    NBAD_ModelConfiguration(
        "Default_Outbound_Model", "Default", "Outbound", True, False
    ),
    NBAD_ModelConfiguration(
        "Default_Click_Through_Rate", "Other", "Inbound", True, False
    ),
    NBAD_ModelConfiguration(
        "Other_Inbound_Click_Through_Rate", "Other", "Inbound", True, False
    ),
    NBAD_ModelConfiguration(
        "OmniAdaptiveModel", "Multi-channel", "Multi-channel", True, True
    ),
]

NBAD_Prediction = namedtuple(
    "NBAD_Prediction",
    ["model_name", "channel", "direction", "standard", "multi_channel"],
)
standardNBADPredictions: List[NBAD_Prediction] = [
    NBAD_Prediction("PredictWebPropensity", "Web", "Inbound", True, False),
    NBAD_Prediction("PredictMobilePropensity", "Mobile", "Inbound", True, False),
    NBAD_Prediction(
        "PredictOutboundEmailPropensity", "E-mail", "Outbound", True, False
    ),
    NBAD_Prediction("PredictOutboundPushPropensity", "Push", "Outbound", True, False),
    NBAD_Prediction("PredictOutboundSMSPropensity", "SMS", "Outbound", True, False),
    NBAD_Prediction("PredictInboundRetailPropensity", "Retail", "Inbound", True, False),
    NBAD_Prediction(
        "PredictOutboundRetailPropensity", "Retail", "Outbound", True, False
    ),
    NBAD_Prediction(
        "PredictInboundCallCenterPropensity", "Call Center", "Inbound", True, False
    ),
    NBAD_Prediction(
        "PredictOutboundCallCenterPropensity", "Call Center", "Outbound", True, False
    ),
    NBAD_Prediction(
        "PredictInboundDefaultPropensity", "Default", "Inbound", True, False
    ),
    NBAD_Prediction(
        "PredictOutboundDefaultPropensity", "Default", "Outbound", True, False
    ),
    NBAD_Prediction("PredictInboundOtherPropensity", "Other", "Inbound", True, False),
    NBAD_Prediction(
        "PredictActionPropensity", "Multi-channel", "Multi-channel", True, True
    ),
    NBAD_Prediction(
        "PredictTreatmentPropensity",
        "Multi-channel",
        "Multi-channel",
        True,
        True,
    ),  # no longer a prediction in 24.1 release
]
