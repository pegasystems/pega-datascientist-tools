"""
Testing the functionality of the ADMDatamart Guidelines functions
mainly to confirm syntax
"""

import pytest
from pdstools.adm.CDH_Guidelines import CDHGuidelines


@pytest.fixture
def guidance():
    return CDHGuidelines()


def test_metrics(guidance):
    assert len(guidance.standard_configurations) > 0
    assert len(guidance.standard_channels) > 0
    assert guidance.standard_directions == ["Inbound", "Outbound"]
    assert guidance.min("Actions") == 50
    assert guidance.max("Responses") is None
    assert guidance.best_practice_min("Issues") == 5
    assert guidance.best_practice_max("Does not exist") is None


def test_predictions_mapping(guidance):
    df = guidance.get_predictions_channel_mapping()
    assert df.columns == [
        "Prediction",
        "Channel",
        "Direction",
        "isMultiChannel",
    ]
    assert df.shape[0] == 14


def test_predictions_w_extensions_mapping(guidance):
    df = guidance.get_predictions_channel_mapping(
        [
            ["MyWebConversion", "Web", "Inbound", False, False],
            ["MyStrangeChannelPrediction", "StrangeChannel", "Inbound", False, False],
        ]
    )
    assert df.shape[0] == 16
    assert "StrangeChannel" in df["Channel"].to_list()


def test_is_standard_NBAD_configuration(guidance):
    import polars as pl
    df = pl.DataFrame(
        {
            "Configuration": [
                "OmniAdaptiveModel",
                "MyWeb",
                "Mobile_Click_Through_Rate_Account",
                "WEB_CLICK_THROUGH_RATE_AGB",
                "Email_Click_Through_Rate_AGB_Customer",
                "Email_Click_Through_Account_Rate"
            ],
            "Prediction":[
                "PredictActionPropensity",
                "PREDICTWEBPROPENSITY",
                "PredictMobilePropensity",
                "PredictWebPropensity",
                "MyEmailPropensity",
                "PredictOutboundEmailPropensity"

            ]
        }
    )
    df = df.with_columns(
        isNBAD=guidance.is_standard_configuration(),
        isStandardPrediction=guidance.is_standard_prediction(),
    )

    assert df["isNBAD"].to_list() == [True, False, True, True, True, False]
    assert df["isStandardPrediction"].to_list() == [True, True, True, True, False, True]

