"""
Testing the functionality of the ADMDatamart Guidelines functions
mainly to confirm syntax
"""

import pytest

from pdstools import CDHGuidelines


@pytest.fixture
def guidance():
    return CDHGuidelines()


def test_metrics(guidance):
    assert len(guidance.standard_configurations) > 0
    assert len(guidance.standard_channels) > 0
    assert guidance.standard_directions == ["Inbound", "Outbound"]
    assert guidance.min("Actions") == 10
    assert guidance.max("Responses") is None
    assert guidance.best_practice_min("Issues") == 5
    assert guidance.best_practice_max("Does not exist") is None


def test_predictions_mapping(guidance):
    df = guidance.get_predictions_channel_mapping()
    assert df.columns == [
        "Prediction",
        "Channel",
        "Direction",
        "isStandardNBADPrediction",
        "isMultiChannelPrediction",
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
    assert 'StrangeChannel' in df['Channel'].to_list()
