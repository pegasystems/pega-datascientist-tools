import pytest
from pdstools.infinity.resources.prediction_studio.v24_1.prediction import Prediction

mock_prediction = {
    "predictionId": "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS",
    "label": "Predict Cards Acceptance",
    "objective": "Accept",
    "subject": "Customer",
    "status": "Completed",
    "lastUpdateTime": "20240718T120557.925 GMT",
}


@pytest.fixture
def prediction_client(mocker):
    client = mocker.MagicMock()
    return Prediction(client=client, **mock_prediction)


mock_prediction_describe = {
    "predictionId": "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS",
    "label": "Predict Cards Acceptance",
    "objective": "Accept",
    "subject": "Customer",
    "outcomeType": "Binary",
    "responseTimeoutType": "Indefinitely",
    "responseTimeoutValue": "-999",
    "responseTimeoutUnit": "--",
    "delayedlearningtype": "MakeDecision",
    "targetLabels": [{"label": "Accept"}],
    "alternateLabels": [{"label": "Reject"}],
    "context": [
        {
            "contextName": "NoContext",
            "defaultModels": [
                {
                    "id": "CDHSample-Data-Customer!Adm_16330376371",
                    "label": "Accept",
                    "role": "ACTIVE",
                    "type": "Adaptive model",
                    "performance": 72.51,
                    "performanceMeasure": "AUC",
                    "componentName": "Accept",
                    "modelingTechnique": "Adaptive model - Bayesian",
                }
            ],
            "categoryModels": [
                {
                    "id": "@baseclass!testModel_falcons",
                    "label": "testModel_falcons",
                    "role": "ACTIVE",
                    "type": "Adaptive model",
                    "performance": 0.0,
                    "categoryName": "Retention",
                    "componentName": "testModel_falcons",
                    "modelingTechnique": "Adaptive model - Bayesian",
                }
            ],
        }
    ],
    "supportingModels": [
        {
            "id": "CDHSample-Data-Customer!RiskModel",
            "label": "Credit Risk Model",
            "role": "ACTIVE",
            "type": "Predictive model",
            "performance": 50.61,
            "performanceMeasure": "AUC",
            "componentName": "RiskModel",
            "contextName": "NoContext",
            "modelingTechnique": "GBM and XGBoost",
        }
    ],
    "metrics": {"lift": 0.0, "performance": 68.87, "performanceMeasure": "AUC"},
}

mock_performance_metric = {
    "monitoringData": [
        {"value": "59.55", "snapshotTime": "2024-07-04T12:00:00.000Z"},
        {"value": "58.22", "snapshotTime": "2024-07-05T12:00:00.000Z"},
        {"value": "74.29", "snapshotTime": "2024-07-06T12:00:00.000Z"},
        {"value": "74.22", "snapshotTime": "2024-07-07T12:00:00.000Z"},
        {"value": "67.66", "snapshotTime": "2024-07-08T12:00:00.000Z"},
        {"value": "70.13", "snapshotTime": "2024-07-09T12:00:00.000Z"},
        {"value": "77.45", "snapshotTime": "2024-07-10T12:00:00.000Z"},
        {"value": "68.87", "snapshotTime": "2024-07-11T12:00:00.000Z"},
    ]
}


def test_prediction_describe(prediction_client, mocker):
    mock_get = mocker.patch.object(
        prediction_client._client, "get", return_value=mock_prediction_describe
    )

    result = prediction_client.describe()

    mock_get.assert_called_once_with(
        "/prweb/api/PredictionStudio/v3/predictions/CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
    )

    assert (
        result["predictionId"] == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
    )
    assert result["label"] == "Predict Cards Acceptance"
    assert result["objective"] == "Accept"
    assert result["subject"] == "Customer"
    assert result["outcomeType"] == "Binary"
    assert result["responseTimeoutType"] == "Indefinitely"
    assert result["responseTimeoutValue"] == "-999"
    assert result["responseTimeoutUnit"] == "--"
    assert result["delayedlearningtype"] == "MakeDecision"
    assert result["targetLabels"] == [{"label": "Accept"}]
    assert result["alternateLabels"] == [{"label": "Reject"}]
    assert result["metrics"] == {
        "lift": 0.0,
        "performance": 68.87,
        "performanceMeasure": "AUC",
    }


def test_prediction_get_metric(prediction_client, mocker):
    mock_get = mocker.patch.object(
        prediction_client._client, "get", return_value=mock_performance_metric
    )
    result = prediction_client.get_metric(metric="Performance", timeframe="7d")

    mock_get.assert_called_once_with(
        "/prweb/api/PredictionStudio/v1/predictions/CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS/metric/Performance",
        time_frame="7d",
    )

    assert result.shape == (8, 3)
