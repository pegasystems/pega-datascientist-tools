from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from pdstools.infinity.internal._pagination import PaginatedList
from pdstools.infinity.resources.prediction_studio.v24_2.prediction import Prediction

mock_prediction = {
            "predictionId": "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS",
            "label": "Predict Cards Acceptance",
            "objective": "Accept",
            "subject": "Customer",
            "status": "Completed",
            "lastUpdateTime": "20240718T120557.925 GMT"
        }

@pytest.fixture
def prediction_client():
    client = MagicMock()
    return Prediction(client=client,**mock_prediction)


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
    "targetLabels": [
        {
            "label": "Accept"
        }
    ],
    "alternateLabels": [
        {
            "label": "Reject"
        }
    ],
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
                    "modelingTechnique": "Adaptive model - Bayesian"
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
                    "modelingTechnique": "Adaptive model - Bayesian"
                }
            ]
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
            "modelingTechnique": "GBM and XGBoost"
        }
    ],
    "metrics": {
        "lift": 0.0,
        "performance": 68.87,
        "performanceMeasure": "AUC"
    }
}

mock_performance_metric = {
    "monitoringData": [
        {
            "value": "59.55",
            "snapshotTime": "2024-07-04T12:00:00.000Z"
        },
        {
            "value": "58.22",
            "snapshotTime": "2024-07-05T12:00:00.000Z"
        },
        {
            "value": "74.29",
            "snapshotTime": "2024-07-06T12:00:00.000Z"
        },
        {
            "value": "74.22",
            "snapshotTime": "2024-07-07T12:00:00.000Z"
        },
        {
            "value": "67.66",
            "snapshotTime": "2024-07-08T12:00:00.000Z"
        },
        {
            "value": "70.13",
            "snapshotTime": "2024-07-09T12:00:00.000Z"
        },
        {
            "value": "77.45",
            "snapshotTime": "2024-07-10T12:00:00.000Z"
        },
        {
            "value": "68.87",
            "snapshotTime": "2024-07-11T12:00:00.000Z"
        }
    ]
}



mock_response_notifications = {
    "notifications": [
        {
            "description": "Model performance is at its minimum of 50 AUC for model context Sales-DepositAccounts-RegularSaving",
            "modelID": "DATA-DECISION-REQUEST-CUSTOMER!OMNIADAPTIVEMODEL",
            "modelType": "Adaptive model",
            "notificationType": "Performance",
            "notificationID": "N-3",
            "notificationMnemonic": "Model performance is at its minimum",
            "context": "Sales-DepositAccounts-BasicChecking",
            "impact": "High",
            "triggerTime": "20240720T050556.826 GMT"
        },
        {
            "description": "Model performance is at its minimum of 50 AUC for model context Sales-Bundles-UMortgageBundle",
            "modelID": "DATA-DECISION-REQUEST-CUSTOMER!OMNIADAPTIVEMODEL",
            "modelType": "Adaptive model",
            "notificationType": "Performance",
            "notificationID": "N-3",
            "notificationMnemonic": "Model performance is at its minimum",
            "context": "Sales-DepositAccounts-BasicChecking",
            "impact": "High",
            "triggerTime": "20240720T050554.923 GMT"
        }]
}

mock_response_get_models = [{'id': 'CDHSample-Data-Customer!Adm_16330376371',
  'label': 'Accept',
  'role': 'ACTIVE',
  'type': 'Adaptive model',
  'performance': 72.51,
  'performanceMeasure': 'AUC',
  'componentName': 'Accept',
  'modelingTechnique': 'Adaptive model - Bayesian',
  'model_type': 'Primary Model',
  'contextName': 'NoContext',
  'prediction_id': 'CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS'},
 {'id': '@baseclass!testModel_falcons',
  'label': 'testModel_falcons',
  'role': 'ACTIVE',
  'type': 'Adaptive model',
  'performance': 0.0,
  'categoryName': 'Retention',
  'componentName': 'testModel_falcons',
  'modelingTechnique': 'Adaptive model - Bayesian',
  'model_type': 'CategoryModel',
  'contextName': 'NoContext',
  'prediction_id': 'CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS'},
 {'id': 'CDHSample-Data-Customer!RiskModel',
  'label': 'Credit Risk Model',
  'role': 'ACTIVE',
  'type': 'Predictive model',
  'performance': 50.61,
  'performanceMeasure': 'AUC',
  'componentName': 'RiskModel',
  'contextName': 'NoContext',
  'modelingTechnique': 'GBM and XGBoost',
  'model_type': 'Supporting model',
  'prediction_id': 'CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS'}]


def test_prediction_describe(prediction_client):
    mock_response = mock_prediction_describe
    
    with patch.object(prediction_client._client, 'get', return_value=mock_response) as mock_get:
        result = prediction_client.describe()
        
        mock_get.assert_called_once_with("/prweb/api/PredictionStudio/v3/predictions/CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS")
        
    
        assert result["predictionId"] == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
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
        assert result["metrics"] == {"lift": 0.0, "performance": 68.87, "performanceMeasure": "AUC"}

@pytest.mark.parametrize("return_df, expected_type, additional_checks", [
    (False, PaginatedList, None),
    (True, pl.DataFrame, lambda df: (
        len(df) == 2,
        df.columns == ['model_id','context', 'notification_type', 'notification_id','notification_mnemonic', 'description', 'model_type', 'impact', 'trigger_time'],
        df.shape == (2, 9)
    ))
])
def test_prediction_notifications(prediction_client, return_df, expected_type, additional_checks):
    mock_response = mock_response_notifications
    method_to_patch = 'get' if not return_df else 'request'
    with patch.object(prediction_client._client, method_to_patch, return_value=mock_response):
        result = prediction_client.get_notifications(return_df=return_df)
    
    assert isinstance(result, expected_type)
    if additional_checks:
        for check in additional_checks(result):
            assert check   


def test_prediction_get_models(prediction_client):
    mock_response = mock_prediction_describe
    
    with patch.object(prediction_client._client, 'get', return_value=mock_response) as mock_get:
        result = prediction_client._get_models()
        
        mock_get.assert_called_once_with("/prweb/api/PredictionStudio/v3/predictions/CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS")
        assert result == mock_response_get_models


def test_prediction_get_cc(prediction_client):
    mock_response = mock_prediction_describe
    
    with patch.object(prediction_client._client, 'get', return_value=mock_response) as mock_get:
        result = prediction_client.get_champion_challengers()
        
        mock_get.assert_called_once_with("/prweb/api/PredictionStudio/v3/predictions/CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS")
        
        assert result[0].prediction_id == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
        assert len(result) == 3


def test_add_conditional_model(prediction_client):
    mock_response_post = {'referenceID': 'M-6011'}
    mock_response = mock_prediction_describe
    
    with patch.object(prediction_client._client, '_post', return_value=mock_response_post),\
        patch.object(prediction_client._client, 'get', return_value=mock_response):
            result = prediction_client.add_conditional_model(new_model = "@baseclass!testModel_falcons", category="Retention")

    
    # mock_add_conditional_model.assert_called_once_with("/prweb/api/PredictionStudio/v4/predictions/CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS/category/Retention/models/@baseclass!testModel_falcons")
    assert result.prediction_id == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"


def test_prediction_get_metric(prediction_client):
    mock_response = mock_performance_metric
    start_date = date(2024,7,2)
    end_date = date(2024,7,11)
    with patch.object(prediction_client._client, 'get', return_value=mock_response) as mock_get:
        result = prediction_client.get_metric(start_date=start_date, end_date=end_date,
                        metric="Performance",
                        frequency="Daily")
        
        mock_get.assert_called_once_with('/prweb/api/PredictionStudio/v2/predictions/CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS/metric/Performance', startDate='02/07/2024', endDate='11/07/2024', frequency='Daily')
        
    
        assert result.shape == (8, 3)  

def test_get_staged_changes(prediction_client):
    mock_responses = {
        "listOfChanges": [
            {
                "change": "Added conditional model for the category Retention",
                "type": "Model",
                "ruleName": "testModel_falcons",
                "caseID": "M-2042",
                "updateDateTime": "20240718T120558.474 GMT"
            }
        ]
    } 
    with patch.object(prediction_client._client, 'get', return_value=mock_responses):
        result = prediction_client.get_staged_changes()

    assert len(result) == 1
    assert result[0]['change'] == "Added conditional model for the category Retention"
    assert result[0]['type'] == "Model"
    assert result[0]['ruleName'] == "testModel_falcons"
    assert result[0]['caseID'] == "M-2042"
    assert result[0]['updateDateTime'] == "20240718T120558.474 GMT"


def test_package_staged_changes(prediction_client):
    mock_responses = {'message': 'Changes are packaged into the branch M-3001 and are submitted for creation of a revision, refer to the revision manager to confirm if the changes are included in the Revision via 1:1 Operations Manager Change request.',
                        'referenceID': 'M-3001'}
    with patch.object(prediction_client._client, 'post', return_value=mock_responses):
        result = prediction_client.package_staged_changes()

    assert result['referenceID'] == 'M-3001'
       




