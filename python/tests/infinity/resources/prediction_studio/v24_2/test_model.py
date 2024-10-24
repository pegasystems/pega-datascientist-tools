import pytest
from pdstools.infinity.internal._pagination import PaginatedList
from pdstools.infinity.resources.prediction_studio.v24_2.model import Model
from unittest.mock import MagicMock, patch
import polars as pl

mock_model = {"modelId": "@BASECLASS!TESTMODEL_FALCONS", 
                       "label": "testModel_falcons", 
                       "modelType": "Adaptive model", 
                       "modelingTechnique": "Adaptive model - Bayesian", 
                       "source": "Pega", 
                       "status": "Completed", 
                       "lastUpdateTime": "20240718T120552.671 GMT", 
                       "updatedBy": "Somnath Paul"}

@pytest.fixture
def model_client():
    client = MagicMock()
    return Model(client=client,**mock_model)


mock_model_describe = {
    "modelId": "@BASECLASS!TESTMODEL_FALCONS",
    "status": "Completed",
    "label": "testModel_falcons",
    "type": "Adaptive Model",
    "subject": "@baseclass",
    "outcomeType": "Scoring",
    "modelingTechnique": "Adaptive model",
    "modelCategory": "Retention",
    "source": "Pega",
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
    "predictors": [
        {
            "name": ".Salary",
            "dataType": "symbolic"
        }
    ],
    "parameterizedPredictors": [],
    "IHSummaryPredictors": [
        {
            "name": "IH.{Channel}.{Direction}.{Outcome}.pxLastGroupID",
            "aggregation": "last"
        },
        {
            "name": "IH.{Channel}.{Direction}.{Outcome}.pxLastOutcomeTime.DaysSince",
            "aggregation": "last"
        },
        {
            "name": "IH.{Channel}.{Direction}.{Outcome}.pyHistoricalOutcomeCount",
            "aggregation": "count"
        }
    ],
    "metrics": {
        "performance": 0.0,
        "performanceMeasure": "AUC"
    }
}



mock_response_notifications = {
    "notifications": [
        {
            "description": "Model performance is at its minimum of 50 AUC",
            "modelType": "Adaptive Model",
            "notificationType": "Performance",
            "notificationID": "N-3",
            "notificationMnemonic": "Model performance is at its minimum",
            "context": None,
            "impact": "High",
            "triggerTime": "20240720T050406.219 GMT"
        },
        {
            "description": "Model performance is at its minimum of 50 AUC",
            "modelType": "Adaptive Model",
            "notificationType": "Performance",
            "notificationID": "N-3",
            "notificationMnemonic": "Model performance is at its minimum",
            "context": None,
            "impact": "High",
            "triggerTime": "20240720T050406.219 GMT"
        }]
}




def test_model_describe(model_client):
    mock_response = mock_model_describe
    
    with patch.object(model_client._client, 'get', return_value=mock_response) as mock_get:
        result = model_client.describe()
        
        mock_get.assert_called_once_with("/prweb/api/PredictionStudio/v1/models/@BASECLASS!TESTMODEL_FALCONS")
        
        # Assertions to verify the Repository object's attributes
        assert result["modelId"] == "@BASECLASS!TESTMODEL_FALCONS"
        assert result["status"] == "Completed"
        assert result["label"] == "testModel_falcons"
        assert result["type"] == "Adaptive Model"
        assert result["subject"] == "@baseclass"
        assert result["outcomeType"] == "Scoring"
        assert result["modelingTechnique"] == "Adaptive model"
        assert result["modelCategory"] == "Retention"
        assert result["source"] == "Pega"
        assert result["targetLabels"] == [{"label": "Accept"}]
        assert result["alternateLabels"] == [{"label": "Reject"}]
        assert result["predictors"] == [{"name": ".Salary", "dataType": "symbolic"}]
        assert result["parameterizedPredictors"] == []
        assert result["IHSummaryPredictors"] == [{"name": "IH.{Channel}.{Direction}.{Outcome}.pxLastGroupID", "aggregation": "last"},
                                                {"name": "IH.{Channel}.{Direction}.{Outcome}.pxLastOutcomeTime.DaysSince", "aggregation": "last"},
                                                {"name": "IH.{Channel}.{Direction}.{Outcome}.pyHistoricalOutcomeCount", "aggregation": "count"}]
        assert result["metrics"] == {"performance": 0.0, "performanceMeasure": "AUC"}

@pytest.mark.parametrize("return_df, expected_type, additional_checks", [
    (False, PaginatedList, None),
    (True, pl.DataFrame, lambda df: (
        len(df) == 2,
        df.columns == ['notification_type', 'notification_id','notification_mnemonic', 'description', 'model_type', 'impact', 'trigger_time'],
        df.shape == (2, 7)
    ))
])
def test_get_notifications(model_client, return_df, expected_type, additional_checks):
    mock_response = mock_response_notifications
    method_to_patch = 'get' if not return_df else 'request'
    with patch.object(model_client._client, method_to_patch, return_value=mock_response):
        result = model_client.get_notifications(return_df=return_df)
    
    assert isinstance(result, expected_type)
    
    if additional_checks:
        for check in additional_checks(result):
            assert check    

