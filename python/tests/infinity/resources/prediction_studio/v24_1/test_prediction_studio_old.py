import pytest
from pdstools.infinity.resources.prediction_studio.v24_1.prediction_studio import PredictionStudio
from pdstools.infinity.internal._pagination import PaginatedList
from unittest.mock import MagicMock, patch

@pytest.fixture
def prediction_studio_client():
    client = MagicMock()
    return PredictionStudio(client=client)


mock_response_repository = {
        "repository_name": "TestRepo"
    }


mock_response_predictions = {
    "predictions": [
        {
            "predictionId": "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS",
            "label": "Predict Cards Acceptance",
            "objective": "Accept",
            "subject": "Customer",
            "status": "Completed",
            "lastUpdateTime": "20240718T120557.925 GMT"
        },
        {
            "predictionId": "DATA-DECISION-REQUEST-CUSTOMER!PREDICTACTIONPROPENSITY",
            "label": "Predict Action Propensity",
            "objective": "Propensity to Accept",
            "status": "Completed",
            "lastUpdateTime": "20231129T113556.481 GMT"
        }]
        }


def test_repository(prediction_studio_client):
    mock_response = mock_response_repository
    
    with patch.object(prediction_studio_client._client, 'get', return_value=mock_response) as mock_get:
        result = prediction_studio_client.repository()
        
        mock_get.assert_called_once_with("/prweb/api/PredictionStudio/v3/predictions/repository")
        
        # Assertions to verify the Repository object's attributes
        assert result.name == "TestRepo"






def test_list_predictions(prediction_studio_client):
    mock_response = mock_response_predictions
    method_to_patch = 'get'
    with patch.object(prediction_studio_client._client, method_to_patch, return_value=mock_response):
        result = prediction_studio_client.list_predictions()
    
    assert isinstance(result, PaginatedList)
    

