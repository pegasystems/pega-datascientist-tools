import pytest
from pdstools.infinity.internal._pagination import PaginatedList
from pdstools.infinity.resources.prediction_studio.types import AdmModelType
from pdstools.infinity.resources.prediction_studio.v24_2.prediction import Prediction
from pdstools.infinity.resources.prediction_studio.v24_2.champion_challenger import ChampionChallenger
from pdstools.infinity.resources.prediction_studio.v24_2.model import Model
from pdstools.infinity.resources.prediction_studio.v24_2.model_upload import UploadedModel
from unittest.mock import MagicMock, patch
import polars as pl




@pytest.fixture
def champion_challenger_client():
    client = MagicMock()
    return ChampionChallenger(client=client, prediction_id="CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS", 
                              context="NoContext", category="Retention", model_objective="CategoryModel", champion_percentage=100,
                                active_model= Model(client=client, 
                                                    modelId="@baseclass!testModel_falcons", 
                                                    label="testModel_falcons", 
                                                    componentName="testModel_falcons", 
                                                    modelType="Adaptive model", 
                                                    modelingTechnique="Adaptive model - Bayesian", 
                                                    status="Active"))

@pytest.fixture
def champion_challenger_delete_client():
    client = MagicMock()
    return ChampionChallenger(client=client, prediction_id="CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS", 
                              context="NoContext", category="Retention", model_objective="CategoryModel", champion_percentage=100,
                                active_model= Model(client=client, 
                                                    modelId="@baseclass!testModel_falcons", 
                                                    label="testModel_falcons", 
                                                    componentName="testModel_falcons", 
                                                    modelType="Adaptive model", 
                                                    modelingTechnique="Adaptive model - Bayesian", 
                                                    status="Active"),
                                                    challenger_model= Model(client=client, 
                                                    modelId="@baseclass!testModel_falcons_copy_HBB", 
                                                    label="testModel_falcons_copy_HBB", 
                                                    componentName="testModel_falcons_copy_HBB",
                                                    modelType="Adaptive model", 
                                                    modelingTechnique="Adaptive model - Gradient Boosting", 
                                                    status="CHALLENGER")
                                                    )

@pytest.fixture
def champion_challenger_shadow_client():
    client = MagicMock()
    return ChampionChallenger(client=client, prediction_id="CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS", 
                              context="NoContext", category="Retention", model_objective="CategoryModel", champion_percentage=100,
                                active_model= Model(client=client, 
                                                    modelId="@baseclass!testModel_falcons", 
                                                    label="testModel_falcons", 
                                                    componentName="testModel_falcons", 
                                                    modelType="Adaptive model", 
                                                    modelingTechnique="Adaptive model - Bayesian", 
                                                    status="Active"),
                                                    challenger_model= Model(client=client, 
                                                    modelId="@baseclass!testModel_falcons_copy_HBB", 
                                                    label="testModel_falcons_copy_HBB", 
                                                    componentName="testModel_falcons_copy_HBB",
                                                    modelType="Adaptive model", 
                                                    modelingTechnique="Adaptive model - Gradient Boosting", 
                                                    status="SHADOW")
                                                    )


mock_champion_challenger_clone_model = [
     ChampionChallenger(client=MagicMock(), prediction_id="CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS", 
                              context="NoContext", category="Retention", model_objective="CategoryModel", champion_percentage=80,
                                active_model= Model(client=MagicMock(), 
                                                    modelId="@baseclass!testModel_falcons", 
                                                    label="testModel_falcons", 
                                                    componentName="testModel_falcons", 
                                                    modelType="Adaptive model", 
                                                    modelingTechnique="Adaptive model - Bayesian", 
                                                    status="CHAMPION"),
                                challenger_model= Model(client=MagicMock(), 
                                                    modelId="@baseclass!testModel_falcons_copy_HBB", 
                                                    label="testModel_falcons_copy_HBB", 
                                                    componentName="testModel_falcons_copy_HBB",
                                                    modelType="Adaptive model", 
                                                    modelingTechnique="Adaptive model - Gradient Boosting", 
                                                    status="CHALLENGER")
                                                    )]

mock_champion_challenger_promote_model = [
     ChampionChallenger(client=MagicMock(), prediction_id="CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS", 
                              context="NoContext", category="Retention", model_objective="CategoryModel", champion_percentage=100,
                                active_model= Model(client=MagicMock(), 
                                                    modelId="@baseclass!testModel_falcons_copy_HBB", 
                                                    label="testModel_falcons_copy_HBB", 
                                                    componentName="testModel_falcons_copy_HBB",
                                                    modelType="Adaptive model", 
                                                    modelingTechnique="Adaptive model - Gradient Boosting", 
                                                    status="ACTIVE")
                                                    )]


mock_champion_challenger_delete_model = [
     ChampionChallenger(client=MagicMock(), prediction_id="CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS", 
                              context="NoContext", category="Retention", model_objective="CategoryModel", champion_percentage=100,
                                active_model= Model(client=MagicMock(), 
                                                    modelId="@baseclass!testModel_falcons", 
                                                    label="testModel_falcons", 
                                                    componentName="testModel_falcons", 
                                                    modelType="Adaptive model", 
                                                    modelingTechnique="Adaptive model - Bayesian", 
                                                    status="CHAMPION"))]


mock_response_model = {
        "models": [
            {"modelId": "@BASECLASS!TESTMODEL_FALCONS", "label": "testModel_falcons", "modelType": "Adaptive model", "modelingTechnique": "Adaptive model - Bayesian", "source": "Pega", "status": "Completed", "lastUpdateTime": "20240718T120552.671 GMT", "updatedBy": "Somnath Paul"},
            {"modelId": "CDHSAMPLE-DATA-CUSTOMER!ADM_16330376371", "label": "Accept", "modelType": "Adaptive model", "modelingTechnique": "Adaptive model - Bayesian", "source": "Pega", "status": "Completed", "lastUpdateTime": "20240718T104417.891 GMT", "updatedBy": "Somnath Paul"}
        ]
     }



def test_clone_model(champion_challenger_client):
    mock_response_post = { "referenceID": "M-2042"}
    mock_response_get = { "ModelUpdateStatus": "Ready for review"}
    mock_response_patch = {"message": "referenceID M-2042 ,is Approved. New status Approved"}
    mock_champion_challenger = mock_champion_challenger_clone_model
    predictor_mapping = [
        {
        "predictor": "Gender",
        "property": ".Gender"
        },
        {
        "predictor": "DataUsage",
        "property": ".RiskCode"
        },
        {
        "predictor": "Age",
        "property": ".Age"
        }]
    mock_prediction = Prediction(client=MagicMock(), predictionId='CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS', label='Predict Cards Acceptance', objective='Accept', subject='Customer', status='Completed', lastUpdateTime='20240718T120557.925 GMT')
    with patch.object(champion_challenger_client._client, '_post', return_value=mock_response_post),\
        patch.object(champion_challenger_client._client, 'get', return_value=mock_response_get), \
         patch.object(champion_challenger_client._client, 'patch', return_value=mock_response_patch), \
         patch.object(champion_challenger_client._client.prediction_studio, "get_prediction" , return_value=mock_prediction),\
            patch.object(mock_prediction, 'get_champion_challengers', return_value=mock_champion_challenger ) :
            champion_challenger_client.clone_model(challenger_response_share=0.8, adm_model_type=AdmModelType.NAIVE_BAYES,predictor_mapping=predictor_mapping)
    
    
    assert champion_challenger_client.prediction_id == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
    assert champion_challenger_client.active_model
    assert champion_challenger_client.challenger_model
    assert champion_challenger_client.champion_percentage == 80


def test_add_model_model_object(champion_challenger_client):
    mock_response_post = { "referenceID": "M-2042"}
    mock_response_get = { "ModelUpdateStatus": "Ready for review"}
    mock_response_patch = {"message": "referenceID M-6002 ,is Approved. New status Approved"}
    new_model= Model(client=MagicMock(), modelId="@baseclass!testModel_falcons_copy_HBB", label="testModel_falcons_copy_HBB", 
                                                    componentName="testModel_falcons_copy_HBB",
                                                    modelType="Adaptive model", 
                                                    modelingTechnique="Adaptive model - Gradient Boosting", 
                                                    status="CHALLENGER")
    mock_champion_challenger = mock_champion_challenger_clone_model
    predictor_mapping = [
        {
        "predictor": "Gender",
        "property": ".Gender"
        },
        {
        "predictor": "DataUsage",
        "property": ".RiskCode"
        },
        {
        "predictor": "Age",
        "property": ".Age"
        }]
    mock_prediction = Prediction(client=MagicMock(), predictionId='CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS', label='Predict Cards Acceptance', objective='Accept', subject='Customer', status='Completed', lastUpdateTime='20240718T120557.925 GMT')
    with patch.object(champion_challenger_client._client, '_post', return_value=mock_response_post),\
        patch.object(champion_challenger_client._client, 'get', return_value=mock_response_get), \
         patch.object(champion_challenger_client._client, 'patch', return_value=mock_response_patch), \
         patch.object(champion_challenger_client._client.prediction_studio, "get_prediction" , return_value=mock_prediction),\
            patch.object(mock_prediction, 'get_champion_challengers', return_value=mock_champion_challenger ) :
            champion_challenger_client.add_model(challenger_response_share=0.8, new_model=new_model,predictor_mapping=predictor_mapping)

    
    assert champion_challenger_client.prediction_id == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
    assert champion_challenger_client.active_model
    assert champion_challenger_client.challenger_model
    assert champion_challenger_client.champion_percentage == 80    

def test_add_model_uploaded_model(champion_challenger_client):
    mock_response_post = { "referenceID": "M-2042"}
    mock_response_get = { "ModelUpdateStatus": "Ready for review"}
    mock_response_patch = {"message": "referenceID M-6002 ,is Approved. New status Approved"}
    new_model= UploadedModel(repository_name ="AWSFalcons", file_path= "model-staging/testModel_falcons_copy_HBB.model")
    mock_champion_challenger = mock_champion_challenger_clone_model
    predictor_mapping = [
        {
        "predictor": "Gender",
        "property": ".Gender"
        },
        {
        "predictor": "DataUsage",
        "property": ".RiskCode"
        },
        {
        "predictor": "Age",
        "property": ".Age"
        }]
    mock_prediction = Prediction(client=MagicMock(), predictionId='CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS', label='Predict Cards Acceptance', objective='Accept', subject='Customer', status='Completed', lastUpdateTime='20240718T120557.925 GMT')
    with patch.object(champion_challenger_client._client, '_post', return_value=mock_response_post),\
        patch.object(champion_challenger_client._client, 'get', return_value=mock_response_get), \
         patch.object(champion_challenger_client._client, 'patch', return_value=mock_response_patch), \
         patch.object(champion_challenger_client._client.prediction_studio, "get_prediction" , return_value=mock_prediction),\
            patch.object(mock_prediction, 'get_champion_challengers', return_value=mock_champion_challenger ) :
            champion_challenger_client.add_model(challenger_response_share=0.8, new_model=new_model,predictor_mapping=predictor_mapping)

    
    assert champion_challenger_client.prediction_id == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
    assert champion_challenger_client.active_model
    assert champion_challenger_client.challenger_model
    assert champion_challenger_client.champion_percentage == 80   

def test_add_model_str(champion_challenger_client):
    mock_response_post = { "referenceID": "M-2042"}
    mock_response_get = { "ModelUpdateStatus": "Ready for review"}
    mock_response_patch = {"message": "referenceID M-6002 ,is Approved. New status Approved"}
    new_model= "testModel_falcons_copy_HBB"
    mock_champion_challenger = mock_champion_challenger_clone_model
    predictor_mapping = [
        {
        "predictor": "Gender",
        "property": ".Gender"
        },
        {
        "predictor": "DataUsage",
        "property": ".RiskCode"
        },
        {
        "predictor": "Age",
        "property": ".Age"
        }]
    mock_prediction = Prediction(client=MagicMock(), predictionId='CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS', label='Predict Cards Acceptance', objective='Accept', subject='Customer', status='Completed', lastUpdateTime='20240718T120557.925 GMT')
    with patch.object(champion_challenger_client._client, '_post', return_value=mock_response_post),\
        patch.object(champion_challenger_client._client, 'get', return_value=mock_response_get), \
         patch.object(champion_challenger_client._client, 'patch', return_value=mock_response_patch), \
         patch.object(champion_challenger_client._client.prediction_studio, "get_prediction" , return_value=mock_prediction),\
            patch.object(mock_prediction, 'get_champion_challengers', return_value=mock_champion_challenger ) :
            champion_challenger_client.add_model(challenger_response_share=0.8, new_model=new_model,predictor_mapping=predictor_mapping)

    
    assert champion_challenger_client.prediction_id == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
    assert champion_challenger_client.active_model
    assert champion_challenger_client.challenger_model
    assert champion_challenger_client.champion_percentage == 80     
    assert repr(champion_challenger_client)
    assert str(champion_challenger_client)


def test_delete_challenger_model(champion_challenger_delete_client):
    mock_response_patch = {"message": "Model is successfully deleted"}
    mock_champion_challenger = mock_champion_challenger_delete_model
    mock_prediction = Prediction(client=MagicMock(), predictionId='CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS', label='Predict Cards Acceptance', objective='Accept', subject='Customer', status='Completed', lastUpdateTime='20240718T120557.925 GMT')
    with patch.object(champion_challenger_delete_client._client, 'patch', return_value=mock_response_patch), \
         patch.object(champion_challenger_delete_client._client.prediction_studio, "get_prediction" , return_value=mock_prediction),\
            patch.object(mock_prediction, 'get_champion_challengers', return_value=mock_champion_challenger ) :
            champion_challenger_delete_client.delete_challenger_model()

    
    assert champion_challenger_delete_client.prediction_id == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
    assert champion_challenger_delete_client.active_model
    assert champion_challenger_delete_client.champion_percentage == 100    


def test_promote_challenger_model(champion_challenger_delete_client):
    mock_response_patch = {"message": "Model is successfully deleted"}
    mock_champion_challenger = mock_champion_challenger_promote_model
    mock_prediction = Prediction(client=MagicMock(), predictionId='CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS', label='Predict Cards Acceptance', objective='Accept', subject='Customer', status='Completed', lastUpdateTime='20240718T120557.925 GMT')
    with patch.object(champion_challenger_delete_client._client, 'patch', return_value=mock_response_patch), \
         patch.object(champion_challenger_delete_client._client.prediction_studio, "get_prediction" , return_value=mock_prediction),\
            patch.object(mock_prediction, 'get_champion_challengers', return_value=mock_champion_challenger ) :
            champion_challenger_delete_client.promote_challenger_model()

    assert repr(champion_challenger_delete_client)
    assert str(champion_challenger_delete_client)
    assert champion_challenger_delete_client.prediction_id == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
    assert champion_challenger_delete_client.active_model
    assert champion_challenger_delete_client.champion_percentage == 100    

def test_update_distribution(champion_challenger_delete_client):
    mock_response_patch = {"message": "Model is successfully deleted"}
    mock_champion_challenger = mock_champion_challenger_clone_model
    mock_prediction = Prediction(client=MagicMock(), predictionId='CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS', label='Predict Cards Acceptance', objective='Accept', subject='Customer', status='Completed', lastUpdateTime='20240718T120557.925 GMT')
    with patch.object(champion_challenger_delete_client._client, 'patch', return_value=mock_response_patch), \
         patch.object(champion_challenger_delete_client._client.prediction_studio, "get_prediction" , return_value=mock_prediction),\
            patch.object(mock_prediction, 'get_champion_challengers', return_value=mock_champion_challenger ) :
            champion_challenger_delete_client.update_challenger_response_share(new_challenger_response_share=0.2)

    
    assert champion_challenger_delete_client.prediction_id == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
    assert champion_challenger_delete_client.active_model
    assert champion_challenger_delete_client.champion_percentage == 80    

def test_update_shadow_to_cc(champion_challenger_shadow_client):
    mock_response_patch = {"message": "Model is successfully deleted"}
    mock_champion_challenger = mock_champion_challenger_clone_model
    mock_prediction = Prediction(client=MagicMock(), predictionId='CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS', label='Predict Cards Acceptance', objective='Accept', subject='Customer', status='Completed', lastUpdateTime='20240718T120557.925 GMT')
    with patch.object(champion_challenger_shadow_client._client, 'patch', return_value=mock_response_patch), \
         patch.object(champion_challenger_shadow_client._client.prediction_studio, "get_prediction" , return_value=mock_prediction),\
            patch.object(mock_prediction, 'get_champion_challengers', return_value=mock_champion_challenger ) :
            champion_challenger_shadow_client.update_challenger_response_share(new_challenger_response_share=0.2)

    
    assert champion_challenger_shadow_client.prediction_id == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
    assert champion_challenger_shadow_client.active_model
    assert champion_challenger_shadow_client.champion_percentage == 80  

def test_add_predictor(champion_challenger_delete_client):
    mock_response_patch = {"message": "predictor is successfully added"}
    with patch.object(champion_challenger_delete_client._client, 'patch', return_value=mock_response_patch) :
            champion_challenger_delete_client.add_predictor(is_active_model=False, name="Income4", parameterized=True, predictor_type="symbolic", data_type="Text",value=".Age")
            champion_challenger_delete_client.add_predictor(is_active_model=True, name="Income4", parameterized=True, predictor_type="numeric", data_type="Double",value=".Age")


def test_remove_predictor(champion_challenger_delete_client):
    mock_response_patch = {"message": "predictor is successfully deleted"}
    with patch.object(champion_challenger_delete_client._client, 'patch', return_value=mock_response_patch) :
            champion_challenger_delete_client.remove_predictor(is_active_model=True, name="Income4", parameterized=True)
            champion_challenger_delete_client.remove_predictor(is_active_model=False, name="Income4", parameterized=True)
  




@pytest.mark.parametrize("return_df, mock_response, expected_type, expected_length, expected_columns", [
    (False, 
     mock_response_model, 
     PaginatedList, None, None),
    (True, 
     mock_response_model, 
     pl.DataFrame, 2, ['model_id', 'label', 'model_type', 'modeling_technique', 'source', 'status', 'last_update_time', 'updated_by']),

])

def test_list_available_models(champion_challenger_client, return_df, mock_response, expected_type, expected_length, expected_columns):
    method_to_patch = 'get' if not return_df else 'request'
    with patch.object(champion_challenger_client._client, method_to_patch, return_value=mock_response):
        result = champion_challenger_client.list_available_models_to_add(return_df=return_df)
    
    assert isinstance(result, expected_type)        
