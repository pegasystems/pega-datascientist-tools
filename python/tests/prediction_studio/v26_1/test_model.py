from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pdstools.infinity.internal._pagination import AsyncPaginatedList, PaginatedList
from pdstools.infinity.resources.prediction_studio.v26_1.model import AsyncModel, Model

mock_model = {
    "modelId": "@BASECLASS!TESTMODEL_FALCONS",
    "label": "testModel_falcons",
    "modelType": "Adaptive model",
    "modelingTechnique": "Adaptive model - Bayesian",
    "source": "Pega",
    "status": "Completed",
    "lastUpdateTime": "20240718T120552.671 GMT",
    "updatedBy": "Somnath Paul",
}

mock_response_instances = {
    "instances": [
        {
            "instanceID": "Sales-CreditCards-GoldCard",
            "name": "Gold Card",
            "type": "Adaptive model instance",
            "status": "Active",
            "active": True,
            "lastUpdateTime": "20240718T120552.671 GMT",
        }
    ]
}


@pytest.fixture
def sync_model():
    client = MagicMock()
    return Model(client=client, **mock_model)


@pytest.fixture
def async_model():
    client = MagicMock()
    client.request = AsyncMock()
    return AsyncModel(client=client, **mock_model)


def test_list_instances_sync_returns_paginated_list(sync_model, mocker):
    mocker.patch.object(
        sync_model._client,
        "request",
        return_value=mock_response_instances.copy(),
    )
    result = sync_model.list_instances(return_df=False)

    assert isinstance(result, PaginatedList)
    elements = list(result)
    assert len(elements) == 1
    instance = elements[0]
    assert instance.instance_id == "Sales-CreditCards-GoldCard"
    assert instance.name == "Gold Card"
    assert instance.type == "Adaptive model instance"
    assert instance.status == "Active"
    assert instance.active is True
    assert instance.last_update_time == datetime(2024, 7, 18, 12, 5, 52, 671000)


def test_list_instances_sync_returns_df(sync_model, mocker):
    mocker.patch.object(
        sync_model._client,
        "request",
        return_value=mock_response_instances.copy(),
    )
    result = sync_model.list_instances(return_df=True)

    assert result.shape == (1, 6)
    assert result.columns == [
        "instance_id",
        "name",
        "type",
        "status",
        "active",
        "last_update_time",
    ]
    assert result["instance_id"][0] == "Sales-CreditCards-GoldCard"


@pytest.mark.asyncio
async def test_list_instances_async_returns_paginated_list(async_model):
    async_model._client.request.return_value = mock_response_instances.copy()
    result = await async_model.list_instances(return_df=False)

    assert isinstance(result, AsyncPaginatedList)
    elements = await result.collect()
    assert len(elements) == 1
    instance = elements[0]
    assert instance.instance_id == "Sales-CreditCards-GoldCard"
    assert instance.name == "Gold Card"
    assert instance.type == "Adaptive model instance"
    assert instance.status == "Active"
    assert instance.active is True
    assert instance.last_update_time == datetime(2024, 7, 18, 12, 5, 52, 671000)


@pytest.mark.asyncio
async def test_list_instances_async_returns_df(async_model):
    async_model._client.request.return_value = mock_response_instances.copy()
    result = await async_model.list_instances(return_df=True)

    assert result.shape == (1, 6)
    assert result.columns == [
        "instance_id",
        "name",
        "type",
        "status",
        "active",
        "last_update_time",
    ]
    assert result["instance_id"][0] == "Sales-CreditCards-GoldCard"
