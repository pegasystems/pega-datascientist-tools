"""Exact-value tests for the Prediction Studio ``NotificationData`` layer.

These cover the ``NotificationData`` payload model directly (alias handling,
forward-compat field retention, Pega datetime parsing, and the locked
``model_dump`` schema) plus the ``_data`` wiring on the sync/async
``Notification`` resources and the locked ``return_df=True`` DataFrame schema.
"""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from pdstools.infinity.internal._pagination import PaginatedList
from pdstools.infinity.resources.prediction_studio.base import (
    AsyncNotification,
    Notification,
)
from pdstools.infinity.resources.prediction_studio.schemas import (
    NotificationData,
    ResourceData,
)

# A realistic camelCase payload as the Pega notifications endpoint returns it.
_PAYLOAD = {
    "modelID": "CDHSAMPLE-DATA-CUSTOMER!ADM_16330376371",
    "predictionID": "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS",
    "context": "SalesContext",
    "notificationType": "Performance",
    "notificationID": "N-3",
    "notificationMnemonic": "Model performance is at its minimum",
    "description": "Performance dropped below threshold",
    "modelType": "Adaptive model",
    "impact": "High",
    "triggerTime": "20240718T120552.671 GMT",
}

# Minimal payload: only the required fields (model/prediction scoping absent).
_MINIMAL_PAYLOAD = {
    "notificationType": "Performance",
    "notificationID": "N-3",
    "notificationMnemonic": "Min performance",
    "description": "desc",
    "modelType": "Adaptive model",
    "impact": "High",
    "triggerTime": "20240718T120552.671 GMT",
}

_EXPECTED_DATETIME = datetime.datetime(2024, 7, 18, 12, 5, 52, 671000)

_FIELD_ORDER = [
    "model_id",
    "prediction_id",
    "context",
    "notification_type",
    "notification_id",
    "notification_mnemonic",
    "description",
    "model_type",
    "impact",
    "trigger_time",
]


# --------------------------------------------------------------------------- #
# NotificationData
# --------------------------------------------------------------------------- #
def test_notification_data_accepts_camelcase_aliases():
    data = NotificationData.model_validate(_PAYLOAD)
    assert data.model_id == "CDHSAMPLE-DATA-CUSTOMER!ADM_16330376371"
    assert data.prediction_id == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
    assert data.context == "SalesContext"
    assert data.notification_type == "Performance"
    assert data.notification_id == "N-3"
    assert data.notification_mnemonic == "Model performance is at its minimum"
    assert data.description == "Performance dropped below threshold"
    assert data.model_type == "Adaptive model"
    assert data.impact == "High"
    assert data.trigger_time == _EXPECTED_DATETIME


def test_notification_data_accepts_snake_case_names():
    data = NotificationData.model_validate(
        {
            "notification_type": "Performance",
            "notification_id": "N-3",
            "notification_mnemonic": "m",
            "description": "d",
            "model_type": "Adaptive model",
            "impact": "High",
            "trigger_time": "20240718T120552.671 GMT",
        },
    )
    assert data.notification_id == "N-3"
    assert data.trigger_time == _EXPECTED_DATETIME


def test_notification_data_parses_pega_datetime():
    data = NotificationData.model_validate(_PAYLOAD)
    assert data.trigger_time == _EXPECTED_DATETIME


def test_notification_data_missing_scope_defaults_to_none():
    data = NotificationData.model_validate(_MINIMAL_PAYLOAD)
    assert data.model_id is None
    assert data.prediction_id is None
    assert data.context is None


def test_notification_data_dump_locks_schema_order():
    data = NotificationData.model_validate(_MINIMAL_PAYLOAD)
    # Even with model/prediction scoping absent, all fields are present.
    assert list(data.model_dump().keys()) == _FIELD_ORDER


def test_notification_data_retains_forward_compat_fields():
    data = NotificationData.model_validate({**_PAYLOAD, "futureField": "surprise"})
    dump = data.model_dump()
    assert dump["futureField"] == "surprise"
    assert list(dump.keys()) == [*_FIELD_ORDER, "futureField"]


def test_notification_data_is_resource_data_subclass():
    assert issubclass(NotificationData, ResourceData)


# --------------------------------------------------------------------------- #
# Resource wiring: _data + attribute delegation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("notification_cls", [Notification, AsyncNotification])
def test_notification_uses_notification_data(notification_cls, mocker):
    notification = notification_cls(client=mocker.MagicMock(), **_PAYLOAD)
    assert type(notification._data) is NotificationData
    assert list(notification._public_dict.keys()) == _FIELD_ORDER


def test_notification_attribute_access_delegates_to_data(mocker):
    notification = Notification(client=mocker.MagicMock(), **_PAYLOAD)
    assert notification.notification_id == "N-3"
    assert notification.model_id == "CDHSAMPLE-DATA-CUSTOMER!ADM_16330376371"
    assert notification.trigger_time == _EXPECTED_DATETIME


def test_notification_public_dict_has_snake_case_values(mocker):
    notification = Notification(client=mocker.MagicMock(), **_PAYLOAD)
    public = notification._public_dict
    assert public["prediction_id"] == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
    assert public["trigger_time"] == _EXPECTED_DATETIME
    assert public["impact"] == "High"


def test_notification_minimal_payload_fills_none_columns(mocker):
    notification = Notification(client=mocker.MagicMock(), **_MINIMAL_PAYLOAD)
    public = notification._public_dict
    assert public["model_id"] is None
    assert public["prediction_id"] is None
    assert public["context"] is None


def test_notification_unknown_attribute_raises(mocker):
    notification = Notification(client=mocker.MagicMock(), **_PAYLOAD)
    with pytest.raises(AttributeError):
        _ = notification.definitely_not_a_field


# --------------------------------------------------------------------------- #
# Locked DataFrame schema (empty / all-null scope / forward-compat extras)
# --------------------------------------------------------------------------- #
def _notifications_df(notifications, mocker):
    client = mocker.MagicMock()
    client.request.return_value = {"notifications": notifications}
    pages: PaginatedList = PaginatedList(
        Notification,
        client,
        "get",
        "endpoint",
        _root="notifications",
    )
    return pages.as_df()


def test_notifications_empty_keeps_locked_schema(mocker):
    df = _notifications_df([], mocker)
    assert df.shape == (0, 10)
    assert df.columns == _FIELD_ORDER


def test_notifications_minimal_scope_have_typed_columns(mocker):
    df = _notifications_df([_MINIMAL_PAYLOAD], mocker)
    assert df.columns == _FIELD_ORDER
    assert df.schema["model_id"] == pl.String
    assert df.schema["prediction_id"] == pl.String
    assert df.schema["trigger_time"] == pl.Datetime
    assert df.select(pl.col("trigger_time").dt.year()).to_dicts() == [{"trigger_time": 2024}]


def test_notifications_extras_do_not_leak_into_dataframe(mocker):
    df = _notifications_df(
        [
            {**_PAYLOAD, "futureField": "surprise"},
            {**_MINIMAL_PAYLOAD, "anotherFuture": 7},
        ],
        mocker,
    )
    assert df.columns == _FIELD_ORDER


def test_notification_polars_schema_types_fields():
    schema = NotificationData.polars_schema()
    assert list(schema.keys()) == _FIELD_ORDER
    assert schema["model_id"] == pl.String
    assert schema["notification_id"] == pl.String
    assert schema["trigger_time"] == pl.Datetime
