"""Exact-value tests for the Prediction Studio ModelInstance Pydantic data layer.

These cover the ``ModelInstanceData`` payload model directly
(alias handling, forward-compat field retention, Pega datetime parsing, and
the locked ``model_dump`` schema) plus the concrete ``ModelInstance`` and
``AsyncModelInstance`` resources.
"""

from __future__ import annotations

import datetime

import pytest
import polars as pl

from pdstools.infinity.resources.prediction_studio.schemas import (
    ModelInstanceData,
    ResourceData,
    parse_pega_datetime,
)
from pdstools.infinity.resources.prediction_studio.base import (
    ModelInstance,
    AsyncModelInstance,
)

_PAYLOAD = {
    "instanceID": "Sales-CreditCards-GoldCard",
    "name": "Gold Card",
    "type": "Adaptive model instance",
    "status": "Active",
    "active": True,
    "lastUpdateTime": "20240718T120552.671 GMT",
}

_EXPECTED_DATETIME = datetime.datetime(2024, 7, 18, 12, 5, 52, 671000)

_BASE_FIELD_ORDER = [
    "instance_id",
    "name",
    "type",
    "status",
    "active",
    "last_update_time",
]


# --------------------------------------------------------------------------- #
# parse_pega_datetime
# --------------------------------------------------------------------------- #
def test_parse_pega_datetime_parses_pega_string():
    assert parse_pega_datetime("20240718T120552.671 GMT") == _EXPECTED_DATETIME


def test_parse_pega_datetime_passthrough_datetime():
    assert parse_pega_datetime(_EXPECTED_DATETIME) is _EXPECTED_DATETIME


def test_parse_pega_datetime_none_and_empty():
    assert parse_pega_datetime(None) is None
    assert parse_pega_datetime("") is None


def test_parse_pega_datetime_bad_string_raises():
    with pytest.raises(ValueError):
        parse_pega_datetime("not-a-date")


def test_parse_pega_datetime_bad_type_raises():
    with pytest.raises(ValueError):
        parse_pega_datetime(12345)


# --------------------------------------------------------------------------- #
# ModelInstanceData
# --------------------------------------------------------------------------- #
def test_model_instance_data_accepts_camelcase_aliases():
    data = ModelInstanceData.model_validate(_PAYLOAD)
    assert data.instance_id == "Sales-CreditCards-GoldCard"
    assert data.name == "Gold Card"
    assert data.type == "Adaptive model instance"
    assert data.status == "Active"
    assert data.active is True
    assert data.last_update_time == _EXPECTED_DATETIME


def test_model_instance_data_accepts_snake_case_names():
    data = ModelInstanceData.model_validate({"instance_id": "A!B", "name": "x", "active": False})
    assert data.instance_id == "A!B"
    assert data.name == "x"
    assert data.active is False


def test_model_instance_data_parses_pega_datetime():
    data = ModelInstanceData.model_validate(_PAYLOAD)
    assert data.last_update_time == _EXPECTED_DATETIME


def test_model_instance_data_missing_optionals_default_to_none():
    data = ModelInstanceData.model_validate({"instanceID": "A!B"})
    assert data.name is None
    assert data.type is None
    assert data.status is None
    assert data.active is None
    assert data.last_update_time is None


def test_model_instance_data_active_bool_handling():
    data_true = ModelInstanceData.model_validate({"instanceID": "A", "active": True})
    assert data_true.active is True
    data_false = ModelInstanceData.model_validate({"instanceID": "A", "active": False})
    assert data_false.active is False


def test_model_instance_data_dump_locks_schema_order():
    data = ModelInstanceData.model_validate({"instanceID": "A!B"})
    # Even with everything optional missing, all base fields are present.
    assert list(data.model_dump().keys()) == _BASE_FIELD_ORDER


def test_model_instance_data_retains_forward_compat_fields():
    data = ModelInstanceData.model_validate({**_PAYLOAD, "futureField": "surprise"})
    dump = data.model_dump()
    assert dump["futureField"] == "surprise"
    # Unknown extras land after the declared fields.
    assert list(dump.keys()) == [*_BASE_FIELD_ORDER, "futureField"]


def test_model_instance_data_is_resource_data_subclass():
    assert issubclass(ModelInstanceData, ResourceData)


# --------------------------------------------------------------------------- #
# Resource wiring: attribute delegation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("instance_cls", [ModelInstance, AsyncModelInstance])
def test_model_instance_uses_model_instance_data(instance_cls, mocker):
    instance = instance_cls(client=mocker.MagicMock(), **_PAYLOAD)
    assert type(instance._data) is ModelInstanceData
    assert list(instance._public_dict.keys()) == _BASE_FIELD_ORDER


def test_model_instance_attribute_access_delegates_to_data(mocker):
    instance = ModelInstance(client=mocker.MagicMock(), **_PAYLOAD)
    assert instance.instance_id == "Sales-CreditCards-GoldCard"
    assert instance.last_update_time == _EXPECTED_DATETIME
    assert instance.name == "Gold Card"


def test_model_instance_public_dict_is_snake_case_with_values(mocker):
    instance = ModelInstance(client=mocker.MagicMock(), **_PAYLOAD)
    public = instance._public_dict
    assert public["instance_id"] == "Sales-CreditCards-GoldCard"
    assert public["last_update_time"] == _EXPECTED_DATETIME
    assert public["active"] is True


def test_model_instance_unknown_attribute_raises(mocker):
    instance = ModelInstance(client=mocker.MagicMock(), **_PAYLOAD)
    with pytest.raises(AttributeError):
        _ = instance.definitely_not_a_field


# --------------------------------------------------------------------------- #
# Locked DataFrame schema (empty / all-null / forward-compat extras)
# --------------------------------------------------------------------------- #
def test_polars_schema_excludes_extras_and_types_fields():
    schema = ModelInstanceData.polars_schema()
    assert list(schema.keys()) == _BASE_FIELD_ORDER
    assert schema["instance_id"] == pl.String
    assert schema["last_update_time"] == pl.Datetime
    assert schema["active"] == pl.Boolean
