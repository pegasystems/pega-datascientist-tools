"""Exact-value tests for the Prediction Studio Pydantic data layer.

These cover the ``ModelData`` / ``ModelDataV26_1`` payload models directly
(alias handling, forward-compat field retention, Pega datetime parsing, and
the locked ``model_dump`` schema) plus the ``_data_cls`` selection wired into
the sync/async ``Model`` resources for both API versions.
"""

from __future__ import annotations

import datetime

import pytest

from pdstools.infinity.resources.prediction_studio.schemas import (
    ModelData,
    ModelDataV26_1,
    ResourceData,
    parse_pega_datetime,
)
from pdstools.infinity.resources.prediction_studio.v24_2.model import (
    AsyncModel as AsyncModelV24_2,
)
from pdstools.infinity.resources.prediction_studio.v24_2.model import (
    Model as ModelV24_2,
)
from pdstools.infinity.resources.prediction_studio.v26_1.model import (
    AsyncModel as AsyncModelV26_1,
)
from pdstools.infinity.resources.prediction_studio.v26_1.model import (
    Model as ModelV26_1,
)

# A realistic camelCase payload as the Pega models endpoint returns it.
_PAYLOAD = {
    "modelId": "CDHSAMPLE-DATA-CUSTOMER!ADM_16330376371",
    "label": "testModel_falcons",
    "modelType": "Adaptive model",
    "modelingTechnique": "Adaptive model - Bayesian",
    "source": "Pega",
    "status": "Completed",
    "componentName": "MyComponent",
    "lastUpdateTime": "20240718T120552.671 GMT",
    "updatedBy": "operator@pega.com",
}

_V26_PAYLOAD = {**_PAYLOAD, "performance": 55.5, "performanceMeasure": "AUC"}

_EXPECTED_DATETIME = datetime.datetime(2024, 7, 18, 12, 5, 52, 671000)

_BASE_FIELD_ORDER = [
    "model_id",
    "label",
    "model_type",
    "modeling_technique",
    "source",
    "status",
    "component_name",
    "last_update_time",
    "updated_by",
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
# ModelData
# --------------------------------------------------------------------------- #
def test_model_data_accepts_camelcase_aliases():
    data = ModelData.model_validate(_PAYLOAD)
    assert data.model_id == "CDHSAMPLE-DATA-CUSTOMER!ADM_16330376371"
    assert data.label == "testModel_falcons"
    assert data.model_type == "Adaptive model"
    assert data.modeling_technique == "Adaptive model - Bayesian"
    assert data.source == "Pega"
    assert data.status == "Completed"
    assert data.component_name == "MyComponent"
    assert data.last_update_time == _EXPECTED_DATETIME
    assert data.updated_by == "operator@pega.com"


def test_model_data_accepts_snake_case_names():
    data = ModelData.model_validate({"model_id": "A!B", "label": "x", "model_type": "Adaptive model"})
    assert data.model_id == "A!B"
    assert data.model_type == "Adaptive model"


def test_model_data_parses_pega_datetime():
    data = ModelData.model_validate(_PAYLOAD)
    assert data.last_update_time == _EXPECTED_DATETIME


def test_model_data_missing_optionals_default_to_none():
    data = ModelData.model_validate({"modelId": "A!B", "label": "x"})
    assert data.model_type is None
    assert data.component_name is None
    assert data.last_update_time is None
    assert data.updated_by is None


def test_model_data_dump_locks_schema_order():
    data = ModelData.model_validate({"modelId": "A!B", "label": "x"})
    # Even with everything optional missing, all base fields are present.
    assert list(data.model_dump().keys()) == _BASE_FIELD_ORDER


def test_model_data_retains_forward_compat_fields():
    data = ModelData.model_validate({**_PAYLOAD, "futureField": "surprise"})
    dump = data.model_dump()
    assert dump["futureField"] == "surprise"
    # Unknown extras land after the declared fields.
    assert list(dump.keys()) == [*_BASE_FIELD_ORDER, "futureField"]


def test_model_data_is_resource_data_subclass():
    assert issubclass(ModelData, ResourceData)


# --------------------------------------------------------------------------- #
# ModelDataV26_1
# --------------------------------------------------------------------------- #
def test_model_data_v26_adds_performance_fields():
    data = ModelDataV26_1.model_validate(_V26_PAYLOAD)
    assert data.performance == 55.5
    assert data.performance_measure == "AUC"


def test_model_data_v26_dump_appends_performance_after_base():
    data = ModelDataV26_1.model_validate(_V26_PAYLOAD)
    assert list(data.model_dump().keys()) == [
        *_BASE_FIELD_ORDER,
        "performance",
        "performance_measure",
    ]


def test_model_data_v26_inherits_datetime_validator():
    data = ModelDataV26_1.model_validate(_V26_PAYLOAD)
    assert data.last_update_time == _EXPECTED_DATETIME


def test_model_data_v26_performance_defaults_to_none():
    data = ModelDataV26_1.model_validate({"modelId": "A!B", "label": "x"})
    assert data.performance is None
    assert data.performance_measure is None


# --------------------------------------------------------------------------- #
# Resource wiring: _data_cls selection + attribute delegation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_cls", [ModelV24_2, AsyncModelV24_2])
def test_v24_2_model_uses_base_model_data(model_cls, mocker):
    model = model_cls(client=mocker.MagicMock(), **_PAYLOAD)
    assert type(model._data) is ModelData
    assert list(model._public_dict.keys()) == _BASE_FIELD_ORDER


@pytest.mark.parametrize("model_cls", [ModelV26_1, AsyncModelV26_1])
def test_v26_1_model_uses_v26_model_data(model_cls, mocker):
    model = model_cls(client=mocker.MagicMock(), **_V26_PAYLOAD)
    assert type(model._data) is ModelDataV26_1
    assert list(model._public_dict.keys()) == [
        *_BASE_FIELD_ORDER,
        "performance",
        "performance_measure",
    ]


def test_model_attribute_access_delegates_to_data(mocker):
    model = ModelV24_2(client=mocker.MagicMock(), **_PAYLOAD)
    assert model.model_id == "CDHSAMPLE-DATA-CUSTOMER!ADM_16330376371"
    assert model.last_update_time == _EXPECTED_DATETIME
    assert model.component_name == "MyComponent"


def test_model_public_dict_is_snake_case_with_values(mocker):
    model = ModelV26_1(client=mocker.MagicMock(), **_V26_PAYLOAD)
    public = model._public_dict
    assert public["model_id"] == "CDHSAMPLE-DATA-CUSTOMER!ADM_16330376371"
    assert public["last_update_time"] == _EXPECTED_DATETIME
    assert public["performance"] == 55.5
    assert public["performance_measure"] == "AUC"


def test_model_unknown_attribute_raises(mocker):
    model = ModelV24_2(client=mocker.MagicMock(), **_PAYLOAD)
    with pytest.raises(AttributeError):
        _ = model.definitely_not_a_field


# --------------------------------------------------------------------------- #
# Locked DataFrame schema (empty / all-null / forward-compat extras)
# --------------------------------------------------------------------------- #
def _list_models_df(ps_cls, models, mocker):
    client = mocker.MagicMock()
    client.request.return_value = {"models": models}
    inst = ps_cls.__new__(ps_cls)
    inst._client = client
    return inst.list_models(return_df=True)


def test_list_models_empty_keeps_locked_schema(mocker):
    from pdstools.infinity.resources.prediction_studio.v24_2.prediction_studio._sync import (
        PredictionStudio,
    )

    df = _list_models_df(PredictionStudio, [], mocker)
    assert df.shape == (0, 9)
    assert df.columns == _BASE_FIELD_ORDER


def test_list_models_all_null_optionals_have_typed_columns(mocker):
    import polars as pl

    from pdstools.infinity.resources.prediction_studio.v24_2.prediction_studio._sync import (
        PredictionStudio,
    )

    df = _list_models_df(PredictionStudio, [{"modelId": "A!B", "label": "x"}], mocker)
    # Optional columns are correctly typed (not Null), so dtype-specific ops work.
    assert df.schema["status"] == pl.String
    assert df.schema["last_update_time"] == pl.Datetime
    assert df.select(pl.col("last_update_time").dt.year()).to_dicts() == [{"last_update_time": None}]


def test_list_models_extras_do_not_leak_into_dataframe(mocker):
    from pdstools.infinity.resources.prediction_studio.v24_2.prediction_studio._sync import (
        PredictionStudio,
    )

    df = _list_models_df(
        PredictionStudio,
        [
            {"modelId": "A!B", "label": "x", "futureField": "surprise"},
            {"modelId": "C!D", "label": "y", "anotherFuture": 7},
        ],
        mocker,
    )
    assert df.columns == _BASE_FIELD_ORDER


def test_polars_schema_excludes_extras_and_types_fields():
    import polars as pl

    schema = ModelDataV26_1.polars_schema()
    assert list(schema.keys()) == [
        *_BASE_FIELD_ORDER,
        "performance",
        "performance_measure",
    ]
    assert schema["model_id"] == pl.String
    assert schema["last_update_time"] == pl.Datetime
    assert schema["performance"] == pl.Float64
