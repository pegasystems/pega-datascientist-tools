"""Exact-value tests for the Prediction Studio Pydantic data layer.

These cover the ``PredictionData`` payload models directly
(alias handling, forward-compat field retention, Pega datetime parsing, and
the locked ``model_dump`` schema) plus the ``_data_cls`` selection wired into
the sync/async ``Prediction`` resources.
"""

from __future__ import annotations

import datetime

import pytest

from pdstools.infinity.resources.prediction_studio.schemas import (
    PredictionData,
    ResourceData,
)
from pdstools.infinity.resources.prediction_studio.v24_2.prediction import (
    AsyncPrediction as AsyncPredictionV24_2,
    Prediction as PredictionV24_2,
)
from pdstools.infinity.resources.prediction_studio.v26_1.prediction import (
    AsyncPrediction as AsyncPredictionV26_1,
    Prediction as PredictionV26_1,
)

# A realistic camelCase payload as the Pega predictions endpoint returns it.
_PAYLOAD = {
    "predictionId": "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS",
    "label": "Predict Cards Acceptance",
    "objective": "Accept",
    "subject": "Customer",
    "status": "Completed",
    "lastUpdateTime": "20240718T120557.925 GMT",
}

_EXPECTED_DATETIME = datetime.datetime(2024, 7, 18, 12, 5, 57, 925000)

_BASE_FIELD_ORDER = [
    "prediction_id",
    "label",
    "objective",
    "subject",
    "status",
    "last_update_time",
]


# --------------------------------------------------------------------------- #
# PredictionData
# --------------------------------------------------------------------------- #
def test_prediction_data_accepts_camelcase_aliases():
    data = PredictionData.model_validate(_PAYLOAD)
    assert data.prediction_id == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
    assert data.label == "Predict Cards Acceptance"
    assert data.objective == "Accept"
    assert data.subject == "Customer"
    assert data.status == "Completed"
    assert data.last_update_time == _EXPECTED_DATETIME


def test_prediction_data_accepts_snake_case_names():
    data = PredictionData.model_validate({"prediction_id": "A!B", "label": "x", "objective": "Accept"})
    assert data.prediction_id == "A!B"
    assert data.objective == "Accept"


def test_prediction_data_parses_pega_datetime():
    data = PredictionData.model_validate(_PAYLOAD)
    assert data.last_update_time == _EXPECTED_DATETIME


def test_prediction_data_missing_optionals_default_to_none():
    data = PredictionData.model_validate({"predictionId": "A!B", "label": "x"})
    assert data.objective is None
    assert data.subject is None
    assert data.last_update_time is None


def test_prediction_data_dump_locks_schema_order():
    data = PredictionData.model_validate({"predictionId": "A!B", "label": "x"})
    # Even with everything optional missing, all base fields are present.
    assert list(data.model_dump().keys()) == _BASE_FIELD_ORDER


def test_prediction_data_retains_forward_compat_fields():
    data = PredictionData.model_validate({**_PAYLOAD, "futureField": "surprise"})
    dump = data.model_dump()
    assert dump["futureField"] == "surprise"
    # Unknown extras land after the declared fields.
    assert list(dump.keys()) == [*_BASE_FIELD_ORDER, "futureField"]


def test_prediction_data_is_resource_data_subclass():
    assert issubclass(PredictionData, ResourceData)


# --------------------------------------------------------------------------- #
# Resource wiring: _data_cls selection + attribute delegation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "pred_cls",
    [
        PredictionV24_2,
        AsyncPredictionV24_2,
        PredictionV26_1,
        AsyncPredictionV26_1,
    ],
)
def test_prediction_uses_prediction_data(pred_cls, mocker):
    prediction = pred_cls(client=mocker.MagicMock(), **_PAYLOAD)
    assert type(prediction._data) is PredictionData
    assert list(prediction._public_dict.keys()) == _BASE_FIELD_ORDER


def test_prediction_attribute_access_delegates_to_data(mocker):
    prediction = PredictionV24_2(client=mocker.MagicMock(), **_PAYLOAD)
    assert prediction.prediction_id == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
    assert prediction.last_update_time == _EXPECTED_DATETIME
    assert prediction.subject == "Customer"


def test_prediction_public_dict_is_snake_case_with_values(mocker):
    prediction = PredictionV24_2(client=mocker.MagicMock(), **_PAYLOAD)
    public = prediction._public_dict
    assert public["prediction_id"] == "CDHSAMPLE-DATA-CUSTOMER!PREDICTCUSTOMERACCEPTSCARDS"
    assert public["last_update_time"] == _EXPECTED_DATETIME
    assert public["objective"] == "Accept"


def test_prediction_unknown_attribute_raises(mocker):
    prediction = PredictionV24_2(client=mocker.MagicMock(), **_PAYLOAD)
    with pytest.raises(AttributeError):
        _ = prediction.definitely_not_a_field


# --------------------------------------------------------------------------- #
# Locked DataFrame schema (empty / all-null / forward-compat extras)
# --------------------------------------------------------------------------- #
def _list_predictions_df(ps_cls, predictions, mocker):
    client = mocker.MagicMock()
    client.request.return_value = {"predictions": predictions}
    inst = ps_cls.__new__(ps_cls)
    inst._client = client
    return inst.list_predictions(return_df=True)


def test_list_predictions_empty_keeps_locked_schema(mocker):
    from pdstools.infinity.resources.prediction_studio.v24_2.prediction_studio._sync import (
        PredictionStudio,
    )

    df = _list_predictions_df(PredictionStudio, [], mocker)
    assert df.shape == (0, 6)
    assert df.columns == _BASE_FIELD_ORDER


def test_list_predictions_all_null_optionals_have_typed_columns(mocker):
    import polars as pl

    from pdstools.infinity.resources.prediction_studio.v24_2.prediction_studio._sync import (
        PredictionStudio,
    )

    df = _list_predictions_df(PredictionStudio, [{"predictionId": "A!B", "label": "x"}], mocker)
    # Optional columns are correctly typed (not Null), so dtype-specific ops work.
    assert df.schema["status"] == pl.String
    assert df.schema["last_update_time"] == pl.Datetime
    assert df.select(pl.col("last_update_time").dt.year()).to_dicts() == [{"last_update_time": None}]


def test_list_predictions_extras_do_not_leak_into_dataframe(mocker):
    from pdstools.infinity.resources.prediction_studio.v24_2.prediction_studio._sync import (
        PredictionStudio,
    )

    df = _list_predictions_df(
        PredictionStudio,
        [
            {"predictionId": "A!B", "label": "x", "futureField": "surprise"},
            {"predictionId": "C!D", "label": "y", "anotherFuture": 7},
        ],
        mocker,
    )
    assert df.columns == _BASE_FIELD_ORDER


def test_polars_schema_excludes_extras_and_types_fields():
    import polars as pl

    schema = PredictionData.polars_schema()
    assert list(schema.keys()) == _BASE_FIELD_ORDER
    assert schema["prediction_id"] == pl.String
    assert schema["last_update_time"] == pl.Datetime
