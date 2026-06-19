from __future__ import annotations

from datetime import datetime  # noqa: TC003  # Pydantic resolves this annotation at runtime

from pydantic import Field, field_validator

from ._base import ResourceData, parse_pega_datetime


class ModelData(ResourceData):
    """Typed payload for a Prediction Studio model.

    Fields use Pythonic snake_case names with Pega camelCase aliases. Unknown
    fields from newer Pega releases are retained (``extra="allow"``) and surface
    on ``model_dump()`` / attribute access.

    Parameters
    ----------
    model_id : str
        Unique model identifier (alias ``modelId``).
    label : str
        Human-readable model name.
    model_type : str or None
        The model type, e.g. ``"Adaptive model"`` (alias ``modelType``).
    modeling_technique : str or None
        Underlying technique (alias ``modelingTechnique``).
    source : str or None
        Origin of the model, e.g. ``"Pega"``.
    status : str or None
        Lifecycle status, e.g. ``"Completed"``.
    component_name : str or None
        Component the model belongs to (alias ``componentName``).
    last_update_time : datetime or None
        Last modification time (alias ``lastUpdateTime``), parsed from the
        Pega timestamp format.
    updated_by : str or None
        Operator who last updated the model (alias ``updatedBy``).
    """

    model_id: str = Field(alias="modelId")
    label: str
    model_type: str | None = Field(default=None, alias="modelType")
    modeling_technique: str | None = Field(default=None, alias="modelingTechnique")
    source: str | None = None
    status: str | None = None
    component_name: str | None = Field(default=None, alias="componentName")
    last_update_time: datetime | None = Field(default=None, alias="lastUpdateTime")
    updated_by: str | None = Field(default=None, alias="updatedBy")

    @field_validator("last_update_time", mode="before")
    @classmethod
    def _parse_last_update_time(cls, value: object) -> datetime | None:
        return parse_pega_datetime(value)


class ModelDataV26_1(ModelData):
    """Typed payload for a Prediction Studio model in API v26.1+.

    Extends :class:`ModelData` with the performance metrics surfaced by the
    v26.1 models endpoint.

    Parameters
    ----------
    performance : float or None
        Model performance metric (e.g. AUC).
    performance_measure : str or None
        The performance metric name (alias ``performanceMeasure``).
    """

    performance: float | None = None
    performance_measure: str | None = Field(default=None, alias="performanceMeasure")
