from __future__ import annotations

from datetime import datetime  # noqa: TC003  # Pydantic resolves this annotation at runtime

from pydantic import Field, field_validator

from ._base import ResourceData, parse_pega_datetime


class PredictionData(ResourceData):
    """Typed payload for a Prediction Studio prediction.

    Fields use Pythonic snake_case names with Pega camelCase aliases. Unknown
    fields from newer Pega releases are retained (``extra="allow"``) and surface
    on ``model_dump()`` / attribute access.

    Parameters
    ----------
    prediction_id : str
        Unique prediction identifier (alias ``predictionId``).
    label : str
        Human-readable prediction name.
    objective : str or None
        Objective of the prediction.
    subject : str or None
        Subject of the prediction.
    status : str or None
        Lifecycle status, e.g. ``"Completed"``.
    last_update_time : datetime or None
        Last modification time (alias ``lastUpdateTime``), parsed from the
        Pega timestamp format.
    """

    prediction_id: str = Field(alias="predictionId")
    label: str
    objective: str | None = None
    subject: str | None = None
    status: str | None = None
    last_update_time: datetime | None = Field(default=None, alias="lastUpdateTime")

    @field_validator("last_update_time", mode="before")
    @classmethod
    def _parse_last_update_time(cls, value: object) -> datetime | None:
        return parse_pega_datetime(value)
