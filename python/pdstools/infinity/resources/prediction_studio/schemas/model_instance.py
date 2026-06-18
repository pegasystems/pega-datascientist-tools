from __future__ import annotations

from datetime import datetime  # noqa: TC003  # Pydantic resolves this annotation at runtime

from pydantic import Field, field_validator

from ._base import ResourceData, parse_pega_datetime


class ModelInstanceData(ResourceData):
    """Typed payload for a Prediction Studio model instance.

    Fields use Pythonic snake_case names with Pega camelCase aliases. Unknown
    fields from newer Pega releases are retained (``extra="allow"``) and surface
    on ``model_dump()`` / attribute access.

    Parameters
    ----------
    instance_id : str
        Unique model instance identifier (alias ``instanceID``).
    name : str or None, default None
        Human-readable instance name.
    type : str or None, default None
        The instance type.
    status : str or None, default None
        Lifecycle status.
    active : bool or None, default None
        Whether the instance is active.
    last_update_time : datetime or None, default None
        Last modification time (alias ``lastUpdateTime``), parsed from the
        Pega timestamp format.
    """

    instance_id: str = Field(alias="instanceID")
    name: str | None = None
    type: str | None = None
    status: str | None = None
    active: bool | None = None
    last_update_time: datetime | None = Field(default=None, alias="lastUpdateTime")

    @field_validator("last_update_time", mode="before")
    @classmethod
    def _parse_last_update_time(cls, value: object) -> datetime | None:
        return parse_pega_datetime(value)
