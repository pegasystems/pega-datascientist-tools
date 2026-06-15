from __future__ import annotations

from datetime import datetime  # noqa: TC003  # Pydantic resolves this annotation at runtime

from pydantic import Field, field_validator

from ._base import ResourceData, parse_pega_datetime


class NotificationData(ResourceData):
    """Typed payload for a Prediction Studio notification.

    Fields use Pythonic snake_case names with Pega camelCase aliases. Unknown
    fields from newer Pega releases are retained (``extra="allow"``) and surface
    on ``model_dump()`` / attribute access.

    ``model_id`` and ``prediction_id`` are optional because a notification is
    scoped to either a model or a prediction (never both). They are declared
    so that ``return_df=True`` results always expose both columns, regardless
    of which endpoint produced the notification.

    Parameters
    ----------
    model_id : str or None, default None
        Owning model identifier (alias ``modelID``).
    prediction_id : str or None, default None
        Owning prediction identifier (alias ``predictionID``).
    context : str or None, default None
        Context the notification applies to.
    notification_type : str
        Type of the notification (alias ``notificationType``).
    notification_id : str
        Unique notification identifier (alias ``notificationID``).
    notification_mnemonic : str
        Machine-readable notification code (alias ``notificationMnemonic``).
    description : str
        Human-readable description.
    model_type : str
        Type of the owning model (alias ``modelType``).
    impact : str
        Impact level of the notification.
    trigger_time : datetime
        Time the notification was raised (alias ``triggerTime``), parsed from
        the Pega timestamp format.
    """

    model_id: str | None = Field(default=None, alias="modelID")
    prediction_id: str | None = Field(default=None, alias="predictionID")
    context: str | None = None
    notification_type: str = Field(alias="notificationType")
    notification_id: str = Field(alias="notificationID")
    notification_mnemonic: str = Field(alias="notificationMnemonic")
    description: str
    model_type: str = Field(alias="modelType")
    impact: str
    trigger_time: datetime = Field(alias="triggerTime")

    @field_validator("trigger_time", mode="before")
    @classmethod
    def _parse_trigger_time(cls, value: object) -> datetime | None:
        return parse_pega_datetime(value)
