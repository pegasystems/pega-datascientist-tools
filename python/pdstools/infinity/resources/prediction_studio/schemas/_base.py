from __future__ import annotations

from datetime import date, datetime
from typing import get_args

import polars as pl
from pydantic import BaseModel, ConfigDict

# Pega serialises timestamps as e.g. "20240718T120552.671 GMT".  Pydantic's
# default datetime parser does not understand this, so every *Data model
# parses it explicitly via :func:`parse_pega_datetime` in a ``mode="before"``
# field validator.
_PEGA_DATETIME_FORMAT = "%Y%m%dT%H%M%S.%f %Z"

# Maps the Python annotation of a declared field to the Polars dtype used when
# materialising ``return_df=True`` results.  Anything unrecognised falls back
# to ``pl.String`` (Pega payloads are overwhelmingly string-typed).
_PY_TO_POLARS: dict[type, pl.DataType] = {
    str: pl.String(),
    bool: pl.Boolean(),
    int: pl.Int64(),
    float: pl.Float64(),
    datetime: pl.Datetime(),
    date: pl.Date(),
}


def parse_pega_datetime(value: object) -> datetime | None:
    """Parse a Pega timestamp string into a naive ``datetime``.

    Parameters
    ----------
    value : object
        The raw value from the API payload.  May already be a ``datetime``
        (passed through unchanged), ``None`` (returns ``None``), or a Pega
        timestamp string such as ``"20240718T120552.671 GMT"``.

    Returns
    -------
    datetime or None
        The parsed naive datetime, or ``None`` when the input is empty.

    Raises
    ------
    ValueError
        When ``value`` is a string that does not match the Pega format.

    """
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        if not value:
            return None
        return datetime.strptime(value, _PEGA_DATETIME_FORMAT)
    raise ValueError(f"Cannot parse Pega datetime from {type(value).__name__}: {value!r}")


class ResourceData(BaseModel):
    """Base for all Prediction Studio payload models.

    Centralises the Pydantic configuration shared by every ``*Data`` model:

    * ``populate_by_name`` — accept both the API alias (``modelId``) and the
      Pythonic field name (``model_id``).
    * ``extra="allow"`` — retain forward-compatible fields a future Pega
      release may add, instead of silently dropping them.
    * ``protected_namespaces=()`` — Pega payloads legitimately use
      ``model_*`` field names; disable Pydantic's protected-namespace
      warning rather than mangle the public field names.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
        protected_namespaces=(),
    )

    @classmethod
    def polars_schema(cls) -> dict[str, pl.DataType]:
        """Return a stable Polars schema for the declared fields.

        Only declared model fields are included (forward-compatible
        ``extra="allow"`` fields are intentionally excluded), so the
        ``return_df=True`` code paths produce a column set and dtypes that do
        not depend on which optional fields a given API response happened to
        populate.

        Returns
        -------
        dict[str, polars.DataType]
            Field name to Polars dtype, in field-declaration order.
        """
        schema: dict[str, pl.DataType] = {}
        for name, field in cls.model_fields.items():
            non_none = [arg for arg in get_args(field.annotation) if arg is not type(None)]
            base_type = non_none[0] if non_none else field.annotation
            schema[name] = _PY_TO_POLARS.get(base_type, pl.String())
        return schema

    @property
    def public_dict(self) -> dict[str, object]:
        """Declared-field payload for DataFrame export (extras excluded)."""
        dump = self.model_dump()
        return {name: dump[name] for name in type(self).model_fields}
