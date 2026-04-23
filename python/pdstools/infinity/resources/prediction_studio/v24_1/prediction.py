from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from pydantic import validate_call

from .....utils import cdh_utils
from ....internal._constants import METRIC  # noqa: TC001 — runtime needed by pydantic.validate_call
from ....internal._exceptions import NoMonitoringInfo
from ....internal._resource import api_method
from ..base import AsyncPrediction as AsyncPredictionBase
from ..base import Prediction as PredictionBase

if TYPE_CHECKING:
    from collections.abc import Callable


class _PredictionV24_1Mixin:
    """v24.1 Prediction business logic — defined once."""

    # Declared for mypy — provided by concrete base classes at runtime
    if TYPE_CHECKING:
        prediction_id: str
        _a_get: Callable[..., Any]

    @api_method
    @validate_call
    async def get_metric(
        self,
        *,
        metric: METRIC,
        timeframe: Literal["7d", "4w", "3m", "6m"],
    ) -> pl.DataFrame:
        endpoint = f"/prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/metric/{metric}"
        try:
            info = await self._a_get(endpoint, time_frame=timeframe)
            return (
                pl.DataFrame(
                    info["monitoringData"],
                    schema={
                        "value": pl.Utf8,
                        "snapshotTime": pl.Utf8,
                        "dataUsage": pl.Utf8,
                    },
                )
                .with_columns(
                    snapshotTime=cdh_utils.parse_pega_date_time_formats(
                        "snapshotTime",
                        "%Y-%m-%dT%H:%M:%S%.fZ",
                    ),
                    value=pl.col("value").replace("", None).cast(pl.Float64),
                    category=pl.col("dataUsage"),
                )
                .drop("dataUsage")
            )
        except NoMonitoringInfo:
            return pl.DataFrame(
                schema={
                    "value": pl.Float64,
                    "snapshotTime": pl.Datetime("ns"),
                    "category": pl.Utf8,
                },
            )

    @api_method
    async def describe(self):
        endpoint = f"/prweb/api/PredictionStudio/v3/predictions/{self.prediction_id}"
        return await self._a_get(endpoint)


class Prediction(_PredictionV24_1Mixin, PredictionBase):
    pass


class AsyncPrediction(_PredictionV24_1Mixin, AsyncPredictionBase):
    pass
