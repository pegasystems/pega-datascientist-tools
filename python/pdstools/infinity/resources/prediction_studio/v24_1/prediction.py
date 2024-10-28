from typing import Literal

import polars as pl
from .....utils import cdh_utils
from pydantic import validate_call

from ....internal._constants import METRIC
from ....internal._exceptions import NoMonitoringInfo
from ..base import Prediction as PredictionBase


class Prediction(PredictionBase):
    @validate_call
    def get_metric(
        self,
        *,
        metric: METRIC,
        timeframe: Literal["7d", "4w", "3m", "6m"],
    ) -> pl.DataFrame:
        endpoint = f"/prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/metric/{metric}"
        try:
            info = self._client.get(endpoint, time_frame=timeframe)
            data = (
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
                        "snapshotTime", "%Y-%m-%dT%H:%M:%S%.fZ"
                    ),
                    value=pl.col("value").replace("", None).cast(pl.Float64),
                    category=pl.col("dataUsage"),
                )
                .drop("dataUsage")
            )
            return data
        except NoMonitoringInfo:
            data = pl.DataFrame(
                schema={
                    "value": pl.Float64,
                    "snapshotTime": pl.Datetime("ns"),
                    "category": pl.Utf8,
                }
            )
            return data

    def describe(self):
        endpoint = f"/prweb/api/PredictionStudio/v3/predictions/{self.prediction_id}"
        return self._client.get(endpoint)
