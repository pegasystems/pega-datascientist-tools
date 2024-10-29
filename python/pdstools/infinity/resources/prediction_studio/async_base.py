from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Literal, Optional, TypedDict


from ...internal._exceptions import IncompatiblePegaVersionError
from ...internal._pagination import PaginatedList
from ...internal._resource import AsyncAPIResource


class Metrics(TypedDict):
    lift: float
    performance: float
    performanceMeasure: Literal["AUC"]


class Stage(TypedDict):
    objective: str
    responseTimeoutType: str
    responseTimeoutValue: int
    targetLabels: List[Dict[Literal["label"], str]]
    alternateLabels: List[Dict[Literal["label"], str]]


class Prediction(AsyncAPIResource, ABC):
    def __init__(
        self,
        client,
        *,
        predictionId: str,
        label: str,
        objective: str,
        status: str,
        lastUpdateTime: str,
        subject: Optional[str] = None,
    ):
        super().__init__(client=client)
        self.prediction_id = predictionId
        self.label = label
        self.objective = objective
        self.subject = subject
        self.status = status
        self.last_update_time = datetime.strptime(lastUpdateTime, "%Y%m%dT%H%M%S.%f %Z")

    @abstractmethod
    async def get_metric(
        self,
        *,
        metric: Literal["Performance", "Total_responses", "Lift", "Success_rate"],
        timeframe: Literal["7d", "4w", "3m", "6m"],
    ): ...


class Repository(AsyncAPIResource, ABC):
    name: str

    @property
    def s3_url(self) -> str:
        raise IncompatiblePegaVersionError("24.2", "Retrieving the S3 URL directly")


class PredictionStudioBase(AsyncAPIResource, ABC):
    version: str

    @abstractmethod
    def list_predictions(self) -> PaginatedList[Prediction]: ...

    @abstractmethod
    def repository(self) -> Repository: ...
