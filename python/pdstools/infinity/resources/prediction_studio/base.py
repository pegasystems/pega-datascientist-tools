from __future__ import annotations

from abc import ABC, abstractmethod
from pydantic import BaseModel
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
)

import polars as pl

from ...internal._exceptions import IncompatiblePegaVersionError
from ...internal._pagination import PaginatedList
from ...internal._resource import AsyncAPIResource, SyncAPIResource

if TYPE_CHECKING:
    from ...client import SyncAPIClient

DEPLOYMENT_MODE = Literal["shadow", "champion/challenger"]


class Metrics(TypedDict):
    lift: float
    performance: float
    performanceMeasure: Literal["AUC"]


class ModelAttributes(TypedDict):
    modelId: str
    status: str
    label: str
    type: str
    subject: str
    objective: str
    outcome_type: str
    model_type: str
    modeling_technique: str
    source: str
    targetLabels: List[Dict[Literal["label"], str]]
    alternativeLabels: List[Dict[Literal["label"], str]]
    metrics: Dict[str, Any]


class Model(SyncAPIResource, ABC):
    def __init__(
        self,
        client: SyncAPIClient,
        *,
        modelId: str,
        label: str,
        modelType: str,
        status: str,
        componentName: Union[str, None] = None,
        source: Union[str, None] = None,
        lastUpdateTime: Union[str, None] = None,
        modelingTechnique: Union[str, None] = None,
        updatedBy: Union[str, None] = None,
    ):
        super().__init__(client=client)
        self.model_id = modelId
        self.label = label
        self.model_type = modelType
        self.modeling_technique = modelingTechnique
        self.source = source
        self.status = status
        if componentName:
            self.component_name = componentName
        if lastUpdateTime:
            self.last_update_time = datetime.strptime(
                lastUpdateTime, "%Y%m%dT%H%M%S.%f %Z"
            )
        self.updated_by = updatedBy

    @abstractmethod
    def describe(self) -> ModelAttributes: ...


class Prediction(SyncAPIResource, ABC):
    def __init__(
        self,
        client: SyncAPIClient,
        *,
        predictionId: str,
        label: str,
        status: str,
        lastUpdateTime: str,
        objective: Optional[str] = None,
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
    def get_metric(
        self,
        *,
        metric: Literal["Performance", "Total_responses", "Lift", "Success_rate"],
        timeframe: Literal["7d", "4w", "3m", "6m"],
    ) -> pl.DataFrame: ...

    @abstractmethod
    def describe(self): ...


class Notification(SyncAPIResource, ABC):
    def __init__(
        self,
        client: SyncAPIClient,
        *,
        description: str,
        modelType: str,
        notificationID: str,
        notificationType: str,
        notificationMnemonic: str,
        context: str,
        impact: str,
        triggerTime: str,
        modelID: Union[str, None] = None,
        predictionID: Union[str, None] = None,
    ):
        super().__init__(client=client)
        if modelID:
            self.model_id = modelID
        if predictionID:
            self.prediction_id = predictionID
        if context:
            self.context = context
        self.notification_type = notificationType
        self.notification_id = notificationID
        self.notification_mnemonic = notificationMnemonic
        self.description = description
        self.model_type = modelType
        self.impact = impact
        self.trigger_time = datetime.strptime(triggerTime, "%Y%m%dT%H%M%S.%f %Z")


class UploadedModel(ABC): ...


class ModelValidationError(Exception):
    """Exception for errors during model validation."""

    pass


class LocalModel(BaseModel):
    def validate(self) -> bool:
        """
        Validates a model.

        Raises
        ------
            ModelValidationError: If the model is invalid or if the validation process fails.
        """
        pass

    def get_file_path(self) -> str:
        """
        Returns the file path of the model.

        Returns
        -------
            str: The file path of the model.
        """
        pass


class Repository(SyncAPIResource, ABC):
    name: str

    @property
    def s3_url(self) -> str:
        raise IncompatiblePegaVersionError("24.2", "Retrieving the S3 URL directly")


class DataMartExport(SyncAPIResource, ABC):
    def __init__(self, client, **kwargs):
        super().__init__(client=client)
        self.referenceId = kwargs.get("referenceId")
        self.location = kwargs.get("location")
        self.repositoryName = kwargs.get("repositoryName")


class PredictionStudioBase(SyncAPIResource, ABC):
    version: str

    @abstractmethod
    def list_predictions(self) -> PaginatedList[Prediction]: ...

    @abstractmethod
    def repository(self) -> Repository: ...


class AsyncPredictionStudioBase(AsyncAPIResource, ABC):
    version: str

    @abstractmethod
    async def list_predictions(self) -> PaginatedList[Prediction]: ...

    @abstractmethod
    async def repository(self) -> Repository: ...


class ChampionChallenger(SyncAPIResource, ABC):
    @abstractmethod
    def _status(self): ...

    @abstractmethod
    def _introduce_model(self, champion_response_percentage: float): ...

    @abstractmethod
    def _check_then_update(self, champion_response_percentage: float): ...
