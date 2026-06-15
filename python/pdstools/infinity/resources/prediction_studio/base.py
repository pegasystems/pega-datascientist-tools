from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    Literal,
    TypedDict,
    TYPE_CHECKING,
)

from pydantic import BaseModel

from ...internal._exceptions import IncompatiblePegaVersionError
from ...internal._resource import AsyncAPIResource, SyncAPIResource
from .schemas import ModelData, ModelInstanceData, NotificationData, PredictionData

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import polars as pl

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
    targetLabels: list[dict[Literal["label"], str]]
    alternativeLabels: list[dict[Literal["label"], str]]
    metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# Mixins — data initialisation + abstract contracts.
#
# These do NOT inherit from Sync/AsyncAPIResource.  They use cooperative
# ``super().__init__()`` so that when combined with a resource base the
# MRO correctly delegates to the right ``__init__``.
# ---------------------------------------------------------------------------


class _ModelMixin(ABC):
    """Behaviour wrapper around a :class:`ModelData` payload.

    The raw API payload is validated into ``self._data`` (a Pydantic model).
    Attribute reads fall through to ``_data`` via ``__getattr__``, so
    ``model.model_id`` etc. keep working, while ``_public_dict`` /
    ``_public_fields`` are sourced from ``_data`` so the ``return_df=True``
    code paths (``list_models``) produce a stable, fully-populated schema.
    """

    _data: ModelData
    _data_cls: type[ModelData] = ModelData

    def __init__(self, client, **payload):
        super().__init__(client=client)  # type: ignore[call-arg]  # cooperative MRO: combined with SyncAPIResource/AsyncAPIResource
        self._data = self._data_cls.model_validate(payload)

    def __getattr__(self, name: str):
        # __getattr__ only fires when normal attribute lookup fails, so real
        # methods and instance attributes still win.  Use object.__getattribute__
        # to fetch _data without re-entering __getattr__ (which would recurse
        # before _data is assigned during __init__).
        try:
            data = object.__getattribute__(self, "_data")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return getattr(data, name)
        except AttributeError:
            raise AttributeError(name) from None

    @property
    def _public_dict(self) -> dict:
        return self._data.public_dict

    @property
    def _public_fields(self) -> list[str]:
        return list(type(self._data).model_fields)

    @classmethod
    def _public_schema(cls) -> dict:
        """Polars schema for ``return_df=True`` results (locked column set)."""
        return cls._data_cls.polars_schema()

    @abstractmethod
    def describe(self) -> ModelAttributes: ...


class _PredictionMixin(ABC):
    """Behaviour wrapper around a :class:`PredictionData` payload.

    The raw API payload is validated into ``self._data`` (a Pydantic model).
    Attribute reads fall through to ``_data`` via ``__getattr__``, so
    ``prediction.prediction_id`` etc. keep working, while ``_public_dict`` /
    ``_public_fields`` are sourced from ``_data`` so the ``return_df=True``
    code paths (``list_predictions``) produce a stable, fully-populated schema.
    """

    _data: PredictionData
    _data_cls: type[PredictionData] = PredictionData

    def __init__(self, client, **payload):
        super().__init__(client=client)  # type: ignore[call-arg]  # cooperative MRO
        self._data = self._data_cls.model_validate(payload)

    def __getattr__(self, name: str):
        try:
            data = object.__getattribute__(self, "_data")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return getattr(data, name)
        except AttributeError:
            raise AttributeError(name) from None

    @property
    def _public_dict(self) -> dict:
        return self._data.public_dict

    @property
    def _public_fields(self) -> list[str]:
        return list(type(self._data).model_fields)

    @classmethod
    def _public_schema(cls) -> dict:
        """Polars schema for ``return_df=True`` results (locked column set)."""
        return cls._data_cls.polars_schema()

    @abstractmethod
    def get_metric(
        self,
        *,
        metric: Literal["Performance", "Total_responses", "Lift", "Success_rate"],
        timeframe: Literal["7d", "4w", "3m", "6m"],
    ) -> pl.DataFrame: ...

    @abstractmethod
    def describe(self): ...


class _NotificationMixin(ABC):
    """Behaviour wrapper around a :class:`NotificationData` payload.

    The raw API payload is validated into ``self._data`` (a Pydantic model).
    Attribute reads fall through to ``_data`` via ``__getattr__``, so
    ``notification.notification_id`` etc. keep working, while ``_public_dict`` /
    ``_public_fields`` are sourced from ``_data`` so the ``return_df=True``
    code paths (``get_notifications``) produce a stable, fully-populated schema.
    """

    _data: NotificationData
    _data_cls: type[NotificationData] = NotificationData

    def __init__(self, client, **payload):
        super().__init__(client=client)  # type: ignore[call-arg]  # cooperative MRO
        self._data = self._data_cls.model_validate(payload)

    def __getattr__(self, name: str):
        try:
            data = object.__getattribute__(self, "_data")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return getattr(data, name)
        except AttributeError:
            raise AttributeError(name) from None

    @property
    def _public_dict(self) -> dict:
        return self._data.public_dict

    @property
    def _public_fields(self) -> list[str]:
        return list(type(self._data).model_fields)

    @classmethod
    def _public_schema(cls) -> dict:
        """Polars schema for ``return_df=True`` results (locked column set)."""
        return cls._data_cls.polars_schema()


class UploadedModel(ABC): ...  # noqa: B024 — ABC marker for subclass discovery


class ModelValidationError(Exception):
    """Exception for errors during model validation."""


class LocalModel(BaseModel):
    def validate(self) -> bool:  # type: ignore[override]  # intentionally overrides BaseModel.validate
        """Validates a model.

        Raises
        ------
            ModelValidationError: If the model is invalid or if the validation process fails.

        """
        return True

    def get_file_path(self) -> str:
        """Returns the file path of the model.

        Returns
        -------
            str: The file path of the model.

        """
        raise NotImplementedError


class _RepositoryMixin(ABC):
    name: str

    @property
    def s3_url(self) -> str:
        raise IncompatiblePegaVersionError("24.2", "Retrieving the S3 URL directly")


class _DataMartExportMixin(ABC):  # noqa: B024 — ABC used for cooperative MRO with pydantic resources
    def __init__(self, client, **kwargs):
        super().__init__(client=client)
        self.referenceId = kwargs.get("referenceId")
        self.location = kwargs.get("location")
        self.repositoryName = kwargs.get("repositoryName")


class _PredictionStudioBaseMixin(ABC):
    version: str

    @abstractmethod
    def list_predictions(self): ...

    @abstractmethod
    def repository(self): ...


class _ChampionChallengerMixin(ABC):
    @abstractmethod
    def _status(self): ...

    @abstractmethod
    def _introduce_model(self, champion_response_percentage: float): ...

    @abstractmethod
    def _check_then_update(self, champion_response_percentage: float): ...


# ---------------------------------------------------------------------------
# Concrete base classes — combine mixin + resource base.
# These are what version-specific subclasses inherit from.
# ---------------------------------------------------------------------------


class Model(_ModelMixin, SyncAPIResource):
    pass


class AsyncModel(_ModelMixin, AsyncAPIResource):
    pass


class Prediction(_PredictionMixin, SyncAPIResource):
    pass


class AsyncPrediction(_PredictionMixin, AsyncAPIResource):
    pass


class Notification(_NotificationMixin, SyncAPIResource):
    pass


class AsyncNotification(_NotificationMixin, AsyncAPIResource):
    pass


class Repository(_RepositoryMixin, SyncAPIResource):
    pass


class AsyncRepository(_RepositoryMixin, AsyncAPIResource):
    pass


class DataMartExport(_DataMartExportMixin, SyncAPIResource):
    pass


class AsyncDataMartExport(_DataMartExportMixin, AsyncAPIResource):
    pass


class PredictionStudioBase(_PredictionStudioBaseMixin, SyncAPIResource):
    pass


class AsyncPredictionStudioBase(_PredictionStudioBaseMixin, AsyncAPIResource):
    pass


class ChampionChallenger(_ChampionChallengerMixin, SyncAPIResource):
    pass


class AsyncChampionChallenger(_ChampionChallengerMixin, AsyncAPIResource):
    pass


class _ModelInstanceMixin(ABC):
    """Behaviour wrapper around a :class:`ModelInstanceData` payload.

    The raw API payload is validated into ``self._data`` (a Pydantic model).
    Attribute reads fall through to ``_data`` via ``__getattr__``, so
    ``instance.instance_id`` etc. keep working, while ``_public_dict`` /
    ``_public_fields`` are sourced from ``_data`` so the ``return_df=True``
    code paths (``list_instances``) produce a stable, fully-populated schema.
    """

    _data: ModelInstanceData
    _data_cls: type[ModelInstanceData] = ModelInstanceData

    def __init__(self, client, **payload):
        super().__init__(client=client)  # type: ignore[call-arg]  # cooperative MRO: combined with SyncAPIResource/AsyncAPIResource
        self._data = self._data_cls.model_validate(payload)

    def __getattr__(self, name: str):
        try:
            data = object.__getattribute__(self, "_data")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return getattr(data, name)
        except AttributeError:
            raise AttributeError(name) from None

    @property
    def _public_dict(self) -> dict:
        return self._data.public_dict

    @property
    def _public_fields(self) -> list[str]:
        return list(type(self._data).model_fields)

    @classmethod
    def _public_schema(cls) -> dict:
        """Polars schema for ``return_df=True`` results (locked column set)."""
        return cls._data_cls.polars_schema()


class ModelInstance(_ModelInstanceMixin, SyncAPIResource):
    pass


class AsyncModelInstance(_ModelInstanceMixin, AsyncAPIResource):
    pass


__all__ = ["AsyncModelInstance", "ModelInstance"]
