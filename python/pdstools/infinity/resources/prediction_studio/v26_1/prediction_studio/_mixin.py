from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import anyio

from .....internal._resource import api_method
from ...local_model_utils import ONNXModel
from ..model_upload import UploadedModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...base import LocalModel


@dataclass(frozen=True)
class _StudioEndpoints:
    model_upload: str
    models: str
    predictions: str
    reports: str
    settings: str
    datamart_export: str
    notifications: str


class _PredictionStudiov26_1Mixin:
    """v26 PredictionStudio business logic — shared parts."""

    # Declared for mypy — provided by concrete base classes at runtime
    if TYPE_CHECKING:
        _a_post: Callable[..., Any]
        _a_get: Callable[..., Any]

    version: str = "26.1"
    _endpoints: ClassVar[_StudioEndpoints] = _StudioEndpoints("v1", "v2", "v3", "v1", "v1", "v1", "v2")
    _uploaded_model_cls = UploadedModel

    @api_method
    async def upload_model(self, model: LocalModel, file_name: str) -> UploadedModel:
        """Uploads a model to the repository.

        This function handles the uploading of a model to the Repository,
        creating an UploadedModel object for further MLops processes.

        Parameters
        ----------
        model : str or ONNX model object
            The model to be uploaded. This can be specified either as a file
            path (str) to the model file or as an ONNX model object.
        file_name : str
            The name of the file (including extension) to be uploaded to
            the repository.

        Returns
        -------
        UploadedModel
            Details about the uploaded model, including API response data.

        Raises
        ------
        ValueError
            Raised if an attempt is made to upload an ONNX model without
            the required conversion library.
        ModelValidationError
            If the model validation fails.

        """
        endpoint = f"/prweb/api/PredictionStudio/{self._endpoints.model_upload}/model"
        model.validate()
        if isinstance(model, ONNXModel):
            file_encode = base64.encodebytes(model._model.SerializeToString()).decode(
                "ascii",
            )
        else:
            file_bytes = await anyio.Path(model.get_file_path()).read_bytes()
            file_encode = base64.encodebytes(file_bytes).decode("ascii")

        data = {"fileSource": file_encode, "fileName": file_name}
        response = await self._a_post(endpoint, data=data)
        return self._uploaded_model_cls(
            repository_name=response["repositoryName"],
            file_path=response["filePath"],
        )

    @api_method
    async def get_model_categories(self):
        """Gets a list of model categories from Prediction Studio.

        This function gives you back a list where each item tells you about
        one type of model.  Which can be useful for adding a conditional
        model to a prediction.

        Returns
        -------
        list of dict
            Each dictionary in the list has details about a model category,
            like its name and other useful information.

        """
        endpoint = "/prweb/api/PredictionStudio/v3/predictions/modelCategories"
        response = await self._a_get(endpoint)
        return [item for item in response["categories"]]

    @api_method
    async def get_reports(self) -> list[dict]:
        """Fetches a list of reports from Prediction Studio.

        Returns
        -------
        list of dict
            Each dictionary contains details about a report.

        """
        endpoint = f"/prweb/api/PredictionStudio/{self._endpoints.reports}/reports"
        response = await self._a_get(endpoint)
        return response["reports"]

    @api_method
    async def get_settings(self) -> dict:
        """Fetches the Prediction Studio settings.

        Returns
        -------
        dict
            A dictionary containing the current Prediction Studio settings.

        """
        endpoint = f"/prweb/api/PredictionStudio/{self._endpoints.settings}/settings"
        return await self._a_get(endpoint)
