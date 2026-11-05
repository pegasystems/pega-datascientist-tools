from __future__ import annotations

import logging
import random
import string
from typing import TYPE_CHECKING, Any

from pydantic import validate_call

from .....internal._exceptions import PegaException, PegaMLopsError
from .....internal._resource import _maybe_await, api_method
from ...types import AdmModelType
from ..model_upload import UploadedModel

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class _ChampionChallengerv26_1Mixin:
    """v26 ChampionChallenger business logic — shared parts."""

    # Declared for mypy — provided by concrete base classes at runtime
    if TYPE_CHECKING:
        prediction_id: str
        _a_patch: Callable[..., Any]
        _a_post: Callable[..., Any]
        _a_get: Callable[..., Any]
        _sleep: Callable[..., Any]
        _client: Any

    def __init__(
        self,
        client,
        prediction_id: str,
        active_model,
        cc_id: str | None = None,
        context: str | None = None,
        category: str | None = None,
        challenger_model=None,
        champion_percentage: float | None = None,
        model_objective: str | None = None,
    ):
        super().__init__(client=client)  # type: ignore[call-arg]  # cooperative mixin init resolves at runtime; mypy sees object.__init__
        self.prediction_id = prediction_id
        self.cc_id = cc_id
        self.context = context
        self.category = category
        self.active_model = active_model
        self.challenger_model = challenger_model
        self.champion_percentage = champion_percentage
        self.model_objective = model_objective
        self._removed = False

    def describe(self) -> dict | str:
        """Describe the champion challenger object."""
        if self._removed:
            return "Champion challenger object has been removed."

        champion_challenger_dict: dict[str, Any] = {
            "prediction_id": self.prediction_id,
            "context": None if self.context == "NoContext" else self.context,
            "category": self.category,
            "model_objective": self.model_objective,
        }

        if self.active_model:
            if self.active_model.modeling_technique == "Unknown model type":
                self.active_model.modeling_technique = "Unknown"
            champion_challenger_dict["active_model"] = {
                "model_id": self.active_model.model_id,
                "label": self.active_model.label,
                "component_name": self.active_model.component_name,
                "model_type": self.active_model.model_type,
                "modeling_technique": self.active_model.modeling_technique
                if self.active_model.modeling_technique
                else None,
                "role": self.active_model.status,
                "champion_percentage": 100 if self.champion_percentage is None else self.champion_percentage,
            }

        if self.challenger_model:
            if self.challenger_model.modeling_technique == "Unknown model type":
                self.challenger_model.modeling_technique = "Unknown"
            champion_challenger_dict["challenger_model"] = {
                "model_id": self.challenger_model.model_id,
                "label": self.challenger_model.label,
                "component_name": self.challenger_model.component_name,
                "model_type": self.challenger_model.model_type,
                "modeling_technique": self.challenger_model.modeling_technique
                if self.challenger_model.modeling_technique
                else None,
                "role": self.challenger_model.status,
                "challenger_percentage": 100 - self.champion_percentage
                if self.champion_percentage is not None
                else 100,
            }

        return champion_challenger_dict

    async def _refresh_champion_challenger(self):
        """Updates the champion and challenger models for a specific prediction."""
        prediction = await _maybe_await(
            self._client.prediction_studio.get_prediction(
                prediction_id=self.prediction_id,
            ),
        )
        champion_challengers = await _maybe_await(prediction.get_champion_challengers())
        for champion_challenger in champion_challengers:
            if champion_challenger.active_model.component_name == self.active_model.component_name:
                self.active_model = champion_challenger.active_model
                self.champion_percentage = champion_challenger.champion_percentage
                self.model_objective = champion_challenger.model_objective
                if champion_challenger.challenger_model:
                    self.challenger_model = champion_challenger.challenger_model
                else:
                    self.challenger_model = None
                break

    async def _status(self) -> dict[str, Any]:
        """Checks the update status of the champion challenger configuration."""
        if not self.cc_id:
            return {"ModelUpdateStatus": "Active", "message": "Active"}
        endpoint = f"/prweb/api/PredictionStudio/v4/predictions/operations/{self.cc_id}"
        return await self._a_get(endpoint)

    async def _introduce_model(
        self,
        champion_response_share: float,
        learn_independently: bool = True,
        action: str = "Approve",
        review_note: str = "Approved",
    ):
        if not self.cc_id:
            return "Model already introduced."
        endpoint = f"/prweb/api/PredictionStudio/v4/predictions/operations/{self.cc_id}"
        if self.active_model.model_type.upper() != "SCORECARD":
            if champion_response_share == 1:
                deployment_mode: dict[str, Any] = {"type": "Shadow"}
            else:
                deployment_mode = {
                    "type": "ChampionChallenger",
                    "championPercentage": champion_response_share * 100,
                    "learnIndependently": learn_independently,
                }

            data = {
                "action": action,
                "reviewNote": review_note,
                "deploymentMode": deployment_mode,
            }
        else:
            data = {
                "action": action,
                "reviewNote": review_note,
                "deploymentMode": {
                    "type": "Replace",
                },
            }
        return await self._a_patch(endpoint=endpoint, data=data)

    async def _check_then_update(
        self,
        champion_response_share: float,
        learn_independently: bool = True,
        auto_approve: bool = True,
    ):
        if not self.cc_id:
            return "Model already introduced."

        status_order = [
            "Not started",
            "Configuration in progress",
            "Validation in progress",
            "Ready for review",
            "Approved",
        ]
        try:
            from tqdm import tqdm
        except ImportError:

            class tqdm:  # type: ignore[no-redef]  # intentional fallback shadowing the imported name
                def __init__(self, total=None):
                    self.n = 0

                def set_description(self, *_a, **_k):
                    pass

                def refresh(self):
                    pass

                def update(self, *_a, **_k):
                    self.n += 1

        pbar = tqdm(total=len(status_order))
        model_status = None
        while model_status not in ("Ready for review", "Approved"):
            await self._sleep(1)
            try:
                status = await self._status()
            except PegaException:
                # In some Pega versions the referenceID returned by the clone
                # endpoint is not queryable via the operations endpoint —
                # the model may have been auto-approved on creation.
                logger.debug(
                    "Status polling failed for cc_id=%s; verifying via refresh.",
                    self.cc_id,
                )
                await self._refresh_champion_challenger()
                if self.challenger_model:
                    return {"message": "Approved"}
                raise PegaMLopsError(
                    "Status polling failed and no challenger model was found. "
                    "The model may not have been added successfully."
                ) from None

            model_status = status["ModelUpdateStatus"]

            if model_status.strip() not in status_order:
                raise PegaMLopsError(
                    f"Error when adding model: {status['ModelUpdateStatus']}",
                )
            pbar.set_description(f"Status: {model_status}")
            pbar.n = status_order.index(model_status) + 1
            pbar.refresh()

        if model_status == "Approved":
            await self._refresh_champion_challenger()
            return {"message": "Approved"}

        if not auto_approve:
            await self._refresh_champion_challenger()
            return {"message": "Ready for review"}

        pbar.set_description("Introducing model...")

        response = await self._introduce_model(
            champion_response_share,
            learn_independently,
        )
        pbar.set_description("Model approved succesfully")
        pbar.update()
        return response

    @api_method
    async def delete_challenger_model(self):
        """Removes the challenger model linked to the current prediction.

        Raises
        ------
        PegaMLopsError
            If there's no challenger model linked to the prediction.

        """
        if not self.challenger_model:
            raise PegaMLopsError("Challenger model is not set.")
        endpoint = f"/prweb/api/PredictionStudio/v4/predictions/{self.prediction_id}/models/{self.challenger_model.model_id}/Remove"
        data = {"contextName": self.context}
        try:
            response = await self._a_patch(endpoint, data=data)
        except PegaException as e:
            raise PegaMLopsError(
                "Error when deleting challenger model: " + str(e),
            ) from e
        await self._refresh_champion_challenger()
        logger.info("Deleted challenger model: %s", response)

    @api_method
    async def promote_challenger_model(self):
        """Switches the challenger model to be the new champion for a prediction.

        Raises
        ------
        PegaMLopsError
            If there's no challenger model linked to the prediction.

        """
        if not self.challenger_model:
            raise PegaMLopsError("Challenger model is not set.")
        endpoint = f"/prweb/api/PredictionStudio/v4/predictions/{self.prediction_id}/models/{self.challenger_model.model_id}/Promote"
        data = {"contextName": self.context}
        try:
            response = await self._a_patch(endpoint, data=data)
            await self._refresh_champion_challenger()
        except PegaException as e:
            raise PegaMLopsError(
                "Error when promoting challenger model: " + str(e),
            ) from e
        logger.info("Promoted challenger model: %s", response)

    @api_method
    async def update_challenger_response_share(
        self,
        new_challenger_response_share: float,
    ):
        """Adjusts traffic distribution between champion and challenger models.

        Parameters
        ----------
        new_challenger_response_share : float
            The desired challenger_response_percentage directed to the
            challenger model.

        Raises
        ------
        ValueError
            If the percentage is outside the 0-1 range, or if either the
            champion or challenger model is missing.
        PegaMLopsError
            If there's an error when updating the challenger response percentage

        """
        if not (0 <= new_challenger_response_share <= 1):
            raise ValueError("Percentage must be between 0 and 1.")
        if not self.active_model:
            raise ValueError("Active model is not set.")
        if not self.challenger_model:
            raise ValueError("Challenger model is not set.")

        if self.challenger_model.status.upper() == "SHADOW":
            endpoint = f"/prweb/api/PredictionStudio/v4/predictions/{self.prediction_id}/models/{self.challenger_model.model_id}/updatePattern"
            data = {
                "contextName": self.context,
                "challengerPercentage": new_challenger_response_share * 100,
            }
        else:
            endpoint = f"/prweb/api/PredictionStudio/v4/predictions/{self.prediction_id}/models/{self.active_model.model_id}/distribution"
            data = {
                "contextName": self.context,
                "championPercentage": float(1 - new_challenger_response_share) * 100,
            }
        try:
            response = await self._a_patch(endpoint, data=data)
        except PegaException as e:
            raise PegaMLopsError(
                "Error when updating challenger model percentage: " + str(e),
            ) from e
        await self._refresh_champion_challenger()
        logger.info("Updated challenger response percentage: %s", response)

    @api_method
    async def add_predictor(
        self,
        name: str,
        predictor_type: str,
        value: str,
        data_type: str,
        is_active_model: bool,
        parameterized: bool = True,
    ):
        """Adds a new predictor to a specific model in a prediction setup.

        Parameters
        ----------
        name : str
            Name of the predictor.
        predictor_type : str
            Predictor's type.
        value : str
            Predictor's value, for static predictors.
        data_type : str
            Data type of the predictor's value.
        is_active_model: bool
            Indicates if the predictor should be added to the active model
            or the challenger model.
        parameterized : bool, optional
            Indicates if the predictor is parameterized, default is True.

        Returns
        -------
        dict
            Outcome of the addition operation as received from the API.

        Raises
        ------
        PegaMLopsError
            If the challenger model is not set or if an error occurs when
            adding the predictor.

        """
        if is_active_model:
            model = self.active_model
        else:
            if not self.challenger_model:
                raise PegaMLopsError("Challenger model is not set.")
            model = self.challenger_model
        endpoint = (
            f"/prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/models/{model.model_id}/predictor/add"
        )
        if parameterized:
            predictor_category = "parameterized"
        else:
            raise NotImplementedError("Static predictors are not supported.")
        data = {
            "predictorName": name,
            "predictorCategory": predictor_category,
            "contextName": self.context,
            "dataType": data_type,
        }

        if predictor_type:
            data["predictorType"] = predictor_type
        elif data_type in ["Text", "TrueFalse"]:
            data["predictorType"] = "symbolic"
        else:
            data["predictorType"] = "numeric"

        if value:
            data["paramPredictorValue"] = value
        try:
            response = await self._a_patch(endpoint, data=data)
        except PegaException as e:
            raise PegaMLopsError("Error when Adding predictor: " + str(e)) from e
        return response

    @api_method
    async def remove_predictor(
        self,
        name: str,
        parameterized: bool | None = False,
        is_active_model: bool = True,
    ):
        """Removes a predictor from a model in a prediction setup.

        Parameters
        ----------
        name : str
            The name of the predictor to remove.
        parameterized : bool, optional
            True if the predictor is parameterized, False if it's static.
            Defaults to False.
        is_active_model : bool, optional
            If True, removes from the active model; otherwise from the
            challenger. Defaults to True.

        Returns
        -------
        dict
            The result of the deletion request.

        Raises
        ------
        PegaMLopsError
            If the challenger model is not set or if an error occurs when
            removing the predictor.

        """
        if is_active_model:
            model = self.active_model
        else:
            if not self.challenger_model:
                raise PegaMLopsError("Challenger model is not set.")
            model = self.challenger_model

        endpoint = (
            f"/prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/models/{model.model_id}/predictor/remove"
        )
        if parameterized:
            predictorCategory = "parameterized"
        else:
            raise NotImplementedError
        data = {
            "predictorName": name,
            "predictorCategory": predictorCategory,
            "contextName": self.context,
        }
        try:
            return await self._a_patch(endpoint, data=data)
        except PegaException as e:
            raise PegaMLopsError("Error when removing predictor: " + str(e)) from e

    @api_method
    async def add_model(
        self,
        new_model,
        challenger_response_share: float,
        predictor_mapping: list[dict[str, str | int]] | None = None,
        model_label: str | None = None,
        learn_independently: bool = True,
    ):
        """Add a new model as a challenger in the prediction setup.

        Parameters
        ----------
        new_model : Model, UploadedModel
            The model to be added as a challenger.
        challenger_response_share : int
            Defines what percentage of traffic should be directed to the
            challenger model.
        predictor_mapping : list of dict, optional
            Custom mappings for predictors to properties.
        model_label : str, optional
            A label for the new challenger model.
        learn_independently: bool, optional
            If True, the challenger model will learn independently.
            Defaults to True.

        Raises
        ------
        ValueError
            If the response_percentage for the challenger is outside the
            0-1 range.
        PegaMLopsError
            If there's an error when adding the challenger model.

        """
        objective = None
        if not (0 <= challenger_response_share <= 1):
            raise ValueError("Percentage must be between 0 and 1.")
        endpoint = f"/prweb/api/PredictionStudio/v4/predictions/{self.prediction_id}/component/{self.active_model.component_name}"
        data: dict[str, Any] = {}
        if hasattr(new_model, "model_id") and not isinstance(new_model, UploadedModel):
            new_model = new_model.model_id.split("!")[1]
        elif isinstance(new_model, UploadedModel):
            data["sourceType"] = "Uploaded Model"
            if model_label is None:
                model_label = new_model.file_path.split("/")[-1].split(".")[0]
                new_model = new_model.file_path.split("/")[-1]
            objective = self.model_objective
            data["objective"] = objective
            data["modelLabel"] = model_label
        else:
            data["sourceType"] = "Existing Model"

        data["modelSource"] = new_model

        if predictor_mapping:
            override_mappings = [
                {"predictor": key["predictor"], "property": key["property"]} for key in predictor_mapping
            ]
            data["overrideMappings"] = override_mappings
        try:
            response = await self._a_post(endpoint, data=data)
            self.cc_id = response["referenceID"]
            champion_response_share = float(1 - challenger_response_share)

            response = await self._check_then_update(
                champion_response_share=champion_response_share,
                learn_independently=learn_independently,
            )
        except PegaException as e:
            raise PegaMLopsError("Error when Adding challenger model: " + str(e)) from e

        logger.info("Add model: Refreshing Champion challenger configuration: ")
        await self._sleep(1)
        await self._refresh_champion_challenger()
        logger.info("Add model: %s", response)

    @api_method
    @validate_call
    async def clone_model(
        self,
        challenger_response_share: float,
        adm_model_type: AdmModelType | str,
        model_label: str | None = None,
        predictor_mapping: list[dict] | None = None,
        learn_independently: bool = True,
    ):
        """Clones the current active model to create a challenger.

        Parameters
        ----------
        challenger_response_share : float
            Defines the traffic percentage for the challenger model.
        adm_model_type : {'Gradient boosting', 'Naive bayes'}
            Specifies the type of the cloned model.
        model_label : str, optional
            A custom label for the cloned model.
        predictor_mapping : list of dict, optional
            Custom mappings of predictors to properties for the cloned model.
        learn_independently: bool, optional
            If True, the challenger model will learn independently.
            Defaults to True.

        Raises
        ------
        PegaMLopsError
            If the challenger_response_percentage is not within the 0-1 range
            or adm_model_type is invalid.

        """
        if isinstance(adm_model_type, AdmModelType):
            adm_model_type = adm_model_type.value
        valid_adm_model_types = [
            AdmModelType.GRADIENT_BOOSTING.value,
            AdmModelType.NAIVE_BAYES.value,
        ]
        if adm_model_type not in valid_adm_model_types:
            raise PegaMLopsError(
                "Invalid adm model type. It should be 'Gradient boosting' or 'Naive bayes'",
            )
        if not (0 <= challenger_response_share <= 1):
            raise PegaMLopsError("Percentage must be between 0 and 1.")
        endpoint = f"/prweb/api/PredictionStudio/v4/predictions/{self.prediction_id}/component/{self.active_model.component_name}/clone"
        if model_label is None:
            unique_suffix = "".join(random.choices(string.ascii_uppercase, k=3))
            model_label = self.active_model.component_name + "_copy_" + unique_suffix

        data = {
            "modelLabel": model_label,
            "admModelType": adm_model_type,
        }
        if predictor_mapping is not None:
            data["overrideMappings"] = [
                {"predictor": key["predictor"], "property": key["property"]} for key in predictor_mapping
            ]
        try:
            response = await self._a_post(endpoint, data=data)
            self.cc_id = response["referenceID"]
            champion_response_percentage = float(1 - challenger_response_share)
            response = await self._check_then_update(
                champion_response_share=champion_response_percentage,
                learn_independently=learn_independently,
            )
        except PegaException as e:
            raise PegaMLopsError("Error when Adding challenger model: " + str(e)) from e
        await self._refresh_champion_challenger()
        logger.info("Clone model: %s", response)
