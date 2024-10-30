import logging
import random
import string
import time
from typing import Dict, List, Optional, Union

import polars as pl
from pydantic import validate_call

from ....internal._exceptions import PegaException, PegaMLopsError
from ....internal._pagination import PaginatedList
from ..base import ChampionChallenger as ChampionChallengerBase
from ..types import AdmModelType
from .model import Model
from .model_upload import UploadedModel

logger = logging.getLogger(__name__)


class ChampionChallenger(ChampionChallengerBase):
    """
    The `ChampionChallenger` class manages champion and challenger models
    within a prediction context. It provides functionalities for:

    - Refreshing champion challenger data
    - Deleting challenger models
    - Promoting challenger models
    - Updating challenger response percentages
    - Adding new models
    - Cloning an ADM active model
    - Adding/removing predictors

    Attributes
    ----------
    client : Client
        The client used to interact with the API.
    prediction_id : str
        The ID of the prediction.
    active_model : Model
        The active model in the prediction.
    cc_id : Union[str, None]
        The ID of the champion challenger.
    context : Union[str, None]
        The context of the prediction.
    category : Union[str, None]
        The category of the prediction.
    challenger_model : Union[Model, None]
        The challenger model.
    champion_percentage : Union[float, None]
        The percentage of responses attributed to the champion model.
    model_objective : Union[str, None]
        The objective of the model.
    """

    def __init__(
        self,
        client,
        prediction_id: str,
        active_model: Model,
        cc_id: Optional[str] = None,
        context: Optional[str] = None,
        category: Optional[str] = None,
        challenger_model: Optional[Model] = None,
        champion_percentage: Optional[float] = None,
        model_objective: Optional[str] = None,
    ):
        super().__init__(client=client)
        self.prediction_id = prediction_id
        self.cc_id = cc_id
        self.context = context
        self.category = category
        self.active_model = active_model
        self.challenger_model = challenger_model
        self.champion_percentage = champion_percentage
        self.model_objective = model_objective
        self._removed = False

    def describe(self) -> dict:
        """
        Describe the champion challenger object.
        """
        if self._removed:
            return "Champion challenger object has been removed."

        # Base dictionary
        champion_challenger_dict = {
            "prediction_id": self.prediction_id,
            "context": None if self.context == "NoContext" else self.context,
            "category": self.category,
            "model_objective": self.model_objective,
        }

        # Active model details
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
                "champion_percentage": 100
                if self.champion_percentage is None
                else self.champion_percentage,
            }

        # Challenger model details
        if self.challenger_model:
            if self.challenger_model.modeling_technique == "Unknown model type":
                self.challenger_model.modeling_technique = "Unknown"
            champion_challenger_dict["challenger_model"] = {
                "model_id": self.challenger_model.model_id,
                "label": self.challenger_model.label,
                "component_name": self.challenger_model.component_name,
                "model_type": self.challenger_model.model_type,
                "modeling_technique": self.challenger_model.modeling_technique
                if self.active_model.modeling_technique
                else None,
                "role": self.challenger_model.status,
                "challenger_percentage": 100 - self.champion_percentage,
            }

        return champion_challenger_dict

    def _refresh_champion_challenger(self):
        """
        Updates the champion and challenger models for a specific prediction.

        This function fetches the latest prediction details and refreshes the champion and challenger models accordingly.
        If there's no challenger model linked, it sets the challenger attribute to None.
        """
        prediction = self._client.prediction_studio.get_prediction(
            prediction_id=self.prediction_id
        )
        champion_challengers = prediction.get_champion_challengers()
        for champion_challenger in champion_challengers:
            if (
                champion_challenger.active_model.component_name
                == self.active_model.component_name
            ):
                self.active_model = champion_challenger.active_model
                self.champion_percentage = champion_challenger.champion_percentage
                self.model_objective = champion_challenger.model_objective
                if champion_challenger.challenger_model:
                    self.challenger_model = champion_challenger.challenger_model
                else:
                    self.challenger_model = None
                break

    def delete_challenger_model(self):
        """
        Removes the challenger model linked to the current prediction.

        This function checks for a challenger model's existence, constructs a request to delete it using the prediction and model IDs,
        and updates the prediction's data accordingly.

        Raises
        ------
        PegaMLopsError
            If there's no challenger model linked to the prediction.
        """
        if not self.challenger_model:
            raise PegaMLopsError("Challenger model is not set.")
        endpoint = f"prweb/api/PredictionStudio/v4/predictions/{self.prediction_id}/models/{self.challenger_model.model_id}/Remove"
        data = {"contextName": self.context}
        try:
            response = self._client.patch(endpoint, data=data)
        except PegaException as e:
            raise PegaMLopsError(
                "Error when deleting challenger model: " + str(e)
            ) from e
        self._refresh_champion_challenger()
        logging.info("Deleted challenger model: %s", response)

    def promote_challenger_model(self):
        """
        Switches the challenger model to be the new champion for a prediction.

        Checks for an existing challenger model and sends a request to make it the new champion model.
        Updates the prediction's model data afterwards.

        Raises
        ------
        PegaMLopsError
            If there's no challenger model linked to the prediction.
        """
        if not self.challenger_model:
            raise PegaMLopsError("Challenger model is not set.")
        endpoint = f"prweb/api/PredictionStudio/v4/predictions/{self.prediction_id}/models/{self.challenger_model.model_id}/Promote"
        data = {"contextName": self.context}
        response = self._client.patch(endpoint, data=data)
        try:
            self._refresh_champion_challenger()
        except PegaException as e:
            raise PegaMLopsError(
                "Error when promoting challenger model: " + str(e)
            ) from e
        logging.info("Promoted challenger model: %s", response)

    def update_challenger_response_share(self, new_challenger_response_share: float):
        """
        Adjusts traffic distribution between champion and challenger models.

        Modifies how incoming traffic is split between the current champion
        and the challenger model by updating the challenger's response
        percentage. Validates the new percentage and the existence of both
        models before applying changes.

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
            endpoint = f"prweb/api/PredictionStudio/v4/predictions/{self.prediction_id}/models/{self.challenger_model.model_id}/updatePattern"
            data = {
                "contextName": self.context,
                "challengerPercentage": new_challenger_response_share * 100,
            }
        else:
            endpoint = f"prweb/api/PredictionStudio/v4/predictions/{self.prediction_id}/models/{self.active_model.model_id}/distribution"
            data = {
                "contextName": self.context,
                "championPercentage": float(1 - new_challenger_response_share) * 100,
            }
        try:
            response = self._client.patch(endpoint, data)
        except PegaException as e:
            raise PegaMLopsError(
                "Error when updating challenger model percentage: " + str(e)
            ) from e
        self._refresh_champion_challenger()
        logger.info("Updated challenger response percentage: %s", response)

    def _status(self):
        """
        Checks the update status of the champion challenger configuration.

        Determines if an update to the champion challenger setup is currently in progress by examining the `cc_id`.
        If no update is underway, it indicates the current setup is active.

        Returns
        -------
        str
            The current status of the update process, or "Active" if no updates are pending.
        """
        if not self.cc_id:
            return "Active"
        endpoint = f"prweb/api/PredictionStudio/v4/predictions/{self.cc_id}"
        return self._client.get(endpoint)

    def _introduce_model(
        self, champion_response_share: float, learn_independently: bool = True
    ):
        if not self.cc_id:
            return "Model already introduced."
        endpoint = f"prweb/api/PredictionStudio/v4/predictions/{self.cc_id}"
        if self.active_model.model_type.upper() != "SCORECARD":
            if champion_response_share == 1:
                deployment_mode = {"type": "Shadow"}
            else:
                deployment_mode = {
                    "type": "ChampionChallenger",
                    "championPercentage": champion_response_share * 100,
                    "learnIndependently": learn_independently,
                }

            data = {
                "action": "Approve",
                "reviewNote": "Approving initial model update from API",
                "deploymentMode": deployment_mode,
            }
        else:
            data = {
                "action": "Approve",
                "reviewNote": "Approving initial model update from API",
                "deploymentMode": {
                    "type": "Replace",
                },
            }
        return self._client.patch(endpoint=endpoint, data=data)

    def _check_then_update(
        self, champion_response_share: float, learn_independently: bool = True
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
        from tqdm import tqdm  # TODO: make this fail gracefully when tqdm not installed

        pbar = tqdm(total=len(status_order))
        model_status = None
        while model_status != "Ready for review":
            time.sleep(1)
            status = self._status()

            model_status = status["ModelUpdateStatus"]

            if model_status.strip() not in status_order:
                raise PegaMLopsError(
                    f"Error when adding model: {status['ModelUpdateStatus']}"
                )
            pbar.set_description(f"Status: {model_status}")
            pbar.n = status_order.index(model_status) + 1
            pbar.refresh()

        pbar.set_description("Introducing model...")

        # Should only get here if status == 'Ready for review'
        response = self._introduce_model(champion_response_share, learn_independently)
        pbar.set_description("Model approved succesfully")
        pbar.update()
        return response

    def list_available_models_to_add(
        self, return_df: bool = False
    ) -> Union[PaginatedList[Model], pl.DataFrame]:
        """
        Fetches a list of models eligible to be challengers.

        Queries for models that can be added as challengers to the current
        prediction for the current active model. Offers the option to return
        the results in a DataFrame format for easier data handling.

        Parameters
        ----------
        return_df : bool, optional
            Determines the format of the returned data: a DataFrame if True,
            otherwise a list of model instances. Defaults to False.

        Returns
        -------
        PaginatedList[Model] or pl.DataFrame
            A list of model instances or a DataFrame of models, based on the
            `return_df` parameter choice.
        """
        endpoint = f"prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/component/{self.active_model.component_name}/replacement-options"
        pages = PaginatedList(Model, self._client, "get", endpoint, _root="models")
        if not return_df:
            return pages
        else:
            return pl.DataFrame([getattr(mod, "_public_dict") for mod in pages])

    def add_model(
        self,
        new_model: Union[Model, UploadedModel],
        challenger_response_share: float,
        predictor_mapping: Optional[List[Dict[str, Union[str, int]]]] = None,
        model_label: Optional[str] = None,
        learn_independently: Optional[bool] = True,
    ):
        """
        Add a new model as a challenger in the prediction setup.

        Enables the addition of a new model as a challenger, accepting various model types.
        It configures the challenger's traffic share, allows for custom predictor to property mappings,
        and supports labeling the model.
        If the active model is a scorecard, the function will replace the active model with the new challenger.

        Parameters
        ----------
        new_model : Model, UploadedModel
            The model to be added as a challenger. Can be a pre-existing model, an uploaded file, or a model identifier.
        challenger_response_share : int
            Defines what percentage of traffic should be directed to the challenger model.
        predictor_mapping : list of dict, optional
            Custom mappings for predictors to properties, with each mapping as a dictionary.
        model_label : str, optional
            A label for the new challenger model.
        learn_independently: bool, optional
            If True, the challenger model will learn independently. Defaults to True.

        Raises
        ------
        NotImplementedError
            If attempting to add a model instance directly, which is not supported.
        ValueError
            If the response_percentage for the challenger is outside the 0-1 range.
        PegaMLopsError
            If there's an error when adding the challenger model.
        """
        objective = None
        # if self.challenger_model:
        #     raise ValueError("Challenger model already exists. Please delete/promote the existing challenger model before creating a new one.")
        if not (0 <= challenger_response_share <= 1):
            raise ValueError("Percentage must be between 0 and 1.")
        endpoint = f"prweb/api/PredictionStudio/v4/predictions/{self.prediction_id}/component/{self.active_model.component_name}"
        data = {}
        if isinstance(new_model, Model):
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
                {"predictor": key["predictor"], "property": key["property"]}
                for key in predictor_mapping
            ]
            data["overrideMappings"] = override_mappings
        try:
            response = self._post(endpoint, data=data)
            self.cc_id = response["referenceID"]
            champion_response_share = float(1 - challenger_response_share)

            response = self._check_then_update(
                champion_response_share=champion_response_share,
                learn_independently=learn_independently,
            )
        except PegaException as e:
            raise PegaMLopsError("Error when Adding challenger model: " + str(e)) from e

        if "Approved" not in response["message"]:
            raise PegaMLopsError("Error when adding model")
        logging.info("Add model: Refreshing Champion challenger configuration: ")
        time.sleep(1)
        self._refresh_champion_challenger()
        logging.info("Add model: %s", response)

    @validate_call
    def clone_model(
        self,
        challenger_response_share: float,
        adm_model_type: Union[AdmModelType, str],
        model_label: Optional[str] = None,
        predictor_mapping: Union[List[Dict], None] = None,
        learn_independently: Union[bool, None] = True,
    ):
        """
        Clones the current active model to create a challenger with specific settings.

        This function duplicates the active model, setting it as a challenger in the prediction setup.
        It allows choosing the model type, adjusting traffic share, and customizing labels and predictor mappings.

        Parameters
        ----------
        challenger_response_share : float
            Defines the traffic percentage for the challenger model.
        adm_model_type : {'Gradient boosting', 'Naive bayes'}
            Specifies the type of the cloned model.
        model_label : str, optional
            A custom label for the cloned model. Defaults to a generated unique label if not provided.
        predictor_mapping : list of dict, optional
            Custom mappings of predictors to properties for the cloned model.
        learn_independently: bool, optional
            If True, the challenger model will learn independently. Defaults to True.

        Raises
        ------
        PegaMLopsError
            If the challenger_response_percentage is not within the 0-1 range or adm_model_type is invalid.
              Or if there's an error when adding the challenger model.
        """
        if isinstance(adm_model_type, AdmModelType):
            adm_model_type = adm_model_type.value
        valid_adm_model_types = [
            AdmModelType.GRADIENT_BOOSTING.value,
            AdmModelType.NAIVE_BAYES.value,
        ]
        if adm_model_type not in valid_adm_model_types:
            raise PegaMLopsError(
                "Invalid adm model type. It should be 'Gradient boosting' or 'Naive bayes'"
            )
        if not (0 <= challenger_response_share <= 1):
            raise PegaMLopsError("Percentage must be between 0 and 1.")
        endpoint = f"prweb/api/PredictionStudio/v4/predictions/{self.prediction_id}/component/{self.active_model.component_name}/clone"
        if model_label is None:
            unique_suffix = "".join(random.choices(string.ascii_uppercase, k=3))
            model_label = self.active_model.component_name + "_copy_" + unique_suffix

        data = {
            "modelLabel": model_label,
            "admModelType": adm_model_type,
        }
        if predictor_mapping is not None:
            data["overrideMmappings"] = [
                {"predictor": key["predictor"], "property": key["property"]}
                for key in predictor_mapping
            ]
        try:
            response = self._post(endpoint, data=data)
            self.cc_id = response["referenceID"]
            champion_response_percentage = float(1 - challenger_response_share)
            response = self._check_then_update(
                champion_response_share=champion_response_percentage,
                learn_independently=learn_independently,
            )
        except PegaException as e:
            raise PegaMLopsError("Error when Adding challenger model: " + str(e)) from e
        if "Approved" not in response["message"]:
            raise PegaMLopsError("Error when adding model")
        self._refresh_champion_challenger()
        logging.info("Clone model: %s", response)

    def add_predictor(
        self,
        name: str,
        predictor_type: str,
        value: str,
        data_type: str,
        is_active_model: bool,
        parameterized: bool = True,
    ):
        """
        Adds a new predictor to a specific model in a prediction setup.

        This function introduces a new predictor to a model tied to a prediction.
        It differentiates between parameterized and static predictors based on a flag.

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
            Indicates if the predictor should be added to the active model or the challenger model.
        parameterized : bool, optional
            Indicates if the predictor is parameterized, default is True.

        Returns
        -------
        Response
            Outcome of the addition operation as received from the API.

        Raises
        ------
        PegaMLopsError
            If the challenger model is not set or if an error occurs when adding the predictor.
        """
        if is_active_model:
            model = self.active_model
        else:
            if not self.challenger_model:
                raise PegaMLopsError("Challenger model is not set.")
            model = self.challenger_model
        endpoint = f"/prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/models/{model.model_id}/predictor/add"
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
        else:
            if data_type in ["Text", "TrueFalse"]:
                data["predictorType"] = "symbolic"
            else:
                data["predictorType"] = "numeric"

        if value:
            data["paramPredictorValue"] = value
        try:
            response = self._client.patch(endpoint, data=data)
        except PegaException as e:
            raise PegaMLopsError("Error when Adding predictor: " + str(e)) from e
        return response

    def remove_predictor(
        self,
        name: str,
        parameterized: Optional[bool] = False,
        is_active_model: bool = True,
    ):
        """
        Removes a predictor from a model in a prediction setup.

        This function deletes a predictor, identified by its name, from a model linked to a prediction.

        Parameters
        ----------
        name : str
            The name of the predictor to remove.
        parameterized : bool, optional
            True if the predictor is parameterized, False if it's static. Defaults to False.

        Returns
        -------
        Response
            The result of the deletion request.

        Raises
        ------
        PegaMLopsError
            If the challenger model is not set or if an error occurs when removing the predictor.
        """
        if is_active_model:
            model = self.active_model
        else:
            if not self.challenger_model:
                raise PegaMLopsError("Challenger model is not set.")
            model = self.challenger_model

        endpoint = f"/prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/models/{model.model_id}/predictor/remove"
        if parameterized:
            predictorCategory = "parameterized"
        else:
            raise NotImplementedError()
        data = {
            "predictorName": name,
            "predictorCategory": predictorCategory,
            "contextName": self.context,
        }
        try:
            return self._client.patch(endpoint, data=data)
        except PegaException as e:
            raise PegaMLopsError("Error when removing predictor: " + str(e)) from e
