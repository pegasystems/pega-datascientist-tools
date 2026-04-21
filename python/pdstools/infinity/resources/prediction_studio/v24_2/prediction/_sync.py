from __future__ import annotations

from typing import Literal, overload

import polars as pl

from .....internal._exceptions import PegaException, PegaMLopsError
from .....internal._pagination import PaginatedList
from ...base import Notification
from ...types import NotificationCategory
from ...v24_1.prediction import Prediction as PredictionPrevious
from ._mixin import _PredictionV24_2Mixin


class Prediction(_PredictionV24_2Mixin, PredictionPrevious):
    """The ``Prediction`` class provide functionality including retrieving
    notifications, models, adding conditional models, getting champion
    challengers, metrics, and plotting metrics.
    """

    @overload
    def get_notifications(
        self,
        category: NotificationCategory | None = None,
        return_df: Literal[False] = False,
    ) -> PaginatedList[Notification]: ...

    @overload
    def get_notifications(
        self,
        category: NotificationCategory | None = None,
        return_df: Literal[True] = True,
    ) -> pl.DataFrame: ...

    def get_notifications(
        self,
        category: NotificationCategory | None = None,
        return_df: bool = False,
    ) -> PaginatedList[Notification] | pl.DataFrame:
        """Fetches a list of notifications for a specific prediction.

        This function retrieves notifications related to a prediction. You
        can filter these notifications by their category.  Optionally, the
        notifications can be returned as a DataFrame for easier analysis
        and visualization.

        Parameters
        ----------
        category : {"All", "Responses", "Performance", "Model approval", "Output", "Predictors", "Prediction deployment", "Generic"} or None
            The category of notifications to retrieve. If not specified,
            all notifications are fetched.
        return_df : bool, default False
            If True, returns the notifications as a DataFrame. Otherwise,
            returns a list.

        Returns
        -------
        PaginatedList[Notification] or polars.DataFrame
            A list of notifications or a DataFrame containing the
            notifications, depending on the value of ``return_df``.

        """
        endpoint = f"prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/notifications"
        if category is None:
            category = "All"
        endpoint = f"{endpoint}?category={category}"

        notifications: PaginatedList[Notification] = PaginatedList(
            Notification,
            self._client,
            "get",
            endpoint,
            _root="notifications",
        )
        if return_df:
            return pl.DataFrame(
                [notification._public_dict for notification in notifications],
            )
        return notifications

    def get_champion_challengers(self):
        """Fetches list of ChampionChallenger objects linked to the prediction.

        This function fetches Champion-challenger pairs from a prediction.
        In cases where a challenger model is absent, it returns a
        ChampionChallenger object containing only the champion model.

        Returns
        -------
        list of ChampionChallenger
            A list of entries, each pairing a primary model with its
            challenger across various segments of the prediction.

        """
        from ..champion_challenger import ChampionChallenger
        from ..model import Model

        ccs = []
        # _get_models is a plain async helper — on the sync path the
        # __init_subclass__ wrapper for get_metric (etc.) runs it inside
        # _run_sync.  Here we call it via _run_sync ourselves since
        # get_champion_challengers is NOT an @api_method.
        from .....internal._resource import _run_sync

        models = _run_sync(self._get_models)
        non_active = [mod for mod in models if mod["role"] not in {"ACTIVE", "CHAMPION"}]

        only_active = [
            mod for mod in models if all(key not in mod for key in ["championPercentage", "challengerPercentage"])
        ]

        for model in non_active:
            active_model_temp = {
                "modelId": model["id"],
                "label": model["label"],
                "modelType": model["type"],
                "status": model["role"],
                "componentName": model["componentName"],
                "modelingTechnique": model["modelingTechnique"],
            }
            ccs.append(
                ChampionChallenger(
                    client=self._client,
                    prediction_id=self.prediction_id,
                    champion_percentage=100 - model["challengerPercentage"],
                    challenger_model=Model(client=self._client, **active_model_temp),
                    context=model["contextName"],
                    category=model["categoryName"] if model.get("categoryName") is not None else None,
                    model_objective=model["model_type"],
                    active_model=[
                        Model(
                            client=self._client,
                            modelId=mod["id"],
                            label=mod["label"],
                            modelType=mod["type"],
                            status=mod["role"],
                            componentName=mod["componentName"],
                            modelingTechnique=mod["modelingTechnique"]
                            if mod.get("modelingTechnique") is not None
                            else None,
                        )
                        for mod in models
                        if model["activeModel"] is not None
                        and mod["id"] == model["activeModel"]
                        and mod["contextName"] == model["contextName"]
                    ][0],
                ),
            )

        for model in only_active:
            active_model_temp = {
                "modelId": model["id"],
                "label": model["label"],
                "modelType": model["type"],
                "status": model["role"],
                "componentName": model["componentName"],
                "modelingTechnique": model["modelingTechnique"] if model.get("modelingTechnique") is not None else None,
            }
            ccs.append(
                ChampionChallenger(
                    client=self._client,
                    prediction_id=self.prediction_id,
                    context=model["contextName"],
                    model_objective=model["model_type"],
                    category=model["categoryName"] if model.get("categoryName") is not None else None,
                    challenger_model=None,
                    active_model=Model(client=self._client, **active_model_temp),
                ),
            )

        return ccs

    def add_conditional_model(
        self,
        new_model,
        category: str,
        context: str | None = None,
    ):
        """Incorporates a new model into a prediction for a specified category
        and context.

        Parameters
        ----------
        new_model : str or Model
            Identifier of the model to be added.
        category : str
            The category under which the model will be classified.
        context : str, optional
            The specific context or scenario in which the model will be
            utilized.

        Returns
        -------
        ChampionChallenger
            An object detailing the updated configuration with the newly
            added model.

        """
        from ..model import Model

        if isinstance(new_model, Model):
            new_model = new_model.model_id
        if context is None:
            context = "NoContext"
        endpoint = (
            f"prweb/api/PredictionStudio/v4/predictions/{self.prediction_id}/category/{category}/models/{new_model}"
        )
        data = {}
        if context:
            data["contextName"] = context
        try:
            response = self._post(endpoint, data=data)
        except PegaException as e:
            raise PegaMLopsError(
                "Error when adding Conditional model: " + str(e),
            ) from e
        if not response:
            raise PegaMLopsError("Add conditional model failed")
        champion_challengers = self.get_champion_challengers()
        for cc in champion_challengers:
            if cc.category is not None:
                if (
                    cc.active_model.model_id.lower() == new_model.lower()
                    and cc.context.lower() == context.lower()
                    and cc.category.lower() == category.lower()
                ):
                    return cc
