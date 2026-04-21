from __future__ import annotations

import polars as pl

from .....internal._exceptions import PegaException, PegaMLopsError
from .....internal._pagination import AsyncPaginatedList
from ...base import AsyncNotification
from ...types import NotificationCategory
from ...v24_1.prediction import AsyncPrediction as AsyncPredictionPrevious
from ._mixin import _PredictionV24_2Mixin


class AsyncPrediction(_PredictionV24_2Mixin, AsyncPredictionPrevious):
    """Async variant of the v24.2 Prediction."""

    async def get_notifications(
        self,
        category: NotificationCategory | None = None,
        return_df: bool = False,
    ) -> AsyncPaginatedList[AsyncNotification] | pl.DataFrame:
        """Fetches a list of notifications for a specific prediction.

        Parameters
        ----------
        category : {"All", "Responses", "Performance", "Model approval", "Output", "Predictors", "Prediction deployment", "Generic"} or None
            The category of notifications to retrieve.
        return_df : bool, default False
            If True, returns the notifications as a DataFrame.

        Returns
        -------
        AsyncPaginatedList[AsyncNotification] or polars.DataFrame
            A list of notifications or a DataFrame.

        """
        endpoint = f"prweb/api/PredictionStudio/v1/predictions/{self.prediction_id}/notifications"
        if category is None:
            category = "All"
        endpoint = f"{endpoint}?category={category}"

        notifications: AsyncPaginatedList[AsyncNotification] = AsyncPaginatedList(
            AsyncNotification,
            self._client,
            "get",
            endpoint,
            _root="notifications",
        )
        if return_df:
            return await notifications.as_df()
        return notifications

    async def get_champion_challengers(self):
        """Fetches list of ChampionChallenger objects linked to the prediction.

        Returns
        -------
        list of AsyncChampionChallenger
            Champion-challenger pairs from a prediction.

        """
        from ..champion_challenger import AsyncChampionChallenger
        from ..model import AsyncModel

        ccs = []
        models = await self._get_models()
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
                AsyncChampionChallenger(
                    client=self._client,
                    prediction_id=self.prediction_id,
                    champion_percentage=100 - model["challengerPercentage"],
                    challenger_model=AsyncModel(
                        client=self._client,
                        **active_model_temp,
                    ),
                    context=model["contextName"],
                    category=model["categoryName"] if model.get("categoryName") is not None else None,
                    model_objective=model["model_type"],
                    active_model=[
                        AsyncModel(
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
                AsyncChampionChallenger(
                    client=self._client,
                    prediction_id=self.prediction_id,
                    context=model["contextName"],
                    model_objective=model["model_type"],
                    category=model["categoryName"] if model.get("categoryName") is not None else None,
                    challenger_model=None,
                    active_model=AsyncModel(client=self._client, **active_model_temp),
                ),
            )

        return ccs

    async def add_conditional_model(
        self,
        new_model,
        category: str,
        context: str | None = None,
    ):
        """Incorporates a new model into a prediction for a specified category
        and context.

        Parameters
        ----------
        new_model : str or AsyncModel
            Identifier of the model to be added.
        category : str
            The category under which the model will be classified.
        context : str, optional
            The specific context or scenario.

        Returns
        -------
        AsyncChampionChallenger
            An object detailing the updated configuration.

        """
        from ..model import AsyncModel

        if isinstance(new_model, AsyncModel):
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
            response = await self._a_post(endpoint, data=data)
        except PegaException as e:
            raise PegaMLopsError(
                "Error when adding Conditional model: " + str(e),
            ) from e
        if not response:
            raise PegaMLopsError("Add conditional model failed")
        champion_challengers = await self.get_champion_challengers()
        for cc in champion_challengers:
            if cc.category is not None:
                if (
                    cc.active_model.model_id.lower() == new_model.lower()
                    and cc.context.lower() == context.lower()
                    and cc.category.lower() == category.lower()
                ):
                    return cc
