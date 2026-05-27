"""Tests for mutating ChampionChallenger and Prediction APIs.

Covers add_conditional_model, delete_challenger_model, and promote_challenger_model
across all three API versions (v24_2, v25, v26).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pdstools.infinity.internal._exceptions import PegaException, PegaMLopsError
from pdstools.infinity.resources.prediction_studio.v24_2.champion_challenger._sync import (
    ChampionChallenger as CCv24_2,
)
from pdstools.infinity.resources.prediction_studio.v24_2.model import Model as Modelv24_2
from pdstools.infinity.resources.prediction_studio.v24_2.prediction._sync import (
    Prediction as Predv24_2,
)
from pdstools.infinity.resources.prediction_studio.v25.champion_challenger._sync import (
    ChampionChallenger as CCv25,
)
from pdstools.infinity.resources.prediction_studio.v25.model import Model as Modelv25
from pdstools.infinity.resources.prediction_studio.v25.prediction._sync import (
    Prediction as Predv25,
)
from pdstools.infinity.resources.prediction_studio.v26.champion_challenger._sync import (
    ChampionChallenger as CCv26,
)
from pdstools.infinity.resources.prediction_studio.v26.model import Model as Modelv26
from pdstools.infinity.resources.prediction_studio.v26.prediction._sync import (
    Prediction as Predv26,
)

# ---------------------------------------------------------------------------
# Parametrize triples: (PredictionClass, CCClass, ModelClass)
# ---------------------------------------------------------------------------

ALL_VERSIONS = pytest.mark.parametrize(
    "PredClass,CCClass,ModelClass",
    [
        pytest.param(Predv24_2, CCv24_2, Modelv24_2, id="v24_2"),
        pytest.param(Predv25, CCv25, Modelv25, id="v25"),
        pytest.param(Predv26, CCv26, Modelv26, id="v26"),
    ],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client():
    client = MagicMock()
    client.get = MagicMock(return_value={"status": "ok"})
    client.post = MagicMock(return_value={"status": "ok"})
    client.patch = MagicMock(return_value={"status": "ok"})
    return client


def _make_pega_exc(msg: str = "error"):
    """Construct a PegaException without requiring a real httpx.Response."""
    exc = PegaException.__new__(PegaException)
    Exception.__init__(exc, msg)
    exc.override_message = msg
    return exc


def _make_model(ModelClass, client, model_id: str = "model-001", component_name: str = "Comp"):
    return ModelClass(
        client=client,
        modelId=model_id,
        label="Test Model",
        modelType="Adaptive model",
        status="Completed",
        componentName=component_name,
    )


def _make_prediction(PredClass, client):
    return PredClass(
        client=client,
        predictionId="PRED-001",
        label="Test Prediction",
        status="Completed",
    )


def _make_cc(CCClass, ModelClass, client, *, challenger_model=None, context="NoContext", category="Retention"):
    active = _make_model(ModelClass, client, "active-001", "ActiveComp")
    return CCClass(
        client=client,
        prediction_id="PRED-001",
        active_model=active,
        challenger_model=challenger_model,
        context=context,
        category=category,
    )


# ---------------------------------------------------------------------------
# add_conditional_model
# ---------------------------------------------------------------------------


class TestAddConditionalModel:
    @ALL_VERSIONS
    def test_returns_matching_cc_on_success(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        prediction = _make_prediction(PredClass, client)

        matching_cc = _make_cc(CCClass, ModelClass, client, context="NoContext", category="Retention")
        matching_cc.active_model = _make_model(ModelClass, client, "new-model-001")

        with patch.object(prediction, "get_champion_challengers", return_value=[matching_cc]):
            result = prediction.add_conditional_model("new-model-001", category="Retention")

        assert result is matching_cc

    @ALL_VERSIONS
    def test_raises_value_error_when_no_cc_matches(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        prediction = _make_prediction(PredClass, client)

        non_matching_cc = _make_cc(CCClass, ModelClass, client, context="NoContext", category="Retention")
        non_matching_cc.active_model = _make_model(ModelClass, client, "other-model")

        with patch.object(prediction, "get_champion_challengers", return_value=[non_matching_cc]):
            with pytest.raises(ValueError, match="could not be found"):
                prediction.add_conditional_model("new-model-001", category="Retention")

    @ALL_VERSIONS
    def test_raises_mlops_error_on_api_exception(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        client.post.side_effect = _make_pega_exc("server error")
        prediction = _make_prediction(PredClass, client)

        with pytest.raises(PegaMLopsError, match="adding Conditional model"):
            prediction.add_conditional_model("new-model-001", category="Retention")

    @ALL_VERSIONS
    def test_raises_mlops_error_when_response_is_falsy(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        client.post.return_value = None
        prediction = _make_prediction(PredClass, client)

        with pytest.raises(PegaMLopsError, match="failed"):
            prediction.add_conditional_model("new-model-001", category="Retention")

    @ALL_VERSIONS
    def test_accepts_model_object_as_new_model(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        prediction = _make_prediction(PredClass, client)
        model_obj = _make_model(ModelClass, client, "new-model-001")

        matching_cc = _make_cc(CCClass, ModelClass, client, context="NoContext", category="Retention")
        matching_cc.active_model = _make_model(ModelClass, client, "new-model-001")

        with patch.object(prediction, "get_champion_challengers", return_value=[matching_cc]):
            result = prediction.add_conditional_model(model_obj, category="Retention")

        assert result is matching_cc

    @ALL_VERSIONS
    def test_context_none_defaults_to_nocontext(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        prediction = _make_prediction(PredClass, client)

        matching_cc = _make_cc(CCClass, ModelClass, client, context="NoContext", category="Cat")
        matching_cc.active_model = _make_model(ModelClass, client, "model-x")

        with patch.object(prediction, "get_champion_challengers", return_value=[matching_cc]):
            result = prediction.add_conditional_model("model-x", category="Cat", context=None)

        assert result is matching_cc

    @ALL_VERSIONS
    def test_category_encoded_in_url(self, PredClass, CCClass, ModelClass):
        """category with a slash must be percent-encoded in the request URL."""
        client = _make_client()
        client.post.side_effect = _make_pega_exc("stop early")
        prediction = _make_prediction(PredClass, client)

        with pytest.raises(PegaMLopsError):
            prediction.add_conditional_model("model-x", category="Ret/ention")

        called_url = client.post.call_args[0][0]
        assert "Ret%2Fention" in called_url
        assert "Ret/ention" not in called_url


# ---------------------------------------------------------------------------
# delete_challenger_model
# ---------------------------------------------------------------------------


class TestDeleteChallengerModel:
    @ALL_VERSIONS
    def test_raises_when_no_challenger_set(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=None)

        with pytest.raises(PegaMLopsError, match="Challenger model is not set"):
            cc.delete_challenger_model()

    @ALL_VERSIONS
    def test_happy_path_calls_patch_with_correct_url(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        challenger = _make_model(ModelClass, client, "chall-001")
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=challenger)

        with patch.object(CCClass, "_refresh_champion_challenger", new=AsyncMock()):
            cc.delete_challenger_model()

        assert client.patch.called
        called_url = client.patch.call_args[0][0]
        assert "chall-001" in called_url
        assert "Remove" in called_url

    @ALL_VERSIONS
    def test_api_error_wrapped_as_mlops_error(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        client.patch.side_effect = _make_pega_exc("network failure")
        challenger = _make_model(ModelClass, client, "chall-001")
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=challenger)

        with patch.object(CCClass, "_refresh_champion_challenger", new=AsyncMock()):
            with pytest.raises(PegaMLopsError, match="deleting challenger"):
                cc.delete_challenger_model()


# ---------------------------------------------------------------------------
# promote_challenger_model
# ---------------------------------------------------------------------------


class TestPromoteChallengerModel:
    @ALL_VERSIONS
    def test_raises_when_no_challenger_set(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=None)

        with pytest.raises(PegaMLopsError, match="Challenger model is not set"):
            cc.promote_challenger_model()

    @ALL_VERSIONS
    def test_happy_path_calls_patch_with_correct_url(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        challenger = _make_model(ModelClass, client, "chall-001")
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=challenger)

        with patch.object(CCClass, "_refresh_champion_challenger", new=AsyncMock()):
            cc.promote_challenger_model()

        assert client.patch.called
        called_url = client.patch.call_args[0][0]
        assert "chall-001" in called_url
        assert "Promote" in called_url

    @ALL_VERSIONS
    def test_api_error_wrapped_as_mlops_error(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        client.patch.side_effect = _make_pega_exc("timeout")
        challenger = _make_model(ModelClass, client, "chall-001")
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=challenger)

        with patch.object(CCClass, "_refresh_champion_challenger", new=AsyncMock()):
            with pytest.raises(PegaMLopsError, match="promoting challenger"):
                cc.promote_challenger_model()

