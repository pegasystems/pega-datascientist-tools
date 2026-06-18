"""Tests for mutating ChampionChallenger and Prediction APIs.

Covers add_conditional_model, delete_challenger_model, and promote_challenger_model
across all three API versions (v24_2, v25/v26).
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
from pdstools.infinity.resources.prediction_studio.v26_1.champion_challenger._sync import (
    ChampionChallenger as CCv26_1,
)
from pdstools.infinity.resources.prediction_studio.v26_1.model import Model as Modelv26_1
from pdstools.infinity.resources.prediction_studio.v26_1.prediction._sync import (
    Prediction as Predv26_1,
)

# v25 and v26 share one implementation — alias for parametrized test readability
CCv25_1 = CCv26_1
Modelv25_1 = Modelv26_1
Predv25_1 = Predv26_1

# ---------------------------------------------------------------------------
# Parametrize triples: (PredictionClass, CCClass, ModelClass)
# ---------------------------------------------------------------------------

ALL_VERSIONS = pytest.mark.parametrize(
    "PredClass,CCClass,ModelClass",
    [
        pytest.param(Predv24_2, CCv24_2, Modelv24_2, id="v24_2"),
        pytest.param(Predv25_1, CCv25_1, Modelv25_1, id="v25.1"),
        pytest.param(Predv26_1, CCv26_1, Modelv26_1, id="v26.1"),
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


def _make_model(
    ModelClass, client, model_id: str = "model-001", component_name: str = "Comp", status: str = "Completed"
):
    return ModelClass(
        client=client,
        modelId=model_id,
        label="Test Model",
        modelType="Adaptive model",
        status=status,
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


# ---------------------------------------------------------------------------
# update_challenger_response_share
# ---------------------------------------------------------------------------


class TestUpdateChallengerResponseShare:
    @ALL_VERSIONS
    def test_shadow_model_calls_update_pattern_endpoint(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        challenger = _make_model(ModelClass, client, "chall-001", status="Shadow")
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=challenger)

        with patch.object(CCClass, "_refresh_champion_challenger", new=AsyncMock()):
            cc.update_challenger_response_share(0.3)

        called_url = client.patch.call_args[0][0]
        assert "updatePattern" in called_url
        assert "chall-001" in called_url

    @ALL_VERSIONS
    def test_non_shadow_model_calls_distribution_endpoint(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        challenger = _make_model(ModelClass, client, "chall-001")
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=challenger)

        with patch.object(CCClass, "_refresh_champion_challenger", new=AsyncMock()):
            cc.update_challenger_response_share(0.3)

        called_url = client.patch.call_args[0][0]
        assert "distribution" in called_url
        assert "active-001" in called_url

    @ALL_VERSIONS
    def test_percentage_below_zero_raises_value_error(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        challenger = _make_model(ModelClass, client, "chall-001")
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=challenger)

        with pytest.raises(ValueError, match="0 and 1"):
            cc.update_challenger_response_share(-0.1)

    @ALL_VERSIONS
    def test_percentage_above_one_raises_value_error(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        challenger = _make_model(ModelClass, client, "chall-001")
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=challenger)

        with pytest.raises(ValueError, match="0 and 1"):
            cc.update_challenger_response_share(1.1)

    @ALL_VERSIONS
    def test_no_challenger_model_raises_value_error(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=None)

        with pytest.raises(ValueError, match="Challenger model is not set"):
            cc.update_challenger_response_share(0.3)

    @ALL_VERSIONS
    def test_api_error_wrapped_as_mlops_error(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        client.patch.side_effect = _make_pega_exc("network failure")
        challenger = _make_model(ModelClass, client, "chall-001")
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=challenger)

        with patch.object(CCClass, "_refresh_champion_challenger", new=AsyncMock()):
            with pytest.raises(PegaMLopsError, match="updating challenger"):
                cc.update_challenger_response_share(0.3)


# ---------------------------------------------------------------------------
# add_predictor
# ---------------------------------------------------------------------------


class TestAddPredictor:
    @ALL_VERSIONS
    def test_active_model_happy_path(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        cc = _make_cc(CCClass, ModelClass, client)

        cc.add_predictor(
            name="Income",
            predictor_type="numeric",
            value="",
            data_type="Decimal",
            is_active_model=True,
        )

        called_url = client.patch.call_args[0][0]
        assert "active-001" in called_url
        assert "predictor/add" in called_url

    @ALL_VERSIONS
    def test_challenger_model_happy_path(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        challenger = _make_model(ModelClass, client, "chall-001")
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=challenger)

        cc.add_predictor(
            name="Income",
            predictor_type="numeric",
            value="",
            data_type="Decimal",
            is_active_model=False,
        )

        called_url = client.patch.call_args[0][0]
        assert "chall-001" in called_url
        assert "predictor/add" in called_url

    @ALL_VERSIONS
    def test_no_challenger_model_raises_mlops_error(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=None)

        with pytest.raises(PegaMLopsError, match="Challenger model is not set"):
            cc.add_predictor(
                name="Income",
                predictor_type="numeric",
                value="",
                data_type="Decimal",
                is_active_model=False,
            )

    @ALL_VERSIONS
    def test_static_predictor_not_implemented(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        cc = _make_cc(CCClass, ModelClass, client)

        with pytest.raises(NotImplementedError):
            cc.add_predictor(
                name="Income",
                predictor_type="numeric",
                value="",
                data_type="Decimal",
                is_active_model=True,
                parameterized=False,
            )

    @ALL_VERSIONS
    def test_api_error_wrapped_as_mlops_error(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        client.patch.side_effect = _make_pega_exc("server error")
        cc = _make_cc(CCClass, ModelClass, client)

        with pytest.raises(PegaMLopsError, match="Adding predictor"):
            cc.add_predictor(
                name="Income",
                predictor_type="numeric",
                value="",
                data_type="Decimal",
                is_active_model=True,
            )


# ---------------------------------------------------------------------------
# remove_predictor
# ---------------------------------------------------------------------------


class TestRemovePredictor:
    @ALL_VERSIONS
    def test_active_model_happy_path(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        cc = _make_cc(CCClass, ModelClass, client)

        cc.remove_predictor(name="Income", parameterized=True, is_active_model=True)

        called_url = client.patch.call_args[0][0]
        assert "active-001" in called_url
        assert "predictor/remove" in called_url

    @ALL_VERSIONS
    def test_challenger_model_happy_path(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        challenger = _make_model(ModelClass, client, "chall-001")
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=challenger)

        cc.remove_predictor(name="Income", parameterized=True, is_active_model=False)

        called_url = client.patch.call_args[0][0]
        assert "chall-001" in called_url
        assert "predictor/remove" in called_url

    @ALL_VERSIONS
    def test_no_challenger_model_raises_mlops_error(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        cc = _make_cc(CCClass, ModelClass, client, challenger_model=None)

        with pytest.raises(PegaMLopsError, match="Challenger model is not set"):
            cc.remove_predictor(name="Income", parameterized=True, is_active_model=False)

    @ALL_VERSIONS
    def test_static_predictor_not_implemented(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        cc = _make_cc(CCClass, ModelClass, client)

        with pytest.raises(NotImplementedError):
            cc.remove_predictor(name="Income", parameterized=False, is_active_model=True)

    @ALL_VERSIONS
    def test_api_error_wrapped_as_mlops_error(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        client.patch.side_effect = _make_pega_exc("server error")
        cc = _make_cc(CCClass, ModelClass, client)

        with pytest.raises(PegaMLopsError, match="removing predictor"):
            cc.remove_predictor(name="Income", parameterized=True, is_active_model=True)


# ---------------------------------------------------------------------------
# add_model
# ---------------------------------------------------------------------------


class TestAddModel:
    @ALL_VERSIONS
    def test_percentage_out_of_range_raises_value_error(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        cc = _make_cc(CCClass, ModelClass, client)

        with pytest.raises(ValueError, match="0 and 1"):
            cc.add_model("existing-model-id", challenger_response_share=1.5)

    @ALL_VERSIONS
    def test_post_error_raises_mlops_error(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        client.post.side_effect = _make_pega_exc("post failed")
        cc = _make_cc(CCClass, ModelClass, client)

        with pytest.raises(PegaMLopsError, match="Adding challenger"):
            cc.add_model("existing-model-id", challenger_response_share=0.3)

    @ALL_VERSIONS
    def test_happy_path_with_approved_message(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        client.post.return_value = {"referenceID": "ref-001"}
        cc = _make_cc(CCClass, ModelClass, client)

        with (
            patch.object(
                CCClass, "_check_then_update", new=AsyncMock(return_value={"message": "Approved successfully"})
            ),
            patch.object(CCClass, "_refresh_champion_challenger", new=AsyncMock()),
            patch.object(CCClass, "_sleep", new=AsyncMock()),
        ):
            cc.add_model("existing-model-id", challenger_response_share=0.3)

    def test_bad_approve_message_raises_mlops_error_v24_2(self):
        """Only v24_2 raises PegaMLopsError when the approve message is absent."""
        client = _make_client()
        client.post.return_value = {"referenceID": "ref-001"}
        cc = _make_cc(CCv24_2, Modelv24_2, client)

        with (
            patch.object(CCv24_2, "_check_then_update", new=AsyncMock(return_value={"message": "Failed"})),
            patch.object(CCv24_2, "_refresh_champion_challenger", new=AsyncMock()),
            patch.object(CCv24_2, "_sleep", new=AsyncMock()),
        ):
            with pytest.raises(PegaMLopsError, match="adding model"):
                cc.add_model("existing-model-id", challenger_response_share=0.3)


# ---------------------------------------------------------------------------
# clone_model
# ---------------------------------------------------------------------------


class TestCloneModel:
    @ALL_VERSIONS
    def test_invalid_adm_type_raises_mlops_error(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        cc = _make_cc(CCClass, ModelClass, client)

        with pytest.raises(PegaMLopsError, match="Invalid adm model type"):
            cc.clone_model(challenger_response_share=0.3, adm_model_type="InvalidType")

    @ALL_VERSIONS
    def test_percentage_out_of_range_raises_mlops_error(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        cc = _make_cc(CCClass, ModelClass, client)

        with pytest.raises(PegaMLopsError, match="0 and 1"):
            cc.clone_model(challenger_response_share=1.5, adm_model_type="Gradient_boost")

    @ALL_VERSIONS
    def test_post_error_raises_mlops_error(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        client.post.side_effect = _make_pega_exc("post failed")
        cc = _make_cc(CCClass, ModelClass, client)

        with pytest.raises(PegaMLopsError, match="Adding challenger"):
            cc.clone_model(challenger_response_share=0.3, adm_model_type="Gradient_boost")

    @ALL_VERSIONS
    def test_happy_path_with_approved_message(self, PredClass, CCClass, ModelClass):
        client = _make_client()
        client.post.return_value = {"referenceID": "ref-001"}
        cc = _make_cc(CCClass, ModelClass, client)

        with (
            patch.object(CCClass, "_check_then_update", new=AsyncMock(return_value={"message": "Approved"})),
            patch.object(CCClass, "_refresh_champion_challenger", new=AsyncMock()),
        ):
            cc.clone_model(challenger_response_share=0.3, adm_model_type="Gradient_boost")

    def test_bad_approve_message_raises_mlops_error_v24_2(self):
        """Only v24_2 raises PegaMLopsError when the approve message is absent."""
        client = _make_client()
        client.post.return_value = {"referenceID": "ref-001"}
        cc = _make_cc(CCv24_2, Modelv24_2, client)

        with (
            patch.object(CCv24_2, "_check_then_update", new=AsyncMock(return_value={"message": "Failed"})),
            patch.object(CCv24_2, "_refresh_champion_challenger", new=AsyncMock()),
        ):
            with pytest.raises(PegaMLopsError, match="adding model"):
                cc.clone_model(challenger_response_share=0.3, adm_model_type="Gradient_boost")
