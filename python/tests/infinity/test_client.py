"""Tests for Infinity and AsyncInfinity client classes."""

from __future__ import annotations

import pytest
from pdstools.infinity import AsyncInfinity, Infinity

# ---------------------------------------------------------------------------
# Infinity (sync)
# ---------------------------------------------------------------------------


class TestInfinity:
    def test_init_without_version(self):
        """Original test: client without version has no prediction_studio."""
        client = Infinity.from_client_id_and_secret("TEST_URL", "NA", "NA")
        assert not client.version
        assert not hasattr(client, "prediction_studio")

    def test_init_no_args_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="constructor methods"):
            Infinity()

    def test_init_with_explicit_version(self, mocker):
        """When pega_version is passed, _infer_version should NOT be called."""
        mocker.patch.object(
            Infinity,
            "_infer_version",
            side_effect=AssertionError("should not be called"),
        )
        client = Infinity.from_client_id_and_secret(
            "https://example.com",
            "id",
            "secret",
            pega_version="24.2",
        )
        assert client.version == "24.2"
        assert hasattr(client, "prediction_studio")

    def test_init_inferred_version_24_1(self, mocker):
        mocker.patch.object(Infinity, "_infer_version", return_value="24.1")
        client = Infinity.from_client_id_and_secret(
            "https://example.com",
            "id",
            "secret",
        )
        assert client.version == "24.1"
        assert hasattr(client, "prediction_studio")

    def test_init_inferred_version_24_2(self, mocker):
        mocker.patch.object(Infinity, "_infer_version", return_value="24.2")
        client = Infinity.from_client_id_and_secret(
            "https://example.com",
            "id",
            "secret",
        )
        assert client.version == "24.2"
        assert hasattr(client, "prediction_studio")

    def test_knowledge_buddy_always_available(self, mocker):
        mocker.patch.object(Infinity, "_infer_version", return_value=None)
        client = Infinity.from_client_id_and_secret(
            "https://example.com",
            "id",
            "secret",
        )
        assert hasattr(client, "knowledge_buddy")

    def test_getattr_version_dependent_resource_raises(self):
        """Accessing prediction_studio when version is None raises helpful error."""
        client = Infinity.from_client_id_and_secret("TEST_URL", "NA", "NA")
        assert client.version is None
        with pytest.raises(AttributeError, match="not available"):
            _ = client.prediction_studio

    def test_getattr_unknown_attribute_raises(self):
        client = Infinity.from_client_id_and_secret("TEST_URL", "NA", "NA")
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = client.nonexistent_thing

    def test_version_dispatch_fallback_for_unknown_version(self, mocker):
        """An unknown version (e.g. '25.1') should fall back to latest (24.2)."""
        mocker.patch.object(Infinity, "_infer_version", return_value="25.1")
        client = Infinity.from_client_id_and_secret(
            "https://example.com",
            "id",
            "secret",
        )
        assert client.version == "25.1"
        # Should still get prediction_studio (falls back to 24.2 dispatch)
        assert hasattr(client, "prediction_studio")


# ---------------------------------------------------------------------------
# AsyncInfinity
# ---------------------------------------------------------------------------


class TestAsyncInfinity:
    def test_init_no_args_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="constructor methods"):
            AsyncInfinity()

    def test_init_with_explicit_version(self, mocker):
        mocker.patch.object(
            AsyncInfinity,
            "_infer_version",
            side_effect=AssertionError("should not be called"),
        )
        client = AsyncInfinity.from_client_id_and_secret(
            "https://example.com",
            "id",
            "secret",
            pega_version="24.2",
        )
        assert client.version == "24.2"
        assert hasattr(client, "prediction_studio")

    def test_init_without_version(self, mocker):
        mocker.patch.object(AsyncInfinity, "_infer_version", return_value=None)
        client = AsyncInfinity.from_client_id_and_secret(
            "https://example.com",
            "id",
            "secret",
        )
        assert client.version is None
        assert not hasattr(client, "prediction_studio")

    def test_init_inferred_version_24_2(self, mocker):
        mocker.patch.object(AsyncInfinity, "_infer_version", return_value="24.2")
        client = AsyncInfinity.from_client_id_and_secret(
            "https://example.com",
            "id",
            "secret",
        )
        assert client.version == "24.2"
        assert hasattr(client, "prediction_studio")

    def test_knowledge_buddy_always_available(self, mocker):
        mocker.patch.object(AsyncInfinity, "_infer_version", return_value=None)
        client = AsyncInfinity.from_client_id_and_secret(
            "https://example.com",
            "id",
            "secret",
        )
        assert hasattr(client, "knowledge_buddy")

    def test_getattr_version_dependent_resource_raises(self, mocker):
        mocker.patch.object(AsyncInfinity, "_infer_version", return_value=None)
        client = AsyncInfinity.from_client_id_and_secret(
            "https://example.com",
            "id",
            "secret",
        )
        with pytest.raises(AttributeError, match="not available"):
            _ = client.prediction_studio

    def test_getattr_unknown_attribute_raises(self, mocker):
        mocker.patch.object(AsyncInfinity, "_infer_version", return_value=None)
        client = AsyncInfinity.from_client_id_and_secret(
            "https://example.com",
            "id",
            "secret",
        )
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = client.nonexistent_thing

    def test_getattr_error_message_mentions_async(self, mocker):
        mocker.patch.object(AsyncInfinity, "_infer_version", return_value=None)
        client = AsyncInfinity.from_client_id_and_secret(
            "https://example.com",
            "id",
            "secret",
        )
        with pytest.raises(AttributeError, match="AsyncInfinity"):
            _ = client.prediction_studio


# ---------------------------------------------------------------------------
# Version dispatch (get / get_async)
# ---------------------------------------------------------------------------


class TestVersionDispatch:
    def test_get_24_1(self):
        from pdstools.infinity.resources.prediction_studio import get
        from pdstools.infinity.resources.prediction_studio.v24_1 import PredictionStudio

        assert get("24.1") is PredictionStudio

    def test_get_24_2(self):
        from pdstools.infinity.resources.prediction_studio import get
        from pdstools.infinity.resources.prediction_studio.v24_2 import PredictionStudio

        assert get("24.2") is PredictionStudio

    def test_get_none_returns_none(self):
        from pdstools.infinity.resources.prediction_studio import get

        assert get("") is None

    def test_get_unknown_falls_back(self):
        from pdstools.infinity.resources.prediction_studio import get
        from pdstools.infinity.resources.prediction_studio.v24_2 import PredictionStudio

        result = get("25.1")
        assert result is PredictionStudio

    def test_get_async_24_1(self):
        from pdstools.infinity.resources.prediction_studio import get_async
        from pdstools.infinity.resources.prediction_studio.v24_1 import (
            AsyncPredictionStudio,
        )

        assert get_async("24.1") is AsyncPredictionStudio

    def test_get_async_24_2(self):
        from pdstools.infinity.resources.prediction_studio import get_async
        from pdstools.infinity.resources.prediction_studio.v24_2 import (
            AsyncPredictionStudio,
        )

        assert get_async("24.2") is AsyncPredictionStudio

    def test_get_async_none_returns_none(self):
        from pdstools.infinity.resources.prediction_studio import get_async

        assert get_async("") is None

    def test_get_async_unknown_falls_back(self):
        from pdstools.infinity.resources.prediction_studio import get_async
        from pdstools.infinity.resources.prediction_studio.v24_2 import (
            AsyncPredictionStudio,
        )

        result = get_async("25.1")
        assert result is AsyncPredictionStudio
