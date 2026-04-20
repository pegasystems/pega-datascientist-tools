"""Tests for BaseClient, SyncAPIClient, and AsyncAPIClient."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest
from pdstools.infinity.internal._base_client import (
    SyncAPIClient,
)
from pdstools.infinity.internal._exceptions import (
    APIConnectionError,
    APITimeoutError,
    MultipleErrors,
    NoMonitoringInfo,
    handle_pega_exception,
)

# ---------------------------------------------------------------------------
# BaseClient
# ---------------------------------------------------------------------------


class TestBaseClient:
    def test_enforce_trailing_slash_adds_slash(self):
        client = SyncAPIClient(
            base_url="https://example.com/api",
            auth=httpx.BasicAuth("user", "pass"),
        )
        assert client._base_url.raw_path.endswith(b"/")

    def test_enforce_trailing_slash_keeps_existing(self):
        client = SyncAPIClient(
            base_url="https://example.com/api/",
            auth=httpx.BasicAuth("user", "pass"),
        )
        assert client._base_url.raw_path.endswith(b"/")
        # Should not have double slash.
        assert not client._base_url.raw_path.endswith(b"//")

    def test_build_request_creates_httpx_request(self):
        client = SyncAPIClient(
            base_url="https://example.com",
            auth=httpx.BasicAuth("user", "pass"),
        )
        request = client._build_request("GET", "api/test", foo="bar")
        assert isinstance(request, httpx.Request)
        assert request.method == "GET"
        assert "foo=bar" in str(request.url)

    def test_build_request_with_data(self):
        client = SyncAPIClient(
            base_url="https://example.com",
            auth=httpx.BasicAuth("user", "pass"),
        )
        request = client._build_request("POST", "api/test", data={"key": "value"})
        assert request.method == "POST"
        body = json.loads(request.content)
        assert body == {"key": "value"}

    def test_get_version_24_1(self):
        client = SyncAPIClient(
            base_url="https://example.com",
            auth=httpx.BasicAuth("user", "pass"),
        )
        repo = {"repository_name": "TestRepo"}
        assert client._get_version(repo) == "24.1"

    def test_get_version_24_2(self):
        client = SyncAPIClient(
            base_url="https://example.com",
            auth=httpx.BasicAuth("user", "pass"),
        )
        repo = {"repository_type": "S3", "repository_name": "TestRepo"}
        assert client._get_version(repo) == "24.2"

    def test_get_version_unknown_warns(self):
        client = SyncAPIClient(
            base_url="https://example.com",
            auth=httpx.BasicAuth("user", "pass"),
        )
        repo = {"unexpected_key": "value"}
        with pytest.warns(UserWarning, match="Could not infer"):
            result = client._get_version(repo)
        assert result is None

    def test_pega_version_stored(self):
        client = SyncAPIClient(
            base_url="https://example.com",
            auth=httpx.BasicAuth("user", "pass"),
            pega_version="24.2",
        )
        assert client.pega_version == "24.2"

    def test_application_name_stored(self):
        client = SyncAPIClient(
            base_url="https://example.com",
            auth=httpx.BasicAuth("user", "pass"),
            application_name="MyApp",
        )
        assert client.application_name == "MyApp"


# ---------------------------------------------------------------------------
# SyncAPIClient — request dispatch
# ---------------------------------------------------------------------------


class TestSyncAPIClientRequest:
    def _make_client(self):
        return SyncAPIClient(
            base_url="https://example.com",
            auth=httpx.BasicAuth("user", "pass"),
        )

    def _mock_response(self, status_code=200, json_data=None):
        response = MagicMock(spec=httpx.Response)
        response.status_code = status_code
        response.json.return_value = json_data or {"ok": True}
        return response

    def test_request_dispatches_get(self, mocker):
        client = self._make_client()
        mocker.patch.object(client, "get", return_value={"data": "get"})
        result = client.request("GET", "/api/test")
        assert result == {"data": "get"}
        client.get.assert_called_once_with(endpoint="/api/test")

    def test_request_dispatches_post(self, mocker):
        client = self._make_client()
        mocker.patch.object(client, "post", return_value={"data": "post"})
        result = client.request("POST", "/api/test", data={"x": 1})
        assert result == {"data": "post"}
        client.post.assert_called_once_with(endpoint="/api/test", data={"x": 1})

    def test_request_dispatches_patch(self, mocker):
        client = self._make_client()
        mocker.patch.object(client, "patch", return_value={"data": "patch"})
        result = client.request("PATCH", "/api/test")
        assert result == {"data": "patch"}

    def test_request_dispatches_put(self, mocker):
        client = self._make_client()
        mocker.patch.object(client, "put", return_value={"data": "put"})
        result = client.request("PUT", "/api/test")
        assert result == {"data": "put"}

    def test_request_dispatches_delete(self, mocker):
        client = self._make_client()
        mocker.patch.object(client, "delete", return_value={"data": "delete"})
        result = client.request("DELETE", "/api/test")
        assert result == {"data": "delete"}

    def test_request_raises_for_unsupported_method(self):
        client = self._make_client()
        with pytest.raises(ValueError, match="Unsupported HTTP method"):
            client.request("TRACE", "/api/test")

    def test_request_case_insensitive(self, mocker):
        client = self._make_client()
        mocker.patch.object(client, "get", return_value={"ok": True})
        client.request("get", "/api/test")
        client.get.assert_called_once()

    def test_get_returns_json_on_200(self, mocker):
        client = self._make_client()
        resp = self._mock_response(200, {"result": "ok"})
        mocker.patch.object(client, "_request", return_value=resp)
        result = client.get("/api/test")
        assert result == {"result": "ok"}

    def test_get_raises_on_non_200(self, mocker):
        client = self._make_client()
        resp = self._mock_response(
            404,
            {"errorDetails": [{"message": "not found", "localizedValue": "Not Found"}]},
        )
        mocker.patch.object(client, "_request", return_value=resp)
        with pytest.raises(Exception):
            client.get("/api/test")

    def test_post_returns_json_on_200(self, mocker):
        client = self._make_client()
        resp = self._mock_response(200, {"created": True})
        mocker.patch.object(client, "_request", return_value=resp)
        result = client.post("/api/test", data={"x": 1})
        assert result == {"created": True}

    def test_post_returns_json_on_201(self, mocker):
        client = self._make_client()
        resp = self._mock_response(201, {"created": True})
        mocker.patch.object(client, "_request", return_value=resp)
        result = client.post("/api/test")
        assert result == {"created": True}

    def test_delete_returns_json_on_200(self, mocker):
        client = self._make_client()
        resp = self._mock_response(200, {"deleted": True})
        mocker.patch.object(client, "_request", return_value=resp)
        result = client.delete("/api/test")
        assert result == {"deleted": True}

    def test_delete_returns_response_on_204_no_body(self, mocker):
        client = self._make_client()
        resp = self._mock_response(204)
        resp.json.side_effect = Exception("no body")
        mocker.patch.object(client, "_request", return_value=resp)
        result = client.delete("/api/test")
        # Should return the raw response when json() fails.
        assert result is resp

    def test_request_timeout_raises(self, mocker):
        client = self._make_client()
        mocker.patch.object(
            client._client,
            "send",
            side_effect=httpx.TimeoutException("timed out"),
        )
        with pytest.raises(APITimeoutError):
            client.get("/api/test")

    def test_request_connect_error_raises(self, mocker):
        client = self._make_client()
        mocker.patch.object(
            client._client,
            "send",
            side_effect=httpx.ConnectError("connection refused"),
        )
        with pytest.raises(Exception, match="connection refused"):
            client.get("/api/test")

    def test_request_generic_error_raises_api_connection_error(self, mocker):
        client = self._make_client()
        mocker.patch.object(
            client._client,
            "send",
            side_effect=OSError("generic IO error"),
        )
        with pytest.raises(APIConnectionError):
            client.get("/api/test")


# ---------------------------------------------------------------------------
# SyncAPIClient — _infer_version
# ---------------------------------------------------------------------------


class TestSyncInferVersion:
    def _make_client(self):
        return SyncAPIClient(
            base_url="https://example.com",
            auth=httpx.BasicAuth("user", "pass"),
        )

    def test_infer_version_24_1(self, mocker):
        client = self._make_client()
        mocker.patch.object(client, "get", return_value={"repository_name": "TestRepo"})
        assert client._infer_version() == "24.1"

    def test_infer_version_24_2(self, mocker):
        client = self._make_client()
        mocker.patch.object(
            client,
            "get",
            return_value={"repository_type": "S3", "repository_name": "TestRepo"},
        )
        assert client._infer_version() == "24.2"

    def test_infer_version_error_mode_raises(self, mocker):
        client = self._make_client()
        mocker.patch.object(client, "get", side_effect=Exception("network down"))
        with pytest.raises(Exception, match="network down"):
            client._infer_version(on_error="error")

    def test_infer_version_warn_mode_prints(self, mocker, capsys):
        client = self._make_client()
        mocker.patch.object(client, "get", side_effect=Exception("network down"))
        client._infer_version(on_error="warn")
        captured = capsys.readouterr()
        assert "Could not validate connection" in captured.out

    def test_infer_version_ignore_mode_returns_none(self, mocker):
        client = self._make_client()
        mocker.patch.object(client, "get", side_effect=Exception("network down"))
        assert client._infer_version(on_error="ignore") is None


# ---------------------------------------------------------------------------
# handle_pega_exception
# ---------------------------------------------------------------------------


class TestHandlePegaException:
    def _make_response(self, status_code, json_data):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.json.return_value = json_data
        return resp

    def test_known_error_dispatches_to_mapped_class(self):
        resp = self._make_response(
            400,
            {
                "errorDetails": [
                    {
                        "message": "Error_NoMonitoringInfo",
                        "localizedValue": "No monitoring",
                    },
                ],
            },
        )
        with pytest.raises(NoMonitoringInfo):
            handle_pega_exception("https://example.com", "/api/test", {}, resp)

    def test_multiple_errors_raises_multiple_errors(self):
        resp = self._make_response(
            400,
            {
                "errorDetails": [
                    {"message": "error1", "localizedValue": "Error 1"},
                    {"message": "error2", "localizedValue": "Error 2"},
                ],
            },
        )
        with pytest.raises(MultipleErrors):
            handle_pega_exception("https://example.com", "/api/test", {}, resp)

    def test_no_details_raises_value_error(self):
        resp = self._make_response(400, {"something": "else"})
        with pytest.raises(ValueError, match="Cannot parse error message"):
            handle_pega_exception("https://example.com", "/api/test", {}, resp)

    def test_non_json_response_raises_invalid_request(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 500
        resp.json.side_effect = Exception("not json")
        resp.content = b"Internal Server Error"
        from pdstools.infinity.internal._exceptions import InvalidRequest

        with pytest.raises(InvalidRequest):
            handle_pega_exception("https://example.com", "/api/test", {}, resp)

    def test_custom_exception_hook(self, mocker):
        """SyncAPIClient.handle_pega_exception should call custom_exception_hook
        if one is installed.
        """
        client = SyncAPIClient(
            base_url="https://example.com",
            auth=httpx.BasicAuth("user", "pass"),
        )
        custom_exc = ValueError("custom error")
        client.custom_exception_hook = MagicMock(return_value=custom_exc)

        resp = self._make_response(
            400,
            {"errorDetails": [{"message": "test", "localizedValue": "test"}]},
        )
        with pytest.raises(ValueError, match="custom error"):
            client.handle_pega_exception("/api/test", {}, resp)

    def test_custom_exception_hook_returns_none_falls_through(self, mocker):
        """If the hook returns None, fall through to the default handling."""
        client = SyncAPIClient(
            base_url="https://example.com",
            auth=httpx.BasicAuth("user", "pass"),
        )
        client.custom_exception_hook = MagicMock(return_value=None)

        resp = self._make_response(
            400,
            {
                "errorDetails": [
                    {
                        "message": "Error_NoMonitoringInfo",
                        "localizedValue": "No monitoring",
                    },
                ],
            },
        )
        with pytest.raises(NoMonitoringInfo):
            client.handle_pega_exception("/api/test", {}, resp)


# ---------------------------------------------------------------------------
# SyncAPIClient — factory methods
# ---------------------------------------------------------------------------


class TestSyncClientFactories:
    def test_from_client_id_and_secret(self):
        client = SyncAPIClient.from_client_id_and_secret(
            base_url="https://example.com",
            client_id="test-id",
            client_secret="test-secret",
        )
        assert isinstance(client, SyncAPIClient)
        assert str(client._base_url) == "https://example.com"

    def test_from_basic_auth(self):
        client = SyncAPIClient.from_basic_auth(
            base_url="https://example.com",
            user_name="admin",
            password="secret",
        )
        assert isinstance(client, SyncAPIClient)

    def test_from_basic_auth_missing_fields_raises(self):
        with pytest.raises(ValueError):
            SyncAPIClient.from_basic_auth(base_url=None, user_name=None, password=None)

    def test_from_client_credentials(self, tmp_path):
        cred_file = tmp_path / "creds.txt"
        # _read_client_credential_file reads alternating lines: key, value, key, value
        cred_file.write_text(
            "Access token endpoint\n"
            "https://example.com/prweb/PRRestService/oauth2/v1/token\n"
            "Client ID\n"
            "test-id\n"
            "Client Secret\n"
            "test-secret\n",
        )
        client = SyncAPIClient.from_client_credentials(str(cred_file))
        assert isinstance(client, SyncAPIClient)
