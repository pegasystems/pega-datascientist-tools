import re
import time

import httpx
import pytest
from pdstools.infinity import Infinity
from pdstools.infinity.internal import _auth, _base_client, _exceptions
from pytest_httpx import HTTPXMock


@pytest.fixture
def mock_auth(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json={"access_token": "ABC", "expires_in": 30})

    return _auth.PegaOAuth(
        "https://pega.com",
        client_id="test_id",
        client_secret="test_secret",
    )


def test_token(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json={"access_token": "ABC", "expires_in": 30})
    PegaAuth = _auth.PegaOAuth(
        "https://TEST.com",
        client_id="TEST",
        client_secret="TEST",
    )

    assert PegaAuth.token == "ABC"
    requests_start = httpx_mock.get_requests()

    assert PegaAuth._token_expiry - time.time() > 5
    # assert PegaAuth._token_expiry - time.time() < 30 #this was flakey and doesn't add much

    PegaAuth.token  # fetching token again to make sure we don't re-retrieve
    assert requests_start == httpx_mock.get_requests()

    assert PegaAuth._auth_header == f"Bearer {'ABC'}"


def test_connection_eror(httpx_mock: HTTPXMock):
    httpx_mock.add_response(status_code=201, json={"Error": "Invalid."})
    PegaAuth = _auth.PegaOAuth(
        "https://TEST.com",
        client_id="TEST",
        client_secret="TEST",
    )
    with pytest.raises(ConnectionError):
        PegaAuth.token


class TestPegaOAuthErrorRedaction:
    """Auth error messages must expose only safe OAuth fields, never URLs or raw bodies."""

    def _make_auth(self) -> _auth.PegaOAuth:
        return _auth.PegaOAuth(
            "https://pega.example.com",
            client_id="id",
            client_secret="secret",
        )

    def test_error_description_surfaced(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            status_code=401,
            json={"error": "invalid_client", "error_description": "Bad credentials"},
        )
        with pytest.raises(ConnectionError, match="Bad credentials"):
            self._make_auth().token

    def test_error_field_used_when_no_description(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            status_code=401,
            json={"error": "invalid_client"},
        )
        with pytest.raises(ConnectionError, match="invalid_client"):
            self._make_auth().token

    def test_no_oauth_fields_shows_fallback(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            status_code=401,
            json={"message": "something internal"},
        )
        with pytest.raises(ConnectionError, match="no error detail returned"):
            self._make_auth().token

    def test_non_json_error_body_safe_message(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            status_code=401,
            content=b"<html>Unauthorized</html>",
        )
        with pytest.raises(ConnectionError, match="non-JSON response from token endpoint"):
            self._make_auth().token

    def test_token_url_not_leaked(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            status_code=403,
            json={"error": "forbidden"},
        )
        with pytest.raises(ConnectionError) as exc_info:
            self._make_auth().token
        assert "/prweb/PRRestService/oauth2/v1/token" not in str(exc_info.value)
        assert "pega.example.com" not in str(exc_info.value)

    def test_raw_body_not_leaked(self, httpx_mock: HTTPXMock):
        secret_body = '{"internal_trace_id": "abc-123", "server": "internal-host"}'
        httpx_mock.add_response(
            status_code=500,
            content=secret_body.encode(),
        )
        with pytest.raises(ConnectionError) as exc_info:
            self._make_auth().token
        assert "internal_trace_id" not in str(exc_info.value)
        assert "internal-host" not in str(exc_info.value)

    def test_success_non_json_response_raises(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            status_code=200,
            content=b"not-json",
        )
        with pytest.raises(ConnectionError, match="non-JSON response"):
            self._make_auth().token

    def test_status_code_included_in_message(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            status_code=503,
            json={"error": "service_unavailable"},
        )
        with pytest.raises(ConnectionError, match="503"):
            self._make_auth().token


@pytest.mark.filterwarnings("ignore:Could not infer Pega version automatically:UserWarning")
def test_base_client(httpx_mock: HTTPXMock, monkeypatch):
    _base_client.BaseClient(base_url="TEST", auth="Test")  # no validation yet

    client = _base_client.BaseClient(base_url="https://TEST.com", auth=httpx.Auth())
    assert isinstance(client._base_url, httpx.URL)
    assert client._base_url == "https://TEST.com"
    assert client.pega_version is None
    assert client._get_version({"repository_name": "Repo"}) == "24.1"
    assert client._get_version({"repository_type": "AWS"}) == "24.2"
    assert client._get_version({"Pega Version": "Undefined"}) is None

    with pytest.raises(ValueError):
        _base_client.BaseClient.from_basic_auth()

    client = _base_client.BaseClient.from_basic_auth(
        base_url="https://PEGA.com/prweb/TEST",
        user_name="USER",
        password="PASSWORD",
    )
    assert client._base_url == "https://pega.com"
    assert isinstance(client.auth, httpx.BasicAuth)

    httpx_mock.add_response(json={"access_token": "ABC", "expires_in": 30})

    def mocked_client_credential_read(path):
        return {
            "Access token endpoint": "https://abc",
            "Client ID": "123",
            "Client Secret": "XYZ",
        }

    monkeypatch.setattr(
        _base_client,
        "_read_client_credential_file",
        mocked_client_credential_read,
    )
    client = _base_client.BaseClient.from_client_credentials(file_path="TEST")

    assert client._base_url == "https://abc"
    assert client.auth.token == "ABC"


def test_sync_client(httpx_mock: HTTPXMock, mock_auth):
    client = _base_client.SyncAPIClient(base_url="https://pega.com", auth=mock_auth)
    assert client.auth.token == "ABC"

    # 25 probe returns 404 (24.x system), fall back to repository.
    httpx_mock.add_response(
        url=re.compile(".*/modelCategories"),
        status_code=404,
        json={},
    )
    httpx_mock.add_response(
        url=re.compile(".*/repository"),
        json={"repository_name": "Repo"},
    )
    assert client._infer_version() == "24.1"

    # 25 probe returns 404 again, repository raises connection error.
    httpx_mock.add_response(
        url=re.compile(".*/modelCategories"),
        status_code=404,
        json={},
    )
    httpx_mock.add_exception(
        _exceptions.APIConnectionError("Failed to read properly"),
        url=re.compile(".*/repository"),
    )

    with pytest.raises(_exceptions.APIConnectionError):
        client._infer_version()

    httpx_mock.add_exception(
        httpx.TimeoutException("Timed out!"),
        url=re.compile(".*/test"),
    )

    with pytest.raises(_exceptions.APITimeoutError):
        client.post("/test")

    httpx_mock.add_exception(
        httpx.ConnectError("Some other connection error!"),
        url=re.compile(".*/test"),
    )

    with pytest.raises(Exception):
        client.post("/test")

    httpx_mock.add_response(url=re.compile(".*/test"), json={"Test": True})
    assert client.request(method="get", endpoint="/test") == {"Test": True}

    httpx_mock.add_response(url=re.compile(".*/test"), status_code=201)
    with pytest.raises(_exceptions.InvalidRequest):
        client.get("/test")

    httpx_mock.add_response(
        url=re.compile(".*/test"),
        status_code=200,
        json={"status": "Success"},
    )
    assert client.post("/test") == {"status": "Success"}

    httpx_mock.add_response(url=re.compile(".*/test"), status_code=300)
    with pytest.raises(_exceptions.InvalidRequest):
        client.post("/test")

    httpx_mock.add_response(
        url=re.compile(".*/test"),
        status_code=200,
        json={"status": "Success"},
    )
    assert client.patch("/test") == {"status": "Success"}

    httpx_mock.add_response(url=re.compile(".*/test"), status_code=300)
    with pytest.raises(_exceptions.InvalidRequest):
        client.patch("/test")

    httpx_mock.add_response(
        url=re.compile(".*/test"),
        status_code=200,
        json={"status": "Success"},
    )
    assert client.put("/test") == {"status": "Success"}

    httpx_mock.add_response(url=re.compile(".*/test"), status_code=300)
    with pytest.raises(_exceptions.InvalidRequest):
        client.put("/test")


def test_infinity_client(httpx_mock: HTTPXMock, mock_auth, monkeypatch):
    with pytest.raises(TypeError):
        Infinity()  # type: ignore[call-arg]

    httpx_mock.add_response(
        url=re.compile(".*/modelCategories"),
        status_code=404,
        json={},
    )
    httpx_mock.add_response(
        url=re.compile(".*/repository"),
        json={"repository_name": "Repo"},
    )

    def mocked_client_credential_read(path):
        return {
            "Access token endpoint": "https://abc",
            "Client ID": "123",
            "Client Secret": "XYZ",
        }

    monkeypatch.setattr(
        _base_client,
        "_read_client_credential_file",
        mocked_client_credential_read,
    )
    client = Infinity.from_client_credentials(file_path="TEST")

    assert client.version == "24.1"


def test_error_handling(httpx_mock: HTTPXMock, mock_auth):
    # Version is resolved lazily on first .version access, so no
    # /repository mock is needed for construction.
    client = Infinity(base_url="https://pega.com", auth=mock_auth)

    httpx_mock.add_response(status_code=205, json={"no_error": "details"})
    with pytest.raises(ValueError):
        client.post("/test")

    httpx_mock.add_response(status_code=205, json={"errorDetails": ["TEST", "TEST2"]})
    with pytest.raises(_exceptions.MultipleErrors):
        client.post("/test")

    httpx_mock.add_response(status_code=205, json={"errorDetails": ["TEST"]})
    with pytest.raises(Exception):
        client.post("/test")

    # httpx_mock.add_response(status_code=205, json={"errorDetails": ["TEST", "TEST2"]})
    # with pytest.raises(Exception):
    #     client.post("/test")

    httpx_mock.add_response(
        status_code=205,
        json={"errorDetails": [{"message": "Error_NoMonitoringInfo"}]},
    )
    with pytest.raises(_exceptions.NoMonitoringInfo):
        client.post("/test")
