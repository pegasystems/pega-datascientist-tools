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
        "https://pega.com", client_id="test_id", client_secret="test_secret"
    )


def test_token(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json={"access_token": "ABC", "expires_in": 30})
    PegaAuth = _auth.PegaOAuth(
        "https://TEST.com", client_id="TEST", client_secret="TEST"
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
        "https://TEST.com", client_id="TEST", client_secret="TEST"
    )
    with pytest.raises(ConnectionError):
        PegaAuth.token


def test_base_client(httpx_mock: HTTPXMock, monkeypatch):
    _base_client.BaseClient(base_url="TEST", auth="Test")  # no validation yet

    client = _base_client.BaseClient(base_url="https://TEST.com", auth=httpx.Auth())
    assert isinstance(client._base_url, httpx.URL)
    assert client._base_url == "https://TEST.com"
    assert client.pega_version is None
    assert client._get_version({"repository_name": "Repo"}) == "24.1"
    assert client._get_version({"repository_type": "AWS"}) == "24.2"
    assert client._get_version({"Pega Version": "Undefined"}) == "Undefined"

    with pytest.raises(ValueError):
        _base_client.BaseClient.from_basic_auth()

    client = _base_client.BaseClient.from_basic_auth(
        base_url="https://PEGA.com/prweb/TEST", user_name="USER", password="PASSWORD"
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
        _base_client, "_read_client_credential_file", mocked_client_credential_read
    )
    client = _base_client.BaseClient.from_client_credentials(file_path="TEST")

    assert client._base_url == "https://abc"
    assert client.auth.token == "ABC"


def test_sync_client(httpx_mock: HTTPXMock, mock_auth):
    client = _base_client.SyncAPIClient(base_url="https://pega.com", auth=mock_auth)
    assert client.auth.token == "ABC"

    httpx_mock.add_response(
        url=re.compile(".*/repository"), json={"repository_name": "Repo"}
    )
    assert client._infer_version() == "24.1"
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
        url=re.compile(".*/test"), status_code=200, json={"status": "Success"}
    )
    assert client.post("/test") == {"status": "Success"}

    httpx_mock.add_response(url=re.compile(".*/test"), status_code=300)
    with pytest.raises(_exceptions.InvalidRequest):
        client.post("/test")

    httpx_mock.add_response(
        url=re.compile(".*/test"), status_code=200, json={"status": "Success"}
    )
    assert client.patch("/test") == {"status": "Success"}

    httpx_mock.add_response(url=re.compile(".*/test"), status_code=300)
    with pytest.raises(_exceptions.InvalidRequest):
        client.patch("/test")

    httpx_mock.add_response(
        url=re.compile(".*/test"), status_code=200, json={"status": "Success"}
    )
    assert client.put("/test") == {"status": "Success"}

    httpx_mock.add_response(url=re.compile(".*/test"), status_code=300)
    with pytest.raises(_exceptions.InvalidRequest):
        client.put("/test")


def test_infinity_client(httpx_mock: HTTPXMock, mock_auth, monkeypatch):
    with pytest.raises(RuntimeError):
        Infinity()

    httpx_mock.add_response(
        url=re.compile(".*/repository"), json={"repository_name": "Repo"}
    )

    def mocked_client_credential_read(path):
        return {
            "Access token endpoint": "https://abc",
            "Client ID": "123",
            "Client Secret": "XYZ",
        }

    monkeypatch.setattr(
        _base_client, "_read_client_credential_file", mocked_client_credential_read
    )
    client = Infinity.from_client_credentials(file_path="TEST")

    assert client.version == "24.1"


def test_error_handling(httpx_mock: HTTPXMock, mock_auth):
    httpx_mock.add_response(
        url=re.compile(".*/repository"), json={"repository_name": "Repo"}
    )
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
        status_code=205, json={"errorDetails": [{"message": "Error_NoMonitoringInfo"}]}
    )
    with pytest.raises(_exceptions.NoMonitoringInfo):
        client.post("/test")
