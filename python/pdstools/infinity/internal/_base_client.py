import asyncio
import json
import logging
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)

import httpx
from anyio import (
    create_task_group,
    from_thread,
    run,
)

from ._auth import PegaOAuth, _read_client_credential_file
from ._exceptions import APIConnectionError, APITimeoutError, handle_pega_exception

_HttpxClientT = TypeVar("_HttpxClientT", bound=Union[httpx.Client, httpx.AsyncClient])
logger = logging.getLogger(__name__)

ResponseT = TypeVar(
    "ResponseT",
    bound=Union[
        object,
        str,
        None,
        List[Any],
        Dict[str, Any],
        httpx.Response,
    ],
)


async def execute_and_collect(
    task_coro: Coroutine, results: List, i: int
):  # pragma: no cover
    try:
        result = await task_coro
    except Exception as e:
        logger.exception(e)
        result = e
    results[i] = result


async def get_results(tasks: List[Coroutine]) -> List[Any]:  # pragma: no cover
    results: List[Any] = [None] * len(tasks)

    async with create_task_group() as tg:
        for i, task in enumerate(tasks):
            tg.start_soon(execute_and_collect, task, results, i)

    return results


class BaseClient(Generic[_HttpxClientT]):
    _client: _HttpxClientT

    def __init__(
        self,
        *,
        base_url: Union[str, httpx.URL],
        auth: Union[httpx.Auth, PegaOAuth],
        verify: bool = False,
        pega_version: Union[str, None] = None,
        timeout: float = 90,
    ):
        self._base_url = self._enforce_trailing_slash(httpx.URL(base_url))
        self.auth = auth
        self.verify = verify
        self.pega_version = pega_version
        self.timeout = timeout

    def _enforce_trailing_slash(self, url: httpx.URL) -> httpx.URL:
        if url.raw_path.endswith(b"/"):
            return url
        return url.copy_with(raw_path=url.raw_path + b"/")

    def _build_request(
        self,
        method,
        endpoint: str,
        data: Union[httpx._types.RequestData, None] = None,
        **params,
    ) -> httpx.Request:
        return httpx.Request(
            method,
            url=self._base_url.join(endpoint),
            content=json.dumps(data) if data else None,
            params=params,
        )

    def _get_version(self, repo):
        if len(repo) == 1 and "repository_name" in repo:
            return "24.1"
        elif "repository_type" in repo:
            return "24.2"
        else:
            print(
                "Could not infer Pega version automatically. ",
                "For full compatibility, please supply the pega_version argument",
                "to the Infinity class.",
            )
            return "Undefined"

    @classmethod
    def from_client_credentials(
        cls,
        file_path: str,
        verify: bool = False,
        pega_version: Union[str, None] = None,
        timeout: float = 20,
    ):
        creds = _read_client_credential_file(file_path)
        base_url = creds["Access token endpoint"].rsplit("/prweb")[0]

        return cls(
            base_url=base_url,
            auth=PegaOAuth(
                base_url,
                creds["Client ID"],
                client_secret=creds["Client Secret"],
                verify=verify,
            ),
            verify=verify,
            pega_version=pega_version,
            timeout=timeout,
        )

    @classmethod
    def from_basic_auth(
        cls,
        base_url: Optional[str] = None,
        user_name: Optional[str] = None,
        password: Optional[str] = None,
        *,
        verify: bool = True,
        pega_version: Union[str, None] = None,
        timeout: int = 20,
    ):
        base_url = base_url or os.environ.get("PEGA_BASE_URL")
        user_name = user_name or os.environ.get("PEGA_USERNAME")
        password = password or os.environ.get("PEGA_PASSWORD")
        if not base_url or not user_name or not password:
            raise ValueError(
                (
                    "To use Basic authentication, either provide ",
                    "base_url, user_name and password directly in `from_basic_auth` ",
                    "or set the PEGA_BASE_URL, PEGA_USERNAME & PEGA_PASSWORD ",
                    "environment variables before running your code. ",
                )
            )
        auth = httpx.BasicAuth(username=user_name, password=password)
        base_url = base_url.rsplit("/prweb")[0]
        return cls(
            base_url=base_url,
            auth=auth,
            verify=verify,
            pega_version=pega_version,
            timeout=timeout,
        )


class SyncAPIClient(BaseClient[httpx.Client]):
    _client: httpx.Client

    def __init__(
        self,
        base_url: Union[str, httpx.URL],
        auth: Union[httpx.Auth, PegaOAuth],
        verify: bool = False,
        pega_version: Union[str, None] = None,
        timeout: float = 90,
    ):
        super().__init__(
            base_url=base_url, auth=auth, verify=verify, pega_version=pega_version
        )
        self._client = httpx.Client(
            base_url=self._base_url,
            auth=auth,
            verify=verify,
            timeout=timeout,
        )

    def _infer_version(self):
        try:
            response = self.get("/prweb/api/PredictionStudio/v3/predictions/repository")
        except APIConnectionError as e:
            print(
                "Could not validate connection to the Infinity system. "
                "Please check if the system is up."
            )
            raise e
        return self._get_version(response)

    def _request(
        self,
        *,
        method,
        endpoint,
        data: Union[httpx._types.RequestData, None] = None,
        # cast_to: Type[ResponseT], #TODO(someday): implement casting of responses
        **params,
    ) -> httpx.Response:
        request = self._build_request(method, endpoint, data=data, **params)
        try:
            response = self._client.send(request)
        except httpx.TimeoutException as err:
            raise APITimeoutError(request=str(request)) from err
        except httpx.ConnectError as err:
            raise Exception(str(err))
        except Exception as err:
            raise APIConnectionError(request=str(request)) from err
        return response

    def handle_pega_exception(self, endpoint, params, response):
        if hasattr(self, "custom_exception_hook"):
            exception: Optional[Exception] = self.custom_exception_hook(
                self._base_url, endpoint, params, response
            )
            if exception:
                raise exception
        raise handle_pega_exception(self._base_url, endpoint, params, response)

    def request(self, method, endpoint, **params):
        if method.lower() == "get":
            return self.get(endpoint=endpoint, **params)

    def get(self, endpoint: str, **params):
        logger.info((self._base_url, endpoint, params))

        response = self._request(method="get", endpoint=endpoint, **params)

        if response.status_code != 200:
            raise self.handle_pega_exception(endpoint, params, response)
        return response.json()

    def post(
        self,
        endpoint: str,
        data: Union[httpx._types.RequestData, None] = None,
        **params,
    ):
        logger.info((self._base_url, endpoint))
        response = self._request(method="post", endpoint=endpoint, data=data, **params)
        if response.status_code not in (200, 201, 202):
            raise self.handle_pega_exception(endpoint, params, response)

        try:
            return response.json()
        except Exception:
            return response

    def patch(
        self, endpoint, data: Union[httpx._types.RequestData, None] = None, **params
    ):
        logger.info((self._base_url, endpoint))
        response = self._request(method="patch", endpoint=endpoint, data=data, **params)
        if response.status_code != 200:
            raise self.handle_pega_exception(endpoint, params, response)
        return response.json()

    def put(
        self, endpoint, data: Union[httpx._types.RequestData, None] = None, **params
    ):
        logger.info((self._base_url, endpoint))
        response = self._request(method="put", endpoint=endpoint, data=data, **params)
        if response.status_code != 200:
            raise self.handle_pega_exception(endpoint, params, response)
        return response.json()

    def delete(self):  # pragma: no cover
        raise NotImplementedError()

    def get_api_list(self):  # pragma: no cover
        raise NotImplementedError()


class _DefaultAsyncHttpxClient(httpx.AsyncClient):  # pragma: no cover
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("follow_redirects", True)
        super().__init__(**kwargs)


if TYPE_CHECKING:  # pragma: no cover
    DefaultAsyncHttpxClient = httpx.AsyncClient
    """An alias to `httpx.AsyncClient` that provides the same defaults that this SDK
    uses internally.

    This is useful because overriding the `http_client` with your own instance of
    `httpx.AsyncClient` will result in httpx's defaults being used, not ours.
    """
else:
    DefaultAsyncHttpxClient = _DefaultAsyncHttpxClient


class AsyncHttpxClientWrapper(DefaultAsyncHttpxClient):
    def __del__(self) -> None:  # pragma: no cover
        try:
            # TODO(someday): support non asyncio runtimes here
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:
            pass


class AsyncAPIClient(BaseClient[httpx.AsyncClient]):  # pragma: no cover
    _client: httpx.AsyncClient

    def __init__(
        self,
        base_url: Union[str, httpx.URL],
        auth: Union[httpx.Auth, PegaOAuth],
        verify: bool = False,
        pega_version: Union[str, None] = None,
    ):
        super().__init__(
            base_url=base_url, auth=auth, verify=verify, pega_version=pega_version
        )
        self._client = AsyncHttpxClientWrapper(
            base_url=self._base_url, auth=auth, verify=verify
        )

    def _collect_awaitable_blocking(
        self, coros: Union[List[Coroutine], Coroutine]
    ) -> Any:
        if not isinstance(coros, List):
            coros = [coros]
        try:
            awaited = run(get_results, coros)
        except RuntimeError:
            with from_thread.start_blocking_portal() as portal:
                awaited = portal.call(get_results, coros)
        if len(awaited) > 1:
            return awaited
        return awaited[0]

    def _infer_version(self):
        try:
            repo = self._collect_awaitable_blocking(
                self.get("/prweb/api/PredictionStudio/v3/predictions/repository")
            )
        except APIConnectionError:
            print(
                "Could not validate connection to the Infinity system."
                "Please check if the system is up."
            )
            return None
        return self._get_version(repo)

    async def _request(
        self,
        *,
        method,
        endpoint,
        # cast_to: Type[ResponseT],
        **params,
        # ) -> ResponseT:
    ) -> httpx.Response:
        request = self._build_request(method, endpoint, **params)

        try:
            response = await self._client.send(request)
        except httpx.TimeoutException as err:
            raise APITimeoutError(request=request) from err
        except Exception as err:
            raise APIConnectionError(request=request) from err
        return response

    async def request(self, method, endpoint, **params):
        if method.lower() == "get":
            return await self.get(endpoint=endpoint, **params)

    async def get(self, endpoint: str, **params):
        logger.info((self._base_url, endpoint, params))

        response = await self._request(method="get", endpoint=endpoint, params=params)

        if response.status_code != 200:
            raise handle_pega_exception(self._base_url, endpoint, params, response)
        return response.json()

    def post(self):
        raise NotImplementedError()

    def patch(self):
        raise NotImplementedError()

    def put(self):
        raise NotImplementedError()

    def delete(self):
        raise NotImplementedError()

    def get_api_list(self):
        raise NotImplementedError()
