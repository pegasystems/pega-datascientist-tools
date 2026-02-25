import asyncio
import logging
import os
import warnings
from collections.abc import Coroutine
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
)

import httpx
from anyio import (
    create_task_group,
    from_thread,
    run,
)

from ._auth import PegaOAuth, _read_client_credential_file
from ._exceptions import APIConnectionError, APITimeoutError, handle_pega_exception

_HttpxClientT = TypeVar("_HttpxClientT", bound=httpx.Client | httpx.AsyncClient)
logger = logging.getLogger(__name__)

ResponseT = TypeVar(
    "ResponseT",
    bound=object | str | None | list[Any] | dict[str, Any] | httpx.Response,
)


async def execute_and_collect(
    task_coro: Coroutine,
    results: list,
    i: int,
):  # pragma: no cover
    try:
        result = await task_coro
    except Exception as e:
        logger.exception(e)
        result = e
    results[i] = result


async def get_results(tasks: list[Coroutine]) -> list[Any]:  # pragma: no cover
    results: list[Any] = [None] * len(tasks)

    async with create_task_group() as tg:
        for i, task in enumerate(tasks):
            tg.start_soon(execute_and_collect, task, results, i)

    return results


class BaseClient(Generic[_HttpxClientT]):
    _client: _HttpxClientT

    def __init__(
        self,
        *,
        base_url: str | httpx.URL,
        auth: httpx.Auth | PegaOAuth,
        application_name: str | None = None,
        verify: bool = False,
        pega_version: str | None = None,
        timeout: float = 90,
    ):
        self._base_url = self._enforce_trailing_slash(httpx.URL(base_url))
        self.auth = auth
        self.application_name = application_name
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
        headers: httpx._types.HeaderTypes | None = None,
        data: httpx._types.RequestData | None = None,
        **params,
    ) -> httpx.Request:
        return httpx.Request(
            method,
            url=self._base_url.join(endpoint),
            json=data,
            headers=headers,
            params=params if params else None,
        )

    def _get_version(self, repo):
        if len(repo) == 1 and "repository_name" in repo:
            return "24.1"
        if "repository_type" in repo:
            return "24.2"
        warnings.warn(
            """Could not infer Pega version automatically.
For full compatibility, please supply the pega_version argument to the Infinity class.
""",
        )
        return None

    @classmethod
    def from_client_id_and_secret(
        cls,
        base_url: str,
        client_id: str,
        client_secret: str,
        application_name: str | None = None,
        verify: bool = False,
        pega_version: str | None = None,
        timeout: float = 90,
    ):
        return cls(
            base_url=base_url,
            auth=PegaOAuth(
                base_url,
                client_id=client_id,
                client_secret=client_secret,
                verify=verify,
            ),
            verify=verify,
            application_name=application_name,
            pega_version=pega_version,
            timeout=timeout,
        )

    @classmethod
    def from_client_credentials(
        cls,
        file_path: str,
        verify: bool = False,
        application_name: str | None = None,
        pega_version: str | None = None,
        timeout: float = 90,
    ):
        creds = _read_client_credential_file(file_path)
        base_url = creds["Access token endpoint"].rsplit("/prweb")[0]

        return cls.from_client_id_and_secret(
            base_url=base_url,
            client_id=creds["Client ID"],
            client_secret=creds["Client Secret"],
            application_name=application_name,
            verify=verify,
            pega_version=pega_version,
            timeout=timeout,
        )

    @classmethod
    def from_basic_auth(
        cls,
        base_url: str | None = None,
        user_name: str | None = None,
        password: str | None = None,
        *,
        verify: bool = True,
        application_name: str | None = None,
        pega_version: str | None = None,
        timeout: int = 90,
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
                ),
            )
        auth = httpx.BasicAuth(username=user_name, password=password)
        base_url = base_url.rsplit("/prweb")[0]
        return cls(
            base_url=base_url,
            auth=auth,
            verify=verify,
            application_name=application_name,
            pega_version=pega_version,
            timeout=timeout,
        )


class SyncAPIClient(BaseClient[httpx.Client]):
    _client: httpx.Client

    def __init__(
        self,
        base_url: str | httpx.URL,
        auth: httpx.Auth | PegaOAuth,
        application_name: str | None = None,
        verify: bool = False,
        pega_version: str | None = None,
        timeout: float = 90,
    ):
        super().__init__(
            base_url=base_url,
            auth=auth,
            verify=verify,
            pega_version=pega_version,
        )
        self._client = httpx.Client(
            base_url=self._base_url,
            auth=auth,
            verify=verify,
            timeout=timeout,
        )
        self.application_name = application_name

    def _infer_version(self, on_error: Literal["error", "warn", "ignore"] = "error"):
        try:
            response = self.get("/prweb/api/PredictionStudio/v3/predictions/repository")
        except Exception as e:
            if on_error == "warn":
                print(
                    "Could not validate connection to the Infinity system. "
                    "Please check if the system is up.",
                )
                return None
            if on_error == "error":
                raise e
            return None
        return self._get_version(response)

    def _request(
        self,
        *,
        method,
        endpoint,
        data: httpx._types.RequestData | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        # cast_to: Type[ResponseT], #TODO(someday): implement casting of responses
        **params,
    ) -> httpx.Response:
        request = self._build_request(
            method,
            endpoint,
            data=data,
            headers=headers,
            **params,
        )
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
            exception: Exception | None = self.custom_exception_hook(
                self._base_url,
                endpoint,
                params,
                response,
            )
            if exception:
                raise exception
        raise handle_pega_exception(self._base_url, endpoint, params, response)

    def request(self, method, endpoint, **params):
        method_lower = method.lower()
        if method_lower == "get":
            return self.get(endpoint=endpoint, **params)
        if method_lower == "post":
            return self.post(endpoint=endpoint, **params)
        if method_lower == "patch":
            return self.patch(endpoint=endpoint, **params)
        if method_lower == "put":
            return self.put(endpoint=endpoint, **params)
        if method_lower == "delete":
            return self.delete(endpoint=endpoint, **params)
        raise ValueError(f"Unsupported HTTP method: {method}")

    def get(
        self,
        endpoint: str,
        headers: httpx._types.HeaderTypes | None = None,
        **params,
    ):
        logger.info((self._base_url, endpoint, params))

        response = self._request(
            method="get",
            endpoint=endpoint,
            headers=headers,
            **params,
        )

        if response.status_code != 200:
            raise self.handle_pega_exception(endpoint, params, response)
        return response.json()

    def post(
        self,
        endpoint: str,
        data: httpx._types.RequestData | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        **params,
    ):
        logger.info((self._base_url, endpoint))
        response = self._request(
            method="post",
            endpoint=endpoint,
            headers=headers,
            data=data,
            **params,
        )
        if response.status_code not in (200, 201, 202):
            raise self.handle_pega_exception(endpoint, params, response)

        try:
            return response.json()
        except Exception:
            return response

    def patch(
        self,
        endpoint,
        data: httpx._types.RequestData | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        **params,
    ):
        logger.info((self._base_url, endpoint))
        response = self._request(
            method="patch",
            endpoint=endpoint,
            data=data,
            headers=headers,
            **params,
        )
        if response.status_code != 200:
            raise self.handle_pega_exception(endpoint, params, response)
        return response.json()

    def put(
        self,
        endpoint,
        data: httpx._types.RequestData | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        **params,
    ):
        logger.info((self._base_url, endpoint))
        response = self._request(
            method="put",
            endpoint=endpoint,
            data=data,
            headers=headers,
            **params,
        )
        if response.status_code != 200:
            raise self.handle_pega_exception(endpoint, params, response)
        return response.json()

    def delete(
        self,
        endpoint,
        headers: httpx._types.HeaderTypes | None = None,
        **params,
    ):
        logger.info((self._base_url, endpoint))
        response = self._request(
            method="delete",
            endpoint=endpoint,
            headers=headers,
            **params,
        )
        if response.status_code not in (200, 204):
            raise self.handle_pega_exception(endpoint, params, response)
        try:
            return response.json()
        except Exception:
            return response

    def get_api_list(self):  # pragma: no cover
        raise NotImplementedError


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
        base_url: str | httpx.URL,
        auth: httpx.Auth | PegaOAuth,
        application_name: str | None = None,
        verify: bool = False,
        pega_version: str | None = None,
        timeout: float = 90,
    ):
        super().__init__(
            base_url=base_url,
            auth=auth,
            verify=verify,
            pega_version=pega_version,
        )
        self._client = AsyncHttpxClientWrapper(
            base_url=self._base_url,
            auth=auth,
            verify=verify,
            timeout=timeout,
        )
        self.application_name = application_name

    def _collect_awaitable_blocking(
        self,
        coros: list[Coroutine] | Coroutine,
    ) -> Any:
        if not isinstance(coros, list):
            coros = [coros]
        try:
            awaited = run(get_results, coros)
        except RuntimeError:
            with from_thread.start_blocking_portal() as portal:
                awaited = portal.call(get_results, coros)
        if len(awaited) > 1:
            return awaited
        return awaited[0]

    def _infer_version(self, on_error: Literal["error", "warn", "ignore"] = "error"):
        try:
            repo = self._collect_awaitable_blocking(
                self.get("/prweb/api/PredictionStudio/v3/predictions/repository"),
            )
            # _collect_awaitable_blocking stores exceptions as return values
            # rather than raising them, so we need to re-raise here.
            if isinstance(repo, Exception):
                raise repo
        except Exception as e:
            if on_error == "warn":
                print(
                    "Could not validate connection to the Infinity system. "
                    "Please check if the system is up.",
                )
                return None
            if on_error == "error":
                raise e
            return None
        return self._get_version(repo)

    async def _request(
        self,
        *,
        method,
        endpoint,
        data: httpx._types.RequestData | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        **params,
    ) -> httpx.Response:
        request = self._build_request(
            method,
            endpoint,
            data=data,
            headers=headers,
            **params,
        )

        try:
            response = await self._client.send(request)
        except httpx.TimeoutException as err:
            raise APITimeoutError(request=str(request)) from err
        except httpx.ConnectError as err:
            raise Exception(str(err))
        except Exception as err:
            raise APIConnectionError(request=str(request)) from err
        return response

    def handle_pega_exception(self, endpoint, params, response):
        if hasattr(self, "custom_exception_hook"):
            exception: Exception | None = self.custom_exception_hook(
                self._base_url,
                endpoint,
                params,
                response,
            )
            if exception:
                raise exception
        raise handle_pega_exception(self._base_url, endpoint, params, response)

    async def request(self, method, endpoint, **params):
        method_lower = method.lower()
        if method_lower == "get":
            return await self.get(endpoint=endpoint, **params)
        if method_lower == "post":
            return await self.post(endpoint=endpoint, **params)
        if method_lower == "patch":
            return await self.patch(endpoint=endpoint, **params)
        if method_lower == "put":
            return await self.put(endpoint=endpoint, **params)
        if method_lower == "delete":
            return await self.delete(endpoint=endpoint, **params)
        raise ValueError(f"Unsupported HTTP method: {method}")

    async def get(
        self,
        endpoint: str,
        headers: httpx._types.HeaderTypes | None = None,
        **params,
    ):
        logger.info((self._base_url, endpoint, params))

        response = await self._request(
            method="get",
            endpoint=endpoint,
            headers=headers,
            **params,
        )

        if response.status_code != 200:
            raise self.handle_pega_exception(endpoint, params, response)
        return response.json()

    async def post(
        self,
        endpoint: str,
        data: httpx._types.RequestData | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        **params,
    ):
        logger.info((self._base_url, endpoint))
        response = await self._request(
            method="post",
            endpoint=endpoint,
            headers=headers,
            data=data,
            **params,
        )
        if response.status_code not in (200, 201, 202):
            raise self.handle_pega_exception(endpoint, params, response)

        try:
            return response.json()
        except Exception:
            return response

    async def patch(
        self,
        endpoint,
        data: httpx._types.RequestData | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        **params,
    ):
        logger.info((self._base_url, endpoint))
        response = await self._request(
            method="patch",
            endpoint=endpoint,
            data=data,
            headers=headers,
            **params,
        )
        if response.status_code != 200:
            raise self.handle_pega_exception(endpoint, params, response)
        return response.json()

    async def put(
        self,
        endpoint,
        data: httpx._types.RequestData | None = None,
        headers: httpx._types.HeaderTypes | None = None,
        **params,
    ):
        logger.info((self._base_url, endpoint))
        response = await self._request(
            method="put",
            endpoint=endpoint,
            data=data,
            headers=headers,
            **params,
        )
        if response.status_code != 200:
            raise self.handle_pega_exception(endpoint, params, response)
        return response.json()

    async def delete(
        self,
        endpoint,
        headers: httpx._types.HeaderTypes | None = None,
        **params,
    ):
        logger.info((self._base_url, endpoint))
        response = await self._request(
            method="delete",
            endpoint=endpoint,
            headers=headers,
            **params,
        )
        if response.status_code not in (200, 204):
            raise self.handle_pega_exception(endpoint, params, response)
        try:
            return response.json()
        except Exception:
            return response

    def get_api_list(self):  # pragma: no cover
        raise NotImplementedError
