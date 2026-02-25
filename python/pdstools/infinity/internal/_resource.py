# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import functools
import time
from abc import ABC
from typing import TYPE_CHECKING, List, Union
import inspect

import anyio
from anyio.from_thread import start_blocking_portal

if TYPE_CHECKING:
    from ._base_client import AsyncAPIClient, SyncAPIClient


def _run_sync(async_fn, *args, **kwargs):
    """Run an async function synchronously.

    Uses ``anyio.run`` when no event loop is running, and falls back to
    ``start_blocking_portal`` when called from within an existing loop
    (e.g. Jupyter).
    """

    async def _wrapper():
        return await async_fn(*args, **kwargs)

    try:
        return anyio.run(_wrapper)
    except RuntimeError:
        with start_blocking_portal() as portal:
            return portal.call(_wrapper)


def api_method(func):
    """Mark an ``async def`` resource method for automatic sync wrapping.

    Methods decorated with ``@api_method`` should be defined as
    ``async def`` and may ``await`` any of the ``_a_get``/``_a_post``/…
    helpers on the resource.

    * On :class:`AsyncAPIResource` subclasses the method is left as-is
      (native coroutine).
    * On :class:`SyncAPIResource` subclasses the method is automatically
      replaced by a synchronous wrapper (via ``__init_subclass__``) that
      executes the coroutine with :func:`_run_sync`.
    """
    func._api_method = True
    return func


async def _maybe_await(result):
    """Await the result if it's awaitable, otherwise return it as-is.

    Useful in mixin methods that call something which is sync on
    ``SyncAPIResource`` (returns a plain value) but async on
    ``AsyncAPIResource`` (returns a coroutine).
    """
    if inspect.isawaitable(result):
        return await result
    return result


class SyncAPIResource(ABC):
    _client: SyncAPIClient

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Walk the MRO so that @api_method methods defined in mixins
        # are also picked up and wrapped.
        _seen: set[str] = set()
        for klass in cls.__mro__:
            for name, method in vars(klass).items():
                if name in _seen:
                    continue
                if (
                    callable(method)
                    and getattr(method, "_api_method", False)
                    and inspect.iscoroutinefunction(method)
                ):
                    _seen.add(name)
                    original = method

                    @functools.wraps(original)
                    def _sync_wrapper(self, *args, _orig=original, **kwargs):
                        return _run_sync(_orig, self, *args, **kwargs)

                    setattr(cls, name, _sync_wrapper)

    def __init__(self, client: SyncAPIClient) -> None:
        self._client = client
        # Sync shortcuts — used directly by existing resource code
        # (e.g. PredictionStudio).  ``self._post(endpoint, data=data)``
        # works synchronously without ``await``.
        self._get = client.get
        self._post = client.post
        self._patch = client.patch
        self._put = client.put
        self._delete = client.delete
        self._get_api_list = client.get_api_list

    # -- Async wrappers for @api_method bodies ----------------------------
    # ``@api_method`` async bodies use ``await self._a_post(…)`` etc.
    # These thin async wrappers delegate to the synchronous httpx client.
    # They are only ever executed inside ``_run_sync`` (which spins up its
    # own event loop), so no concurrency or nesting issues arise.

    async def _a_get(self, endpoint: str, **kwargs):
        return self._client.get(endpoint, **kwargs)

    async def _a_post(self, endpoint: str, **kwargs):
        return self._client.post(endpoint, **kwargs)

    async def _a_patch(self, endpoint: str, **kwargs):
        return self._client.patch(endpoint, **kwargs)

    async def _a_put(self, endpoint: str, **kwargs):
        return self._client.put(endpoint, **kwargs)

    async def _a_delete(self, endpoint: str, **kwargs):
        return self._client.delete(endpoint, **kwargs)

    async def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    @property
    def _public_fields(self):
        return [field for field in self.__dict__ if not field.startswith("_")]

    @property
    def _public_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def __repr__(self, fields: Union[List[str], None] = None):
        classname = self.__class__.__name__

        def format_field(field):
            value = self.__getattribute__(field)
            if isinstance(value, (int, float)):
                return str(value)
            if isinstance(value, bool):
                return str(value)
            if value is None:
                return "None"
            value = str(value)
            return "'" + value + "'"

        if fields:
            return f"{classname}({', '.join([field + '=' + format_field(field) for field in fields])})"
        return f"{classname}({', '.join([field + '=' + format_field(field) for field in self._public_fields])})"


class AsyncAPIResource(ABC):
    _client: AsyncAPIClient

    def __init__(self, client: AsyncAPIClient) -> None:
        self._client = client
        self._get_api_list = client.get_api_list

    # -- Async delegates --------------------------------------------------
    # Used by @api_method bodies (``await self._a_post(…)``).
    # Also aliased to ``_get``/``_post``/… for convenience.

    async def _a_get(self, endpoint: str, **kwargs):
        return await self._client.get(endpoint, **kwargs)

    async def _a_post(self, endpoint: str, **kwargs):
        return await self._client.post(endpoint, **kwargs)

    async def _a_patch(self, endpoint: str, **kwargs):
        return await self._client.patch(endpoint, **kwargs)

    async def _a_put(self, endpoint: str, **kwargs):
        return await self._client.put(endpoint, **kwargs)

    async def _a_delete(self, endpoint: str, **kwargs):
        return await self._client.delete(endpoint, **kwargs)

    # Aliases so that pure-async resource code can use either naming style.
    _get = _a_get
    _post = _a_post
    _patch = _a_patch
    _put = _a_put
    _delete = _a_delete

    async def _sleep(self, seconds: float) -> None:
        await anyio.sleep(seconds)

    @property
    def _public_fields(self):
        return [field for field in self.__dict__ if not field.startswith("_")]

    @property
    def _public_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def __repr__(self, fields: Union[List[str], None] = None):
        classname = self.__class__.__name__

        def format_field(field):
            value = self.__getattribute__(field)
            if isinstance(value, (int, float)):
                return str(value)
            if isinstance(value, bool):
                return str(value)
            if value is None:
                return "None"
            value = str(value)
            return "'" + value + "'"

        if fields:
            return f"{classname}({', '.join([field + '=' + format_field(field) for field in fields])})"
        return f"{classname}({', '.join([field + '=' + format_field(field) for field in self._public_fields])})"
