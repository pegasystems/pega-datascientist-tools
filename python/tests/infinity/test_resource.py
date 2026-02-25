"""Tests for the @api_method decorator, _run_sync, __init_subclass__, and
SyncAPIResource / AsyncAPIResource base classes.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest
from pdstools.infinity.internal._resource import (
    AsyncAPIResource,
    SyncAPIResource,
    _maybe_await,
    _run_sync,
    api_method,
)

# ---------------------------------------------------------------------------
# Helpers — tiny resource classes used exclusively by these tests.
# ---------------------------------------------------------------------------


class _GreetMixin:
    """Mixin with one @api_method and one plain helper."""

    @api_method
    async def greet(self, name: str) -> str:
        result = await self._a_get(f"/hello/{name}")
        return result

    async def _internal_helper(self) -> str:
        """Plain async helper — should NOT be wrapped by __init_subclass__."""
        return "helper"


class SyncGreet(_GreetMixin, SyncAPIResource):
    pass


class AsyncGreet(_GreetMixin, AsyncAPIResource):
    pass


class _MultiMixin:
    """Mixin with multiple @api_method methods to test MRO walking."""

    @api_method
    async def alpha(self) -> str:
        return "alpha"

    @api_method
    async def beta(self, x: int) -> int:
        return x * 2


class SyncMulti(_MultiMixin, SyncAPIResource):
    pass


class AsyncMulti(_MultiMixin, AsyncAPIResource):
    pass


# ---------------------------------------------------------------------------
# @api_method decorator
# ---------------------------------------------------------------------------


class TestApiMethodDecorator:
    def test_sets_marker_attribute(self):
        @api_method
        async def dummy(self):
            pass

        assert getattr(dummy, "_api_method", False) is True

    def test_plain_function_not_marked(self):
        def plain(self):
            pass

        assert getattr(plain, "_api_method", False) is False

    def test_preserves_function_name(self):
        @api_method
        async def my_func(self):
            """Docstring."""

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "Docstring."


# ---------------------------------------------------------------------------
# __init_subclass__ — sync wrapping
# ---------------------------------------------------------------------------


class TestInitSubclass:
    def test_sync_subclass_wraps_api_methods(self):
        """@api_method async defs should be replaced with sync wrappers on
        SyncAPIResource subclasses.
        """
        greet_method = SyncGreet.__dict__.get("greet") or SyncGreet.greet
        # The wrapper is a plain function, NOT a coroutine function.
        assert not inspect.iscoroutinefunction(greet_method)

    def test_sync_subclass_does_not_wrap_plain_helpers(self):
        """Plain async defs (without @api_method) should not be touched."""
        helper = SyncGreet._internal_helper
        assert inspect.iscoroutinefunction(helper)

    def test_async_subclass_leaves_api_methods_as_coroutines(self):
        """On AsyncAPIResource subclasses, @api_method methods stay as native
        coroutines.
        """
        greet_method = AsyncGreet.greet
        assert inspect.iscoroutinefunction(greet_method)

    def test_multiple_methods_wrapped(self):
        """All @api_method methods in the mixin MRO should be wrapped."""
        alpha = SyncMulti.alpha
        beta = SyncMulti.beta
        assert not inspect.iscoroutinefunction(alpha)
        assert not inspect.iscoroutinefunction(beta)

    def test_no_double_wrapping(self):
        """If two subclasses inherit from the same mixin, each gets its own
        wrapping without interfering.
        """

        class SyncA(_GreetMixin, SyncAPIResource):
            pass

        class SyncB(_GreetMixin, SyncAPIResource):
            pass

        # Both should work independently.
        assert not inspect.iscoroutinefunction(SyncA.greet)
        assert not inspect.iscoroutinefunction(SyncB.greet)


# ---------------------------------------------------------------------------
# _run_sync
# ---------------------------------------------------------------------------


class TestRunSync:
    def test_runs_simple_coroutine(self):
        async def add(a, b):
            return a + b

        assert _run_sync(add, 3, 7) == 10

    def test_runs_coroutine_with_kwargs(self):
        async def greet(name, *, greeting="Hello"):
            return f"{greeting}, {name}!"

        assert _run_sync(greet, "World", greeting="Hi") == "Hi, World!"

    def test_propagates_exceptions(self):
        async def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            _run_sync(fail)


# ---------------------------------------------------------------------------
# _maybe_await
# ---------------------------------------------------------------------------


class TestMaybeAwait:
    @pytest.mark.asyncio
    async def test_returns_plain_value_as_is(self):
        result = await _maybe_await(42)
        assert result == 42

    @pytest.mark.asyncio
    async def test_returns_none_as_is(self):
        result = await _maybe_await(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_string_as_is(self):
        result = await _maybe_await("hello")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_awaits_coroutine(self):
        async def make_value():
            return 99

        result = await _maybe_await(make_value())
        assert result == 99


# ---------------------------------------------------------------------------
# SyncAPIResource — instance behaviour
# ---------------------------------------------------------------------------


class TestSyncAPIResource:
    def _make_client(self):
        client = MagicMock()
        client.get.return_value = {"greeting": "hello"}
        client.post.return_value = {"created": True}
        client.patch.return_value = {"updated": True}
        client.put.return_value = {"replaced": True}
        client.delete.return_value = {"deleted": True}
        client.get_api_list.return_value = []
        return client

    def test_init_assigns_shortcuts(self):
        client = self._make_client()
        resource = SyncGreet(client=client)
        assert resource._get is client.get
        assert resource._post is client.post
        assert resource._patch is client.patch
        assert resource._put is client.put
        assert resource._delete is client.delete
        assert resource._get_api_list is client.get_api_list

    def test_sync_wrapped_method_calls_through(self):
        """A sync-wrapped @api_method should call the underlying async body,
        which uses _a_get -> client.get.
        """
        client = self._make_client()
        resource = SyncGreet(client=client)
        result = resource.greet("World")
        assert result == {"greeting": "hello"}
        client.get.assert_called_once_with("/hello/World")

    def test_a_get_delegates_to_sync_client(self):
        """_a_get on SyncAPIResource delegates to self._client.get."""
        client = self._make_client()
        resource = SyncGreet(client=client)

        async def _call():
            return await resource._a_get("/test")

        result = _run_sync(_call)
        assert result == {"greeting": "hello"}

    def test_a_post_delegates_to_sync_client(self):
        client = self._make_client()
        resource = SyncGreet(client=client)

        async def _call():
            return await resource._a_post("/test", data={"x": 1})

        result = _run_sync(_call)
        assert result == {"created": True}
        client.post.assert_called_once_with("/test", data={"x": 1})

    def test_sleep_uses_time_sleep(self, mocker):
        client = self._make_client()
        resource = SyncGreet(client=client)
        mock_sleep = mocker.patch("time.sleep")

        _run_sync(resource._sleep, 0.5)
        mock_sleep.assert_called_once_with(0.5)


# ---------------------------------------------------------------------------
# AsyncAPIResource — instance behaviour
# ---------------------------------------------------------------------------


class TestAsyncAPIResource:
    def _make_client(self):
        client = MagicMock()

        # Async client methods need to be coroutines.
        async def async_get(endpoint, **kwargs):
            return {"greeting": "hello"}

        async def async_post(endpoint, **kwargs):
            return {"created": True}

        async def async_patch(endpoint, **kwargs):
            return {"updated": True}

        async def async_put(endpoint, **kwargs):
            return {"replaced": True}

        async def async_delete(endpoint, **kwargs):
            return {"deleted": True}

        client.get = async_get
        client.post = async_post
        client.patch = async_patch
        client.put = async_put
        client.delete = async_delete
        client.get_api_list.return_value = []
        return client

    def test_init_assigns_get_api_list(self):
        client = self._make_client()
        resource = AsyncGreet(client=client)
        assert resource._get_api_list is client.get_api_list

    @pytest.mark.asyncio
    async def test_a_get_delegates_to_async_client(self):
        client = self._make_client()
        resource = AsyncGreet(client=client)
        result = await resource._a_get("/test")
        assert result == {"greeting": "hello"}

    @pytest.mark.asyncio
    async def test_a_post_delegates_to_async_client(self):
        client = self._make_client()
        resource = AsyncGreet(client=client)
        result = await resource._a_post("/test", data={"x": 1})
        assert result == {"created": True}

    @pytest.mark.asyncio
    async def test_async_api_method_works(self):
        """Calling an @api_method on AsyncAPIResource should return a coroutine
        that, when awaited, gives the expected result.
        """
        client = self._make_client()
        resource = AsyncGreet(client=client)
        result = await resource.greet("World")
        assert result == {"greeting": "hello"}

    @pytest.mark.asyncio
    async def test_sleep_uses_anyio_sleep(self, mocker):
        client = self._make_client()
        resource = AsyncGreet(client=client)
        mock_sleep = mocker.patch("anyio.sleep", return_value=None)
        await resource._sleep(0.5)
        mock_sleep.assert_awaited_once_with(0.5)


# ---------------------------------------------------------------------------
# __repr__ / _public_fields / _public_dict
# ---------------------------------------------------------------------------


class _ReprMixin:
    @api_method
    async def noop(self):
        pass


class SyncRepr(_ReprMixin, SyncAPIResource):
    def __init__(self, client, **kwargs):
        super().__init__(client=client)
        for k, v in kwargs.items():
            setattr(self, k, v)


class AsyncRepr(_ReprMixin, AsyncAPIResource):
    def __init__(self, client, **kwargs):
        super().__init__(client=client)
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestRepr:
    def _client(self):
        client = MagicMock()
        client.get_api_list.return_value = []
        return client

    def test_repr_string_field(self):
        resource = SyncRepr(self._client(), name="Alice")
        assert repr(resource) == "SyncRepr(name='Alice')"

    def test_repr_int_field(self):
        resource = SyncRepr(self._client(), count=42)
        assert repr(resource) == "SyncRepr(count=42)"

    def test_repr_float_field(self):
        resource = SyncRepr(self._client(), score=3.14)
        assert repr(resource) == "SyncRepr(score=3.14)"

    def test_repr_none_field(self):
        resource = SyncRepr(self._client(), value=None)
        assert repr(resource) == "SyncRepr(value=None)"

    def test_repr_bool_field(self):
        resource = SyncRepr(self._client(), active=True)
        assert repr(resource) == "SyncRepr(active=True)"

    def test_repr_multiple_fields(self):
        resource = SyncRepr(self._client(), name="Bob", age=30)
        assert repr(resource) == "SyncRepr(name='Bob', age=30)"

    def test_repr_with_explicit_fields(self):
        resource = SyncRepr(self._client(), name="Charlie", age=25, score=9.5)
        assert (
            resource.__repr__(fields=["name", "score"])
            == "SyncRepr(name='Charlie', score=9.5)"
        )

    def test_repr_no_public_fields(self):
        resource = SyncRepr(self._client())
        assert repr(resource) == "SyncRepr()"

    def test_public_fields(self):
        resource = SyncRepr(self._client(), x=1, y=2)
        assert resource._public_fields == ["x", "y"]

    def test_public_dict(self):
        resource = SyncRepr(self._client(), x=1, y=2)
        assert resource._public_dict == {"x": 1, "y": 2}

    def test_async_repr_string_field(self):
        resource = AsyncRepr(self._client(), name="Alice")
        assert repr(resource) == "AsyncRepr(name='Alice')"

    def test_async_public_fields(self):
        resource = AsyncRepr(self._client(), x=1, y=2)
        assert resource._public_fields == ["x", "y"]
