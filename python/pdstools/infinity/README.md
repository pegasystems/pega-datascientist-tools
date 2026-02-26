# Infinity Client — Developer Guide

This document explains how the Infinity API client is structured and how to
add new resource methods using the **write-once async-first** pattern.

## Architecture overview

The client exposes two entry points:

| Class | Import | Usage |
|---|---|---|
| `Infinity` | `from pdstools import Infinity` | Synchronous (default) |
| `AsyncInfinity` | `from pdstools import AsyncInfinity` | Native `async`/`await` |

Both clients share the same resource methods. Business logic is defined
**once** as `async def` in a mixin class, and automatically works for both
sync and async callers.

```
Infinity (SyncAPIClient)
├── knowledge_buddy: KnowledgeBuddy
│     └── inherits: _KnowledgeBuddyMixin + SyncAPIResource
└── prediction_studio: PredictionStudio  (version-dependent)

AsyncInfinity (AsyncAPIClient)
├── knowledge_buddy: AsyncKnowledgeBuddy
│     └── inherits: _KnowledgeBuddyMixin + AsyncAPIResource
└── prediction_studio: AsyncPredictionStudio  (version-dependent)
```

## The write-once pattern

### Key components

| Component | Location | Purpose |
|---|---|---|
| `@api_method` | `internal/_resource.py` | Decorator that marks an `async def` for auto sync-wrapping |
| `_run_sync` | `internal/_resource.py` | Runs a coroutine synchronously via `anyio` |
| `SyncAPIResource` | `internal/_resource.py` | Base class; `__init_subclass__` replaces `@api_method` methods with sync wrappers |
| `AsyncAPIResource` | `internal/_resource.py` | Base class; leaves `@api_method` methods as native coroutines |

### How it works

1. You write business logic **once** as `async def` methods decorated with
   `@api_method` in a mixin class.
2. Inside those methods, use `await self._a_post(...)`, `await self._a_get(...)`,
   etc. for HTTP calls.
3. Create two thin concrete classes — one inheriting from the mixin +
   `SyncAPIResource`, the other from the mixin + `AsyncAPIResource`.
4. `SyncAPIResource.__init_subclass__` walks the MRO at class-creation time,
   finds all `@api_method`-decorated methods, and replaces them with sync
   wrappers that call `_run_sync(original_async_method, self, *args, **kwargs)`.
5. On the async side, the methods stay as-is — native coroutines.

**Result**: callers use the same method names, the same signatures, the same
docstrings. Sync callers get a regular return value; async callers `await` it.

## How to add a new resource

### Step 1: Create the mixin

Define all business logic in a mixin class. Methods that call the API must
be `async def` and decorated with `@api_method`.

```python
# resources/my_resource/my_resource.py

from typing import Optional
from ...internal._resource import AsyncAPIResource, SyncAPIResource, api_method

class _MyResourceMixin:
    """Business logic for MyResource — defined once."""

    @api_method
    async def do_something(self, item_id: str, *, verbose: bool = False) -> dict:
        """Fetch something from the API.

        Parameters
        ----------
        item_id : str
            The item identifier.
        verbose : bool
            Include extra detail in the response.

        Returns
        -------
        dict
            The API response payload.
        """
        response = await self._a_get(
            "/prweb/api/MyResource/v1/items",
            itemId=item_id,
            verbose=verbose,
        )
        return response

    @api_method
    async def create_item(self, name: str, data: Optional[dict] = None) -> dict:
        """Create a new item."""
        response = await self._a_post(
            "/prweb/api/MyResource/v1/items",
            data={"name": name, **(data or {})},
        )
        return response
```

### Step 2: Create the concrete classes

```python
# Same file, below the mixin:

class MyResource(_MyResourceMixin, SyncAPIResource):
    def __init__(self, client):
        super().__init__(client)


class AsyncMyResource(_MyResourceMixin, AsyncAPIResource):
    def __init__(self, client):
        super().__init__(client)
```

That's it. `MyResource.do_something(...)` is synchronous.
`AsyncMyResource.do_something(...)` is a coroutine you `await`.

### Step 3: Wire it up to the client

```python
# client.py — inside Infinity.__init__:
self.my_resource = resources.MyResource(client=self)

# client.py — inside AsyncInfinity.__init__:
self.my_resource = resources.AsyncMyResource(client=self)
```

And export from `resources/__init__.py` and `infinity/__init__.py` as needed.

## Rules for writing `@api_method` bodies

### DO

- **Use `await self._a_post(...)`, `await self._a_get(...)`, etc.** for all
  HTTP calls. These are the async-safe helpers that work correctly on both
  sync and async resource bases.
- **Use `await self._sleep(seconds)`** instead of `time.sleep()` if you need
  to wait (e.g. polling). On sync resources this delegates to `time.sleep`;
  on async resources it uses `anyio.sleep`.
- **Keep method signatures identical** between what sync and async callers
  would expect — the mixin is the single source of truth.
- **Use type hints and docstrings** — they propagate to both concrete classes
  automatically.
- **Return Pydantic models or plain dicts** — avoid returning httpx internals.

### DON'T

- **Don't use `self._post(...)` or `self._client.post(...)` in new code.**
  These work but are sync-only shortcuts kept for backward compatibility with
  older PredictionStudio resources. New code should always use the `_a_*`
  async helpers.
- **Don't use `asyncio.run()` or `loop.run_until_complete()`** — the
  framework handles sync/async bridging via `_run_sync` and `anyio`.
- **Don't import `asyncio`** in resource files unless you have a specific
  need beyond what `anyio` provides.

## Available HTTP helpers

Inside an `@api_method` body, use these on `self`:

| Method | HTTP verb | Notes |
|---|---|---|
| `await self._a_get(endpoint, **params)` | GET | Query params as kwargs |
| `await self._a_post(endpoint, data=..., **params)` | POST | `data` is JSON-serialized |
| `await self._a_put(endpoint, data=..., **params)` | PUT | |
| `await self._a_patch(endpoint, data=..., **params)` | PATCH | |
| `await self._a_delete(endpoint, **params)` | DELETE | |
| `await self._sleep(seconds)` | — | Async-safe sleep |

## Custom exception hooks

If your resource needs custom error handling, define a static method and
install it in `__init__`:

```python
class _MyResourceMixin:
    def _install_exception_hook(self):
        self.custom_exception_hook = self._custom_exception_hook

    @staticmethod
    def _custom_exception_hook(base_url, endpoint, params, response):
        if response.status_code == 418:
            return TeapotError(base_url, endpoint, params, response)
        return None  # Fall through to default handling


class MyResource(_MyResourceMixin, SyncAPIResource):
    def __init__(self, client):
        super().__init__(client)
        self._install_exception_hook()
```

## PredictionStudio architecture

PredictionStudio is the largest resource and uses the write-once mixin
pattern throughout. The code is version-gated:

```
prediction_studio/
├── base.py              # Mixin + sync/async base class pairs
├── types.py             # Enums and type aliases
├── local_model_utils.py # ONNX/PMML/H2O model handling (no API calls)
├── __init__.py          # get(version) / get_async(version) dispatch
├── v24_1/               # Pega 24.1 resources
│   ├── prediction_studio.py
│   ├── prediction.py
│   └── repository.py
└── v24_2/               # Pega 24.2 resources (extends v24_1)
    ├── prediction_studio.py
    ├── prediction.py
    ├── model.py
    ├── champion_challenger.py
    ├── datamart_export.py
    ├── repository.py
    └── model_upload.py   # Pure data classes, no client dependency
```

### Methods that cannot be write-once

Some methods construct version-specific objects (`Model` vs `AsyncModel`)
or return `PaginatedList` vs `AsyncPaginatedList`. These are defined
separately in the sync and async concrete classes rather than in the mixin.
When modifying one, mirror the change in the other.

### Internal helpers and `@api_method` nesting

If a mixin helper calls another `@api_method` method, the sync path would
nest `_run_sync` calls and crash. To avoid this:

- Public-facing API methods get `@api_method`.
- Internal helpers are plain `async def` (no decorator) and call `_a_*`
  HTTP helpers directly instead of going through other `@api_method` methods.
- Use `_maybe_await()` from `_resource.py` when calling a method that is
  sync on `Infinity` but async on `AsyncInfinity`.

## Known limitations

- **PegaOAuth is sync-only**: The `PegaOAuth.token` property uses
  `httpx.post()` for token refresh, which blocks the event loop in async
  contexts. Needs an `AsyncPegaOAuth` variant or `anyio.to_thread.run_sync()`
  wrapper.

See [MIGRATION.md](MIGRATION.md) for a full list of changes and upgrade notes.
