# Migration Guide — Async-First Infinity Client

This guide covers changes introduced by the async-first refactor of the
Infinity API client. **The sync API is fully backward-compatible** — no
changes are required for existing synchronous code.

---

## Who needs to read this?

| You are… | Action needed |
|---|---|
| A user of `Infinity` (sync client) | **None.** All existing code works unchanged. |
| A developer adding new resource methods | Read the [Developer Guide](README.md) for the new write-once pattern. |
| Interested in using `AsyncInfinity` | Read the async section below. |

---

## What changed

### New exports

```python
from pdstools import AsyncInfinity  # NEW — async client
from pdstools import Infinity       # unchanged
```

### Forward-compatible version handling

The client no longer crashes on unknown Pega versions. If `pega_version`
is set to a version newer than what pdstools knows about, it falls back to
the latest supported API (currently 24.2) with an info-level log message.

### `prediction_studio` access without version

If the Pega version could not be determined (and wasn't provided explicitly),
accessing `client.prediction_studio` now raises a clear `AttributeError`
with guidance:

```
AttributeError: 'prediction_studio' is not available because the Pega
version could not be determined. Pass 'pega_version' explicitly when
constructing the client, e.g.:
  Infinity.from_client_id_and_secret(..., pega_version='24.2')
```

---

## Sync API — no breaking changes

All public method signatures, return types, and behaviors are identical.
If your code looks like this, it continues to work as-is:

```python
from pdstools import Infinity

client = Infinity.from_client_id_and_secret(
    base_url="https://my-pega.example.com",
    client_id="...",
    client_secret="...",
    pega_version="24.2",
)

ps = client.prediction_studio
predictions = ps.list_predictions()
model = ps.get_model(model_id="...")
model.describe()
```

---

## Async API — new capability

`AsyncInfinity` mirrors the sync client but all resource methods are
native coroutines:

```python
from pdstools import AsyncInfinity

client = AsyncInfinity.from_client_id_and_secret(
    base_url="https://my-pega.example.com",
    client_id="...",
    client_secret="...",
    pega_version="24.2",
)

ps = client.prediction_studio
predictions = await ps.list_predictions()
model = await ps.get_model(model_id="...")
description = await model.describe()
```

### Async iteration

Methods that return paginated lists use `AsyncPaginatedList`, which
supports `async for`:

```python
async for prediction in await ps.list_predictions():
    print(prediction.label)
```

### Known async limitations

- **`PegaOAuth` token refresh is sync-only.** The OAuth token property uses
  synchronous `httpx.post()` under the hood. In async contexts this blocks
  the event loop briefly during token refresh. A future release will add
  `AsyncPegaOAuth` or use `anyio.to_thread.run_sync()`.

---

## Internal changes (for contributors)

### Architecture shift

Resource business logic is now defined **once** in mixin classes as
`async def` methods decorated with `@api_method`. Two thin concrete classes
(sync + async) inherit from the mixin and the appropriate resource base.

```
Before:
  class Model(SyncAPIResource):
      def describe(self): ...      # sync-only

After:
  class _ModelMixin:
      @api_method
      async def describe(self): ...  # defined once

  class Model(_ModelMixin, SyncAPIResource):    # auto-wrapped to sync
      pass

  class AsyncModel(_ModelMixin, AsyncAPIResource):  # stays async
      pass
```

### `async_base.py` removed

The separate `async_base.py` file has been removed. All base classes
(sync and async variants) now live in `base.py`.

### New internal utilities

| Utility | Location | Purpose |
|---|---|---|
| `@api_method` | `internal/_resource.py` | Marks methods for auto sync-wrapping |
| `_run_sync()` | `internal/_resource.py` | Runs coroutines synchronously via anyio |
| `_maybe_await()` | `internal/_resource.py` | Awaits a value only if it's awaitable |
| `AsyncPaginatedList` | `internal/_pagination.py` | Async iterator for paginated API responses |
| `AsyncAPIClient` | `internal/_base_client.py` | Async HTTP client base |

### HTTP helpers in `@api_method` bodies

New code should use the `_a_*` async helpers instead of the legacy sync
shortcuts:

| New (use this) | Old (still works for sync) |
|---|---|
| `await self._a_get(endpoint, **params)` | `self._get(endpoint, **params)` |
| `await self._a_post(endpoint, data=...)` | `self._post(endpoint, data=...)` |
| `await self._a_put(endpoint, data=...)` | `self._put(endpoint, data=...)` |
| `await self._a_patch(endpoint, data=...)` | `self._patch(endpoint, data=...)` |
| `await self._a_delete(endpoint, **params)` | `self._delete(endpoint, **params)` |
| `await self._sleep(seconds)` | `time.sleep(seconds)` |

### Methods defined separately (not write-once)

Some methods cannot use the write-once pattern because they return
type-specific objects (sync `Model` vs async `AsyncModel`) or use
`PaginatedList` vs `AsyncPaginatedList`. These are defined separately
in the sync and async concrete classes:

- `PredictionStudio.list_predictions()` / `AsyncPredictionStudio.list_predictions()`
- `PredictionStudio.list_models()` / `AsyncPredictionStudio.list_models()`
- `PredictionStudio.get_prediction()` / `AsyncPredictionStudio.get_prediction()`
- `PredictionStudio.get_model()` / `AsyncPredictionStudio.get_model()`
- `PredictionStudio.repository()` / `AsyncPredictionStudio.repository()`
- `Prediction.get_champion_challengers()` / `AsyncPrediction.get_champion_challengers()`
- `ChampionChallenger.list_available_models_to_add()` / `AsyncChampionChallenger.list_available_models_to_add()`

These pairs are kept in sync manually — changes to one should be
mirrored in the other.
