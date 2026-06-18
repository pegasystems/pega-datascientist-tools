"""Tests for PaginatedList and AsyncPaginatedList."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import polars as pl
import pytest
from pdstools.infinity.internal._pagination import AsyncPaginatedList, PaginatedList
from pdstools.infinity.internal._resource import AsyncAPIResource, SyncAPIResource
from pdstools.infinity.resources.prediction_studio.base import ChampionChallengerList

# ---------------------------------------------------------------------------
# Minimal content class for pagination tests.
# ---------------------------------------------------------------------------


class _Item(SyncAPIResource):
    """Tiny sync resource used as the content_class for PaginatedList."""

    def __init__(self, client, *, id: str, name: str):
        super().__init__(client=client)
        self.id = id
        self.name = name


class _AsyncItem(AsyncAPIResource):
    """Tiny async resource used as the content_class for AsyncPaginatedList."""

    def __init__(self, client, *, id: str, name: str):
        super().__init__(client=client)
        self.id = id
        self.name = name


class _AsyncCustomIdItem(AsyncAPIResource):
    """Async content class whose id lives on a non-default field name."""

    _id_field = "model_id"

    def __init__(self, client, *, model_id: str, name: str):
        super().__init__(client=client)
        self.model_id = model_id
        self.name = name


class _LabeledItem(SyncAPIResource):
    """Tiny sync resource with both id and label fields."""

    _id_field = "model_id"

    def __init__(self, client, *, model_id: str, label: str, name: str):
        super().__init__(client=client)
        self.model_id = model_id
        self.label = label
        self.name = name


class _AsyncLabeledItem(AsyncAPIResource):
    """Tiny async resource with both id and label fields."""

    _id_field = "model_id"

    def __init__(self, client, *, model_id: str, label: str, name: str):
        super().__init__(client=client)
        self.model_id = model_id
        self.label = label
        self.name = name


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_page(items, next_token=None):
    """Build a response dict that looks like a paginated API response."""
    page = {"items": list(items)}
    if next_token is not None:
        page["nextToken"] = next_token
    return page


def _single_page_client():
    """Client that returns one page with 3 items."""
    client = MagicMock()
    client.request.return_value = _make_page(
        [
            {"id": "A", "name": "Alice"},
            {"id": "B", "name": "Bob"},
            {"id": "C", "name": "Charlie"},
        ],
    )
    return client


def _multi_page_client():
    """Client that returns two pages: first with token, second without."""
    client = MagicMock()
    client.request.side_effect = [
        _make_page(
            [{"id": "A", "name": "Alice"}, {"id": "B", "name": "Bob"}],
            next_token="page2",
        ),
        _make_page(
            [{"id": "C", "name": "Charlie"}],
        ),
    ]
    return client


def _empty_page_client():
    """Client that returns an empty page."""
    client = MagicMock()
    client.request.return_value = {"items": []}
    return client


def _labeled_page():
    return _make_page(
        [
            {"model_id": "m1", "label": "Alpha", "name": "First"},
            {"model_id": "m2", "label": "Beta", "name": "Second"},
            {"model_id": "m3", "label": "Gamma", "name": "Third"},
        ],
    )


def _ambiguous_labeled_page():
    return _make_page(
        [
            {"model_id": "m1", "label": "Shared", "name": "First"},
            {"model_id": "m2", "label": "Shared", "name": "Second"},
            {"model_id": "m3", "label": "Unique", "name": "Third"},
        ],
    )


# ---------------------------------------------------------------------------
# PaginatedList (sync)
# ---------------------------------------------------------------------------


class TestPaginatedList:
    def test_iteration(self):
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        items = list(pl_)
        assert len(items) == 3
        assert items[0].id == "A"
        assert items[1].name == "Bob"
        assert items[2].id == "C"

    def test_getitem_by_index(self):
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        item = pl_[1]
        assert item.id == "B"
        assert item.name == "Bob"

    def test_getitem_by_string_id(self):
        """String indexing looks up by the id field (defaults to ``id``)."""
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        item = pl_["B"]
        assert item.id == "B"
        assert item.name == "Bob"

    def test_getitem_by_string_id_missing_raises(self):
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        with pytest.raises(KeyError, match="Available ids"):
            pl_["Z"]

    def test_getitem_negative_index_raises(self):
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        with pytest.raises(IndexError, match="Cannot negative index"):
            pl_[-1]

    def test_getitem_slice(self):
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        sliced = pl_[0:2]
        items = list(sliced)
        assert len(items) == 2
        assert items[0].id == "A"
        assert items[1].id == "B"

    def test_get_by_index(self):
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        item = pl_.get(0)
        assert item.id == "A"

    def test_get_by_string_id(self):
        """get() by string resolves via the id field."""
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        result = pl_.get("C", "not_found")
        assert result.id == "C"
        assert result.name == "Charlie"

    def test_get_default_on_missing(self):
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        result = pl_.get("Z", "fallback")
        assert result == "fallback"

    def test_get_by_kwargs(self):
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        item = pl_.get(name="Charlie")
        assert item.id == "C"

    def test_multi_page_iteration(self):
        client = _multi_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        items = list(pl_)
        assert len(items) == 3
        assert items[2].id == "C"
        # Should have been called twice (two pages).
        assert client.request.call_count == 2

    def test_multi_page_token_passing(self):
        client = _multi_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        list(pl_)  # consume all pages
        # Second call should include pageToken.
        second_call = client.request.call_args_list[1]
        assert second_call == call("get", "/items", pageToken="page2")

    def test_empty_page(self):
        client = _empty_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        items = list(pl_)
        assert len(items) == 0

    def test_missing_root_key_raises(self):
        client = MagicMock()
        client.request.return_value = {"wrong_key": []}
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        with pytest.raises(ValueError, match="Json format unexpected"):
            list(pl_)

    def test_empty_dict_response_yields_empty(self):
        """An empty ``{}`` body (no items configured) is an empty page, not an
        error — distinct from a populated dict missing the root key.
        """
        client = MagicMock()
        client.request.return_value = {}
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        assert list(pl_) == []
        assert pl_.as_df().is_empty()

    def test_repr(self):
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        assert repr(pl_) == "<PaginatedList of type _Item>"

    def test_extra_attribs_merged(self):
        """extra_attribs should be merged into each element dict before
        constructing the content class.
        """
        client = MagicMock()
        client.request.return_value = {"items": [{"id": "X", "name": "Extra"}]}

        class _ItemExt(SyncAPIResource):
            def __init__(self, client, *, id, name, bonus=None):
                super().__init__(client=client)
                self.id = id
                self.name = name
                self.bonus = bonus

        pl_ = PaginatedList(
            _ItemExt,
            client,
            "get",
            "/items",
            extra_attribs={"bonus": "yes"},
            _root="items",
        )
        items = list(pl_)
        assert items[0].bonus == "yes"

    def test_as_df_via_slice(self):
        """_Slice.as_df should produce a polars DataFrame."""
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        sliced = pl_[0:3]
        df = sliced.as_df()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert "id" in df.columns
        assert "name" in df.columns

    def test_slice_negative_raises(self):
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        with pytest.raises(IndexError, match="Cannot negative index"):
            pl_[-1:2]

    def test_slice_getitem_index(self):
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        sliced = pl_[0:3]
        item = sliced[1]
        assert item.id == "B"

    def test_slice_getitem_index_out_of_bounds_raises(self):
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        sliced = pl_[0:3]
        with pytest.raises(IndexError):
            sliced[3]

    def test_contains_by_id(self):
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        assert "B" in pl_
        assert "Z" not in pl_

    def test_keys(self):
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        assert pl_.keys() == ["A", "B", "C"]


class TestLabeledPaginatedList:
    def _client(self, page):
        client = MagicMock()
        client.request.return_value = page
        return client

    def test_getitem_by_label(self):
        pl_ = PaginatedList(_LabeledItem, self._client(_labeled_page()), "get", "/m", _root="items")
        assert pl_["Beta"].model_id == "m2"

    def test_getitem_by_id_with_label_present(self):
        pl_ = PaginatedList(_LabeledItem, self._client(_labeled_page()), "get", "/m", _root="items")
        assert pl_["m1"].label == "Alpha"

    def test_getitem_by_label_ambiguous_raises(self):
        pl_ = PaginatedList(_LabeledItem, self._client(_ambiguous_labeled_page()), "get", "/m", _root="items")
        with pytest.raises(KeyError, match="ambiguous"):
            pl_["Shared"]

    def test_getitem_missing_label_lists_available_labels(self):
        pl_ = PaginatedList(_LabeledItem, self._client(_labeled_page()), "get", "/m", _root="items")
        with pytest.raises(KeyError, match="Available labels: \\['Alpha', 'Beta', 'Gamma'\\]"):
            pl_["Missing"]

    def test_contains_get_and_keys_use_label_fallback(self):
        pl_ = PaginatedList(_LabeledItem, self._client(_labeled_page()), "get", "/m", _root="items")
        assert "Gamma" in pl_
        assert pl_.get("Alpha").name == "First"
        assert pl_.get("Missing", "fallback") == "fallback"
        assert pl_.keys() == ["Alpha", "Beta", "Gamma"]


# ---------------------------------------------------------------------------
# Custom _id_field resolution
# ---------------------------------------------------------------------------


class _CustomIdItem(SyncAPIResource):
    """Content class whose id lives on a non-default field name."""

    _id_field = "model_id"

    def __init__(self, client, *, model_id: str, name: str):
        super().__init__(client=client)
        self.model_id = model_id
        self.name = name


class TestCustomIdField:
    def _client(self):
        client = MagicMock()
        client.request.return_value = {
            "items": [
                {"model_id": "m1", "name": "First"},
                {"model_id": "m2", "name": "Second"},
            ],
        }
        return client

    def test_getitem_uses_id_field(self):
        pl_ = PaginatedList(_CustomIdItem, self._client(), "get", "/m", _root="items")
        assert pl_["m2"].name == "Second"

    def test_get_uses_id_field(self):
        pl_ = PaginatedList(_CustomIdItem, self._client(), "get", "/m", _root="items")
        assert pl_.get("m1").name == "First"

    def test_contains_uses_id_field(self):
        pl_ = PaginatedList(_CustomIdItem, self._client(), "get", "/m", _root="items")
        assert "m1" in pl_
        assert "nope" not in pl_

    def test_keys_uses_id_field(self):
        pl_ = PaginatedList(_CustomIdItem, self._client(), "get", "/m", _root="items")
        assert pl_.keys() == ["m1", "m2"]


# ---------------------------------------------------------------------------
# AsyncPaginatedList
# ---------------------------------------------------------------------------


class TestAsyncPaginatedList:
    def _make_async_client(self, pages):
        """Build an async client mock that returns pages in sequence."""
        client = MagicMock()
        client.request = AsyncMock(side_effect=pages)
        return client

    @pytest.mark.asyncio
    async def test_async_iteration(self):
        client = self._make_async_client(
            [
                _make_page(
                    [{"id": "A", "name": "Alice"}, {"id": "B", "name": "Bob"}],
                ),
            ],
        )
        apl = AsyncPaginatedList(_AsyncItem, client, "get", "/items", _root="items")
        items = []
        async for item in apl:
            items.append(item)
        assert len(items) == 2
        assert items[0].id == "A"
        assert items[1].name == "Bob"

    @pytest.mark.asyncio
    async def test_async_collect(self):
        client = self._make_async_client(
            [
                _make_page(
                    [{"id": "A", "name": "Alice"}, {"id": "B", "name": "Bob"}],
                ),
            ],
        )
        apl = AsyncPaginatedList(_AsyncItem, client, "get", "/items", _root="items")
        items = await apl.collect()
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_async_multi_page(self):
        client = self._make_async_client(
            [
                _make_page(
                    [{"id": "A", "name": "Alice"}],
                    next_token="p2",
                ),
                _make_page(
                    [{"id": "B", "name": "Bob"}],
                ),
            ],
        )
        apl = AsyncPaginatedList(_AsyncItem, client, "get", "/items", _root="items")
        items = await apl.collect()
        assert len(items) == 2
        assert items[1].id == "B"
        assert client.request.await_count == 2

    @pytest.mark.asyncio
    async def test_async_get_by_index(self):
        client = self._make_async_client(
            [
                _make_page(
                    [{"id": "A", "name": "Alice"}, {"id": "B", "name": "Bob"}],
                ),
            ],
        )
        apl = AsyncPaginatedList(_AsyncItem, client, "get", "/items", _root="items")
        item = await apl.get(1)
        assert item.id == "B"

    @pytest.mark.asyncio
    async def test_async_get_by_string_id(self):
        client = self._make_async_client(
            [
                _make_page(
                    [{"id": "A", "name": "Alice"}, {"id": "B", "name": "Bob"}],
                ),
            ],
        )
        apl = AsyncPaginatedList(_AsyncItem, client, "get", "/items", _root="items")
        item = await apl.get("B")
        assert item.name == "Bob"

    @pytest.mark.asyncio
    async def test_async_get_default(self):
        client = self._make_async_client(
            [
                _make_page([{"id": "A", "name": "Alice"}]),
            ],
        )
        apl = AsyncPaginatedList(_AsyncItem, client, "get", "/items", _root="items")
        result = await apl.get("Z", "fallback")
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_async_get_by_kwargs(self):
        client = self._make_async_client(
            [
                _make_page(
                    [{"id": "A", "name": "Alice"}, {"id": "B", "name": "Bob"}],
                ),
            ],
        )
        apl = AsyncPaginatedList(_AsyncItem, client, "get", "/items", _root="items")
        item = await apl.get(name="Bob")
        assert item.id == "B"

    @pytest.mark.asyncio
    async def test_async_as_df(self):
        client = self._make_async_client(
            [
                _make_page(
                    [{"id": "A", "name": "Alice"}, {"id": "B", "name": "Bob"}],
                ),
            ],
        )
        apl = AsyncPaginatedList(_AsyncItem, client, "get", "/items", _root="items")
        df = await apl.as_df()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert "id" in df.columns

    @pytest.mark.asyncio
    async def test_async_empty_page(self):
        client = self._make_async_client([_make_page([])])
        apl = AsyncPaginatedList(_AsyncItem, client, "get", "/items", _root="items")
        items = await apl.collect()
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_async_missing_root_raises(self):
        client = MagicMock()
        client.request = AsyncMock(return_value={"wrong": []})
        apl = AsyncPaginatedList(_AsyncItem, client, "get", "/items", _root="items")
        with pytest.raises(ValueError, match="Json format unexpected"):
            await apl.collect()

    @pytest.mark.asyncio
    async def test_async_empty_dict_response_yields_empty(self):
        """An empty ``{}`` body is an empty page, not an error (async)."""
        client = MagicMock()
        client.request = AsyncMock(return_value={})
        apl = AsyncPaginatedList(_AsyncItem, client, "get", "/items", _root="items")
        assert await apl.collect() == []
        df = await apl.as_df()
        assert df.is_empty()

    def test_async_repr(self):
        client = MagicMock()
        apl = AsyncPaginatedList(_AsyncItem, client, "get", "/items", _root="items")
        assert repr(apl) == "<AsyncPaginatedList of type _AsyncItem>"

    @pytest.mark.asyncio
    async def test_async_token_passing(self):
        client = self._make_async_client(
            [
                _make_page([{"id": "A", "name": "Alice"}], next_token="tok"),
                _make_page([{"id": "B", "name": "Bob"}]),
            ],
        )
        apl = AsyncPaginatedList(_AsyncItem, client, "get", "/items", _root="items")
        await apl.collect()
        second_call = client.request.call_args_list[1]
        assert second_call == call("get", "/items", pageToken="tok")

    @pytest.mark.asyncio
    async def test_async_keys(self):
        client = self._make_async_client(
            [
                _make_page(
                    [{"id": "A", "name": "Alice"}, {"id": "B", "name": "Bob"}],
                ),
            ],
        )
        apl = AsyncPaginatedList(_AsyncItem, client, "get", "/items", _root="items")
        assert await apl.keys() == ["A", "B"]

    @pytest.mark.asyncio
    async def test_async_get_uses_id_field(self):
        client = MagicMock()
        client.request = AsyncMock(
            return_value={
                "items": [
                    {"model_id": "m1", "name": "First"},
                    {"model_id": "m2", "name": "Second"},
                ],
            },
        )
        apl = AsyncPaginatedList(_AsyncCustomIdItem, client, "get", "/m", _root="items")
        item = await apl.get("m2")
        assert item.name == "Second"


class TestAsyncLabeledPaginatedList:
    def _client(self, page):
        client = MagicMock()
        client.request = AsyncMock(return_value=page)
        return client

    @pytest.mark.asyncio
    async def test_async_get_by_label(self):
        apl = AsyncPaginatedList(_AsyncLabeledItem, self._client(_labeled_page()), "get", "/m", _root="items")
        assert (await apl.get("Beta")).model_id == "m2"

    @pytest.mark.asyncio
    async def test_async_get_by_id_with_label_present(self):
        apl = AsyncPaginatedList(_AsyncLabeledItem, self._client(_labeled_page()), "get", "/m", _root="items")
        assert (await apl.get("m1")).label == "Alpha"

    @pytest.mark.asyncio
    async def test_async_get_by_label_ambiguous_returns_default(self):
        apl = AsyncPaginatedList(
            _AsyncLabeledItem,
            self._client(_ambiguous_labeled_page()),
            "get",
            "/m",
            _root="items",
        )
        assert await apl.get("Shared", "fallback") == "fallback"

    @pytest.mark.asyncio
    async def test_async_get_missing_label_returns_default(self):
        apl = AsyncPaginatedList(_AsyncLabeledItem, self._client(_labeled_page()), "get", "/m", _root="items")
        assert await apl.get("Missing", "fallback") == "fallback"

    @pytest.mark.asyncio
    async def test_async_keys_prefer_labels(self):
        apl = AsyncPaginatedList(_AsyncLabeledItem, self._client(_labeled_page()), "get", "/m", _root="items")
        assert await apl.keys() == ["Alpha", "Beta", "Gamma"]


def _cc(active_label: str, challenger_label: str | None = None):
    challenger_model = None
    if challenger_label is not None:
        challenger_model = SimpleNamespace(label=challenger_label)
    return SimpleNamespace(
        active_model=SimpleNamespace(label=active_label),
        challenger_model=challenger_model,
    )


class TestChampionChallengerList:
    def test_string_lookup_uses_challenger_label(self):
        ccs = ChampionChallengerList([_cc("Champion A", "Challenger A"), _cc("Champion B", "Challenger B")])
        assert ccs["Challenger B"].active_model.label == "Champion B"

    def test_string_lookup_falls_back_to_active_label(self):
        ccs = ChampionChallengerList([_cc("Champion Only"), _cc("Champion B", "Challenger B")])
        assert ccs["Champion Only"].active_model.label == "Champion Only"

    def test_string_lookup_ambiguous_raises(self):
        ccs = ChampionChallengerList([_cc("Champion A", "Shared"), _cc("Champion B", "Shared")])
        with pytest.raises(KeyError, match="ambiguous"):
            ccs["Shared"]

    def test_missing_label_lists_available_labels(self):
        ccs = ChampionChallengerList([_cc("Champion Only"), _cc("Champion B", "Challenger B")])
        with pytest.raises(
            KeyError,
            match="Available challenger labels: \\['Champion Only', 'Challenger B'\\]",
        ):
            ccs["Missing"]

    def test_contains_keys_and_integer_indexing(self):
        ccs = ChampionChallengerList([_cc("Champion Only"), _cc("Champion B", "Challenger B")])
        assert "Champion Only" in ccs
        assert "Missing" not in ccs
        assert ccs.keys() == ["Champion Only", "Challenger B"]
        assert ccs[1].challenger_model.label == "Challenger B"
