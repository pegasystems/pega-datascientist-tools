"""Tests for PaginatedList and AsyncPaginatedList."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call

import polars as pl
import pytest
from pdstools.infinity.internal._pagination import AsyncPaginatedList, PaginatedList
from pdstools.infinity.internal._resource import AsyncAPIResource, SyncAPIResource

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

    def test_getitem_by_string_id_requires_dict_content_class(self):
        """String indexing expects the content class to support `in` and dict-like
        item access. With SyncAPIResource subclasses it raises TypeError.
        """
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        with pytest.raises(TypeError):
            pl_["B"]

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

    def test_get_by_string_returns_default_for_resource_classes(self):
        """get() by string falls back to default because __getitem__ for string
        fails with resource classes that don't support dict-like 'in' checks.
        """
        client = _single_page_client()
        pl_ = PaginatedList(_Item, client, "get", "/items", _root="items")
        result = pl_.get("C", "not_found")
        assert result == "not_found"

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
