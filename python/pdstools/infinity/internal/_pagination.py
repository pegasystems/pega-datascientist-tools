from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, Generic, TypeVar, overload

import polars as pl

T = TypeVar("T")


class _Slice(Generic[T]):
    """A lazy slice view over a :class:`PaginatedList`."""

    def __init__(self, the_list: PaginatedList[T], the_slice: slice):
        self._list = the_list
        self._start = the_slice.start or 0
        self._stop = the_slice.stop
        self._step = the_slice.step or 1

        if self._start < 0 or self._stop < 0:
            raise IndexError("Cannot negative index a PaginatedList slice")

    def __iter__(self) -> Iterator[T]:
        index = self._start
        while not self._finished(index):
            if self._list._is_larger_than(index):
                try:
                    yield self._list[index]
                except IndexError:
                    return
                index += self._step
            else:
                return

    def __getitem__(self, index: int) -> T | None:
        i: int = 0
        for e in self:
            if i == index:
                return e
            i += 1
        return None

    def _finished(self, index: int) -> bool:
        return self._stop is not None and index >= self._stop

    def as_df(self) -> pl.DataFrame:
        return pl.DataFrame(getattr(prediction, "_public_dict", {}) for prediction in self)


class PaginatedList(Generic[T]):
    """Abstracts pagination of Pega API

    Currently supports format where a 'nextToken' is supplied
    and the next page is retrieved by supplying that token as the
    'pageToken' of the next call to the same URL.

    Can be iterated, indexed or sliced.
    """

    def __init__(
        self,
        content_class: Any,
        client: Any,
        request_method: str,
        url: str,
        extra_attribs: dict[str, Any] | None = None,
        _root: str | None = None,
        **kwargs: Any,
    ):
        self._elements: list[T] = []

        self._client = client
        self._content_class = content_class
        self._url = url
        self._first_params = kwargs or {}
        self._next_token: str | bool | None = True
        self._next_params = kwargs or {}
        self._extra_attribs = extra_attribs or {}
        self._request_method = request_method
        self._root = _root

    @overload
    def __getitem__(self, key: str | int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> _Slice[T]: ...

    def __getitem__(self, index: int | slice | str) -> T | _Slice[T]:
        assert isinstance(index, (int, slice, str))
        if isinstance(index, int):
            if index < 0:
                raise IndexError("Cannot negative index a PaginatedList")
            self._get_up_to_index(index)
            return self._elements[index]
        if isinstance(index, slice):
            return _Slice(self, index)
        assert "id" in self._content_class, (
            "To pass a string as index for a paginated list, the content class needs an 'id' field."
        )
        for element in self.__iter__():
            if element["id"] == index:  # type: ignore[index]
                return element

        raise IndexError(index)

    @overload
    def get(self, __key: int | str, __default: str | None) -> T: ...

    @overload
    def get(self, __key: slice, __default: str | None) -> _Slice[T]: ...

    def get(
        self,
        __key: int | slice | str | None = None,
        __default: str | None = None,
        **kwargs: Any,
    ) -> T | _Slice[T] | Any:
        """Returns the specified key or default.

        If string type provided as key, the content_class needs to be a Pydantic class,
        with an attribute called 'id'.

        Parameters
        ----------
        __key : int | slice | str
            Can be a int (index), slice (start:end), or string (id attribute)
        __default : str | None, optional
            The value to return if none found, by default None

        Returns
        -------
        Any
            The element, or slice of elements. If not found, returns default

        """
        if kwargs:
            for element in self:
                if all(getattr(element, name) == value for name, value in kwargs.items()):
                    return element
        try:
            response = self.__getitem__(__key)  # type: ignore[index]
            if not response:
                raise ValueError(__key)
            return response
        except Exception:
            return __default

    def __iter__(self) -> Iterator[T]:
        for element in self._elements:
            yield element
        while self._has_next():
            new_elements = self._grow()
            for element in new_elements:
                yield element

    def __repr__(self) -> str:
        return f"<PaginatedList of type {self._content_class.__name__}>"

    def _get_next_page(self) -> list[T]:
        response = self._client.request(
            self._request_method,
            self._url,
            **self._next_params,
        )
        self._next_token = response.pop("nextToken", None)
        if self._next_token is not None:
            self._next_params = {"pageToken": self._next_token}

        content: list[T] = []
        if self._root:
            try:
                response = response[self._root]
            except KeyError as e:
                raise ValueError(f"Json format unexpected, {self._root} not found.{e}")

        for element in response:
            if element is not None:
                element.update(self._extra_attribs)
                content.append(self._content_class(client=self._client, **element))

        return content

    def _get_up_to_index(self, index: int) -> None:
        while len(self._elements) <= index and self._has_next():
            self._grow()

    def _grow(self) -> list[T]:
        new_elements = self._get_next_page()
        self._elements += new_elements
        return new_elements

    def _has_next(self) -> bool:
        return self._next_token is not None

    def _is_larger_than(self, index: int) -> bool:
        return len(self._elements) > index or self._has_next()


class AsyncPaginatedList(Generic[T]):
    """Async variant of :class:`PaginatedList`.

    Same constructor interface.  Uses ``await client.request(...)`` and
    exposes ``async for`` iteration via ``__aiter__``.
    """

    def __init__(
        self,
        content_class: Any,
        client: Any,
        request_method: str,
        url: str,
        extra_attribs: dict[str, Any] | None = None,
        _root: str | None = None,
        **kwargs: Any,
    ):
        self._elements: list[T] = []
        self._client = client
        self._content_class = content_class
        self._url = url
        self._first_params = kwargs or {}
        self._next_token: str | bool | None = True
        self._next_params = kwargs or {}
        self._extra_attribs = extra_attribs or {}
        self._request_method = request_method
        self._root = _root

    async def __aiter__(self) -> AsyncIterator[T]:
        for element in self._elements:
            yield element
        while self._has_next():
            new_elements = await self._grow()
            for element in new_elements:
                yield element

    def __repr__(self) -> str:
        return f"<AsyncPaginatedList of type {self._content_class.__name__}>"

    async def _get_next_page(self) -> list[T]:
        response = await self._client.request(
            self._request_method,
            self._url,
            **self._next_params,
        )
        self._next_token = response.pop("nextToken", None)
        if self._next_token is not None:
            self._next_params = {"pageToken": self._next_token}

        content: list[T] = []
        if self._root:
            try:
                response = response[self._root]
            except KeyError as e:
                raise ValueError(f"Json format unexpected, {self._root} not found.{e}")

        for element in response:
            if element is not None:
                element.update(self._extra_attribs)
                content.append(self._content_class(client=self._client, **element))

        return content

    async def _get_up_to_index(self, index: int) -> None:
        while len(self._elements) <= index and self._has_next():
            await self._grow()

    async def _grow(self) -> list[T]:
        new_elements = await self._get_next_page()
        self._elements += new_elements
        return new_elements

    def _has_next(self) -> bool:
        return self._next_token is not None

    def _is_larger_than(self, index: int) -> bool:
        return len(self._elements) > index or self._has_next()

    async def collect(self) -> list[T]:
        """Eagerly fetch all pages and return as a plain list."""
        items: list[T] = []
        async for item in self:
            items.append(item)
        return items

    async def get(
        self,
        __key: int | slice | str | None = None,
        __default: str | None = None,
        **kwargs: Any,
    ) -> T | None:
        """Async version of PaginatedList.get()."""
        if kwargs:
            async for element in self:
                if all(getattr(element, name) == value for name, value in kwargs.items()):
                    return element
        if __key is not None:
            try:
                items = await self.collect()
                if isinstance(__key, int):
                    return items[__key]
                if isinstance(__key, str):
                    for el in items:
                        if getattr(el, "id", None) == __key:
                            return el
            except Exception:
                pass
        return __default  # type: ignore[return-value]

    async def as_df(self) -> pl.DataFrame:
        """Collect all pages into a polars DataFrame."""
        items = await self.collect()
        return pl.DataFrame(getattr(item, "_public_dict", {}) for item in items)
