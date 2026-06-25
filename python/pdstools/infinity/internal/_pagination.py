from __future__ import annotations

from typing import Any, Generic, TypeVar, cast, overload, TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from collections.abc import AsyncIterator, Iterator

T = TypeVar("T", covariant=True)


class _MissingLookupKeyError(KeyError):
    """Internal KeyError used for missing string-key lookups."""


class _AmbiguousLabelKeyError(KeyError):
    """Internal KeyError used for ambiguous label-based lookups."""


def _resolve_id_field(content_class: Any) -> str:
    """Return the attribute name used for string/``in`` lookups.

    Resources backed by the Pydantic data layer declare ``_id_field`` (e.g.
    ``"prediction_id"``). Anything else falls back to ``"id"``.
    """
    return getattr(content_class, "_id_field", "id")


def _preferred_keys(items: list[Any], id_field: str) -> list[Any]:
    """Return user-facing mapping keys for a resource collection.

    Parameters
    ----------
    items : list[Any]
        Collected resource items.
    id_field : str
        Attribute name used for id-based lookup.

    Returns
    -------
    list[Any]
        Labels when the collection exposes them, otherwise ids.

    """
    labels = [getattr(item, "label", None) for item in items if getattr(item, "label", None) is not None]
    if labels:
        return labels
    return [getattr(item, id_field, None) for item in items]


def _resolve_string_lookup(items: Any, key: str, id_field: str) -> Any:
    """Resolve a string key by id first, then by label.

    Parameters
    ----------
    items : iterable
        Resource items to scan.
    key : str
        String key to resolve.
    id_field : str
        Attribute name used for id-based lookup.

    Returns
    -------
    Any
        The matching resource.

    Raises
    ------
    _AmbiguousLabelKeyError
        If multiple resources share the requested label.
    _MissingLookupKeyError
        If neither an id nor a label match is found.

    """
    label_matches: list[Any] = []
    available_keys: list[Any] = []
    has_labels = False

    for element in items:
        id_value = getattr(element, id_field, None)
        if id_value == key:
            return element

        label_value = getattr(element, "label", None)
        if label_value is not None:
            has_labels = True
            available_keys.append(label_value)
            if label_value == key:
                label_matches.append(element)
        else:
            available_keys.append(id_value)

    if len(label_matches) == 1:
        return label_matches[0]
    if len(label_matches) > 1:
        raise _AmbiguousLabelKeyError(
            f"Label {key!r} is ambiguous; matched {len(label_matches)} resources. Use the id instead.",
        )

    visible_keys = available_keys[:5]
    key_type = "labels" if has_labels else "ids"
    raise _MissingLookupKeyError(
        f"{key!r} was not found. Available {key_type}: {visible_keys}.",
    )


def _frame_from_resources(content_class: Any, items: Any) -> pl.DataFrame:
    """Build a DataFrame from resource items with a locked schema when available.

    Resources backed by the Pydantic data layer expose ``_public_schema()``,
    which yields a stable column set and dtypes so that empty results,
    all-null optionals, and forward-compatible extra fields don't change the
    output schema. Resources that have not yet migrated fall back to Polars'
    value inference (``schema=None``).
    """
    schema_fn = getattr(content_class, "_public_schema", None)
    schema = cast(
        "Mapping[str, Any] | Sequence[str | tuple[str, Any]] | None",
        schema_fn() if callable(schema_fn) else None,
    )
    return pl.DataFrame((getattr(item, "_public_dict", {}) for item in items), schema=schema)


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

    def __getitem__(self, index: int) -> T:
        if index < 0:
            raise IndexError("Cannot negative index a PaginatedList slice")
        i: int = 0
        for e in self:
            if i == index:
                return e
            i += 1
        raise IndexError(index)

    def _finished(self, index: int) -> bool:
        return self._stop is not None and index >= self._stop

    def as_df(self) -> pl.DataFrame:
        return _frame_from_resources(self._list._content_class, self)


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
    def __getitem__(self, key: slice) -> _Slice[T]: ...

    def __getitem__(self, key: int | slice | str) -> T | _Slice[T]:
        assert isinstance(key, (int, slice, str))
        if isinstance(key, int):
            if key < 0:
                raise IndexError("Cannot negative index a PaginatedList")
            self._get_up_to_index(key)
            return self._elements[key]
        if isinstance(key, slice):
            return _Slice(self, key)
        id_field = _resolve_id_field(self._content_class)
        return cast("T", _resolve_string_lookup(self.__iter__(), key, id_field))

    def __contains__(self, key: object) -> bool:
        """Perform mapping-style membership tests by id, then label.

        ``"PREDICT_X" in client.prediction_studio.list_predictions()`` walks the
        list (fetching pages as needed), first comparing each element's id field
        and then falling back to ``label`` when the resource exposes one.
        """
        if not isinstance(key, str):
            return False
        id_field = _resolve_id_field(self._content_class)
        try:
            _resolve_string_lookup(iter(self), key, id_field)
        except _MissingLookupKeyError:
            return False
        return True

    def keys(self) -> list[Any]:
        """Return mapping-style keys for every element.

        String lookups try ids first and then labels, so this returns labels
        when available and otherwise falls back to ids.
        """
        id_field = _resolve_id_field(self._content_class)
        return _preferred_keys(list(self), id_field)

    @overload
    def get(self, __key: int | str, __default: str | None = None) -> T: ...

    @overload
    def get(self, __key: slice, __default: str | None = None) -> _Slice[T]: ...

    @overload
    def get(self, __key: None = None, __default: str | None = None, **kwargs: Any) -> T | None: ...

    def get(
        self,
        __key: int | slice | str | None = None,
        __default: str | None = None,
        **kwargs: Any,
    ) -> T | _Slice[T] | Any:
        """Returns the specified key or default.

        If a string is provided as key, lookup first uses the content class id
        field (``_id_field``, defaulting to ``"id"``) and then falls back to
        ``label`` when available.

        Parameters
        ----------
        __key : int | slice | str
            Can be an int (index), slice (start:end), or string (id/label)
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
        if __key is None:
            return __default
        try:
            response = self.__getitem__(__key)
            if not response:
                raise ValueError(__key)
            return response
        except (IndexError, KeyError, ValueError, AttributeError, TypeError):
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

    def as_df(self) -> pl.DataFrame:
        """Collect all pages into a polars DataFrame."""
        return _frame_from_resources(self._content_class, self)

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
            if self._root in response:
                response = response[self._root]
            elif not response:
                # An empty body (e.g. ``{}`` when there are no items) is an
                # empty page, not a malformed payload — yield zero elements.
                # A *populated* dict missing the root key is still a genuine
                # format error and falls through to the raise below.
                response = []
            else:
                raise ValueError(
                    f"Json format unexpected, {self._root} not found.",
                )

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
            if self._root in response:
                response = response[self._root]
            elif not response:
                # An empty body (e.g. ``{}`` when there are no items) is an
                # empty page, not a malformed payload — yield zero elements.
                # A *populated* dict missing the root key is still a genuine
                # format error and falls through to the raise below.
                response = []
            else:
                raise ValueError(
                    f"Json format unexpected, {self._root} not found.",
                )

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
        __default: T | None = None,
        **kwargs: Any,
    ) -> T | None:
        """Async version of :meth:`PaginatedList.get`.

        String lookup first uses the content class id field and then falls back
        to ``label`` when available.
        """
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
                    id_field = _resolve_id_field(self._content_class)
                    return cast("T", _resolve_string_lookup(items, __key, id_field))
            except (IndexError, KeyError, ValueError, AttributeError, TypeError):
                pass
        return __default

    async def keys(self) -> list[Any]:
        """Return mapping-style keys for every element.

        String lookups try ids first and then labels, so this returns labels
        when available and otherwise falls back to ids.
        """
        id_field = _resolve_id_field(self._content_class)
        items = await self.collect()
        return _preferred_keys(items, id_field)

    async def as_df(self) -> pl.DataFrame:
        """Collect all pages into a polars DataFrame."""
        items = await self.collect()
        return _frame_from_resources(self._content_class, items)
