from typing import Generic, Iterator, TypeVar, Union, overload

import polars as pl

T = TypeVar("T")


class PaginatedList(Generic[T]):
    """Abstracts pagination of Pega API

    Currently supports format where a 'nextToken' is supplied
    and the next page is retrieved by supplying that token as the
    'pageToken' of the next call to the same URL.

    Can be iterated, indexed or sliced.
    """

    def __init__(
        self,
        content_class,
        client,
        request_method,
        url,
        extra_attribs=None,
        _root=None,
        **kwargs,
    ):
        self._elements = list()

        self._client = client
        self._content_class = content_class
        self._url = url
        self._first_params = kwargs or {}
        self._next_token = True
        self._next_params = kwargs or {}
        self._extra_attribs = extra_attribs or {}
        self._request_method = request_method
        self._root = _root

    @overload
    def __getitem__(self, key: Union[str, int]) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> "_Slice[T]": ...

    def __getitem__(self, index: Union[int, slice, str]) -> Union[T, "_Slice[T]"]:
        assert isinstance(index, (int, slice, str))
        if isinstance(index, int):
            if index < 0:
                raise IndexError("Cannot negative index a PaginatedList")
            self._get_up_to_index(index)
            return self._elements[index]
        elif isinstance(index, slice):
            return self._Slice(self, index)
        else:
            assert (
                "id" in self._content_class
            ), "To pass a string as index for a paginated list, the content class needs an 'id' field."
            for element in self.__iter__():
                if element["id"] == index:
                    return element

        raise IndexError(index)

    @overload
    def get(self, __key: Union[int, str], __default: Union[str, None]) -> T: ...

    @overload
    def get(self, __key: slice, __default: Union[str, None]) -> "_Slice[T]": ...

    def get(
        self,
        __key: Union[int, slice, str, None] = None,
        __default: Union[str, None] = None,
        **kwargs,
    ) -> Union[T, "_Slice[T]"]:
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
                if all(
                    getattr(element, name) == value for name, value in kwargs.items()
                ):
                    return element
        try:
            response = self.__getitem__(__key)
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

    def __repr__(self):
        return f"<PaginatedList of type {self._content_class.__name__}>"

    def _get_next_page(self):
        response = self._client.request(
            self._request_method, self._url, **self._next_params
        )
        self._next_token = response.pop("nextToken", None)
        if self._next_token is not None:
            self._next_params = {"pageToken": self._next_token}

        content = []
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

    def _get_up_to_index(self, index):
        while len(self._elements) <= index and self._has_next():
            self._grow()

    def _grow(self):
        new_elements = self._get_next_page()
        self._elements += new_elements
        return new_elements

    def _has_next(self):
        return self._next_token is not None

    def _is_larger_than(self, index):
        return len(self._elements) > index or self._has_next()

    class _Slice(Generic[T]):
        def __init__(self, the_list: "PaginatedList", the_slice: slice):
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

        def __getitem__(self, index: int):
            i = 0
            for e in self:
                if i == index:
                    return e
                i += 1

        def _finished(self, index):
            return self._stop is not None and index >= self._stop

        def as_df(self):
            return pl.DataFrame(
                getattr(prediction, "_public_dict") for prediction in self
            )
