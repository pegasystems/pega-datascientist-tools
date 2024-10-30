# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import time
from abc import ABC
from typing import TYPE_CHECKING, List, Union

import anyio

if TYPE_CHECKING:
    from ..client import AsyncAPIClient, SyncAPIClient


class SyncAPIResource(ABC):
    _client: SyncAPIClient

    def __init__(self, client: SyncAPIClient) -> None:
        self._client = client
        self._get = client.get
        self._post = client.post
        self._patch = client.patch
        self._put = client.put
        self._delete = client.delete
        self._get_api_list = client.get_api_list

    def _sleep(self, seconds: float) -> None:
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
            return f"{classname}({', '.join([field+'='+format_field(field) for field in fields])})"
        return f"{classname}({', '.join([field+'='+format_field(field) for field in self._public_fields])})"


class AsyncAPIResource(ABC):
    _client: AsyncAPIClient

    def __init__(self, client: AsyncAPIClient) -> None:
        self._client = client
        self._get = client.get
        self._post = client.post
        self._patch = client.patch
        self._put = client.put
        self._delete = client.delete
        self._get_api_list = client.get_api_list

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
                return value
            value = str(value)
            return "'" + value + "'"

        if fields:
            return f"{classname}({', '.join([field+'='+format_field(field) for field in fields])})"
        return f"{classname}({', '.join([field+'='+format_field(field) for field in self.__dict__ if not field.startswith('_')])}"
