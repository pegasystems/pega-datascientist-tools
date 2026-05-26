from __future__ import annotations

from ..v24_2.repository import AsyncRepository as AsyncRepositoryPrevious
from ..v24_2.repository import Repository as RepositoryPrevious


class _Repositoryv25Mixin:
    """v26 Repository data — defined once.

    Add new or overridden methods here.
    """


class Repository(_Repositoryv25Mixin, RepositoryPrevious):
    pass


class AsyncRepository(_Repositoryv25Mixin, AsyncRepositoryPrevious):
    pass
