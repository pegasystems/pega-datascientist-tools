from __future__ import annotations

from ..v24_2.repository import AsyncRepository as AsyncRepositoryPrevious
from ..v24_2.repository import Repository as RepositoryPrevious


class _Repositoryv26_1Mixin:
    """v26 Repository data — defined once.

    Add new or overridden methods here.
    """


class Repository(_Repositoryv26_1Mixin, RepositoryPrevious):
    pass


class AsyncRepository(_Repositoryv26_1Mixin, AsyncRepositoryPrevious):
    pass
