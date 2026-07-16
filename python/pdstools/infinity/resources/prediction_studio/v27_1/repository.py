from __future__ import annotations

from ..v24_2.repository import AsyncRepository as AsyncRepositoryPrevious
from ..v24_2.repository import Repository as RepositoryPrevious


class _Repositoryv27_1Mixin:
    """v27 Repository data — defined once.

    Add new or overridden methods here.
    """


class Repository(_Repositoryv27_1Mixin, RepositoryPrevious):
    pass


class AsyncRepository(_Repositoryv27_1Mixin, AsyncRepositoryPrevious):
    pass
