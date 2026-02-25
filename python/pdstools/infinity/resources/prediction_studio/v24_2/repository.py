from ..v24_1.repository import AsyncRepository as AsyncRepositoryPrevious
from ..v24_1 import Repository as RepositoryPrevious


class _RepositoryV24_2Mixin:
    """v24.2 Repository data — defined once."""

    def __init__(
        self,
        client,
        type,
        repository_name,
        bucket_name,
        root_path,
        datamart_export_location,
    ):
        super().__init__(client=client, repository_name=repository_name)
        self.type = type
        self.bucket_name = bucket_name
        self.root_path = root_path
        self.datamart_export_location = datamart_export_location

    @property
    def s3_url(self):  # TODO: implement
        return "s3://test"


class Repository(_RepositoryV24_2Mixin, RepositoryPrevious):
    pass


class AsyncRepository(_RepositoryV24_2Mixin, AsyncRepositoryPrevious):
    pass
