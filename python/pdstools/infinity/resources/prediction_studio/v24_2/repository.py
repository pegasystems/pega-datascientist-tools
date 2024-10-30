from ..v24_1 import Repository as RepositoryPrevious


class Repository(RepositoryPrevious):
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
