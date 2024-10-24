from ..base import Repository as BaseRepository


class Repository(BaseRepository):
    def __init__(self, client, repository_name: str):
        super().__init__(client=client)
        self.name = repository_name
