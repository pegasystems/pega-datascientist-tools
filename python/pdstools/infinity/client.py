__all__ = ["Infinity", "AsyncInfinity"]


from importlib.util import find_spec

from ..utils.namespaces import MissingDependenciesException
from .internal._base_client import AsyncAPIClient, SyncAPIClient

instructions = """To initialize the Infinity API client, please use one of the constructor methods:
`Infinity.from_basic_auth` or "`Infinity.from_client_credentials`.


`.from_basic_auth` takes in the user credentials you use to login to Infinity,
and it assumes the same permissions as the operator with which you've logged in.

`.from_client_credentials` uses an OAuth credentials file,
which you can create by first going to Dev Studio, then navigating to
Create -> Security -> OAuth 2.0 Client Registration. If the OAuth Client does not show up in the
Security tab, this is likely due to insufficient permissions for your operator.
"""


class Infinity(SyncAPIClient):
    """The Pega Infinity DX API client"""

    version: str

    def __init__(self, *args, **kwargs):
        if not args and not kwargs:
            raise RuntimeError(instructions)

        if not find_spec("pydantic"):
            raise MissingDependenciesException(
                ["pydantic"], "the Infinity API client", "api"
            )

        super().__init__(*args, **kwargs)

        self.version = kwargs.get("pega_version") or self._infer_version()

        from . import resources

        self.knowledge_buddy = resources.KnowledgeBuddy(client=self)

        self.prediction_studio = resources.prediction_studio.get(self.version)(
            client=self
        )


class AsyncInfinity(AsyncAPIClient):  # pragma: no cover
    version: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version = kwargs.get("pega_version") or self._infer_version()

        # self.PredictionStudio = resources.PredictionStudio.get_async(self.version)(
        #     client=self
        # )
