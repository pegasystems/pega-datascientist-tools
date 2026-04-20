__all__ = ["AsyncInfinity", "Infinity"]

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
                ["pydantic"],
                "the Infinity API client",
                "api",
            )

        super().__init__(*args, **kwargs)

        self.version = kwargs.get("pega_version") or self._infer_version(
            on_error="ignore",
        )

        from . import resources

        self.knowledge_buddy = resources.KnowledgeBuddy(client=self)
        if self.version:
            self.prediction_studio = resources.prediction_studio.get(self.version)(
                client=self,
            )

    _VERSION_DEPENDENT_RESOURCES = frozenset({"prediction_studio"})

    def __getattr__(self, name: str):
        if name in self._VERSION_DEPENDENT_RESOURCES:
            raise AttributeError(
                f"'{name}' is not available because the Pega version could "
                "not be determined. Pass 'pega_version' explicitly when "
                "constructing the client, e.g.:\n"
                "  Infinity.from_client_id_and_secret(..., pega_version='24.2')",
            )
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'",
        )


class AsyncInfinity(AsyncAPIClient):
    """The async Pega Infinity DX API client.

    Provides the same functionality as :class:`Infinity` but with
    native ``async``/``await`` support.  Resources expose ``async def``
    methods that can be awaited directly.
    """

    version: str

    def __init__(self, *args, **kwargs):
        if not args and not kwargs:
            raise RuntimeError(instructions)

        if not find_spec("pydantic"):
            raise MissingDependenciesException(
                ["pydantic"],
                "the Infinity API client",
                "api",
            )

        super().__init__(*args, **kwargs)

        self.version = kwargs.get("pega_version") or self._infer_version(
            on_error="ignore",
        )

        from . import resources

        self.knowledge_buddy = resources.AsyncKnowledgeBuddy(client=self)
        if self.version:
            self.prediction_studio = resources.prediction_studio.get_async(
                self.version,
            )(client=self)

    _VERSION_DEPENDENT_RESOURCES = frozenset({"prediction_studio"})

    def __getattr__(self, name: str):
        if name in self._VERSION_DEPENDENT_RESOURCES:
            raise AttributeError(
                f"'{name}' is not available because the Pega version could "
                "not be determined. Pass 'pega_version' explicitly when "
                "constructing the client, e.g.:\n"
                "  AsyncInfinity.from_client_id_and_secret(..., pega_version='24.2')",
            )
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'",
        )
