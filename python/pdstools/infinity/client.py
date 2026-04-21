__all__ = ["AsyncInfinity", "Infinity"]

from importlib.util import find_spec
from typing import TYPE_CHECKING

from ..utils.namespaces import MissingDependenciesException
from .internal._base_client import AsyncAPIClient, SyncAPIClient

if TYPE_CHECKING:  # pragma: no cover
    import httpx

    from .internal._auth import PegaOAuth


class Infinity(SyncAPIClient):
    """The Pega Infinity DX API client.

    Prefer one of the :py:meth:`from_basic_auth`,
    :py:meth:`from_client_credentials`, or
    :py:meth:`from_client_id_and_secret` constructors over calling
    ``Infinity(...)`` directly — they handle auth construction for you.
    """

    version: str

    def __init__(
        self,
        *,
        base_url: "str | httpx.URL",
        auth: "httpx.Auth | PegaOAuth",
        application_name: str | None = None,
        verify: bool = False,
        pega_version: str | None = None,
        timeout: float = 90,
    ):
        if not find_spec("pydantic"):
            raise MissingDependenciesException(
                ["pydantic"],
                "the Infinity API client",
                "api",
            )

        super().__init__(
            base_url=base_url,
            auth=auth,
            application_name=application_name,
            verify=verify,
            pega_version=pega_version,
            timeout=timeout,
        )

        self.version = pega_version or self._infer_version(on_error="ignore")

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
                "  Infinity.from_client_id_and_secret(..., pega_version='25.1')",
            )
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'",
        )


class AsyncInfinity(AsyncAPIClient):
    """The async Pega Infinity DX API client.

    Provides the same functionality as :class:`Infinity` but with
    native ``async``/``await`` support.  Resources expose ``async def``
    methods that can be awaited directly.

    Prefer one of the :py:meth:`from_basic_auth`,
    :py:meth:`from_client_credentials`, or
    :py:meth:`from_client_id_and_secret` constructors over calling
    ``AsyncInfinity(...)`` directly.
    """

    version: str

    def __init__(
        self,
        *,
        base_url: "str | httpx.URL",
        auth: "httpx.Auth | PegaOAuth",
        application_name: str | None = None,
        verify: bool = False,
        pega_version: str | None = None,
        timeout: float = 90,
    ):
        if not find_spec("pydantic"):
            raise MissingDependenciesException(
                ["pydantic"],
                "the Infinity API client",
                "api",
            )

        super().__init__(
            base_url=base_url,
            auth=auth,
            application_name=application_name,
            verify=verify,
            pega_version=pega_version,
            timeout=timeout,
        )

        self.version = pega_version or self._infer_version(on_error="ignore")

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
                "  AsyncInfinity.from_client_id_and_secret(..., pega_version='25.1')",
            )
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'",
        )
