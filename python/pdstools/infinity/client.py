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

    The Pega version is resolved lazily on first access of
    :pyattr:`version`. When ``pega_version=`` is passed explicitly, no
    HTTP request is ever made; otherwise the first read of
    ``client.version`` (or any version-dependent resource such as
    ``client.prediction_studio``) issues a single
    ``GET /prweb/api/PredictionStudio/v3/predictions/repository`` and
    caches the result on the instance.
    """

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

        self._version: str | None = pega_version
        self._version_resolved: bool = pega_version is not None

        from . import resources

        self.knowledge_buddy = resources.KnowledgeBuddy(client=self)

    @property
    def version(self) -> str | None:
        """The Pega platform version (e.g. ``"24.2"``).

        Resolved lazily on first access by calling the prediction-studio
        repository endpoint. Returns ``None`` if the version could not be
        inferred (e.g. the host is unreachable). Pass ``pega_version=``
        to the constructor or any ``from_*`` classmethod to skip the
        round-trip entirely.
        """
        if not self._version_resolved:
            self._version = self._infer_version(on_error="warn")
            self._version_resolved = True
        return self._version

    _VERSION_DEPENDENT_RESOURCES = frozenset({"prediction_studio"})

    def __getattr__(self, name: str):
        if name in self._VERSION_DEPENDENT_RESOURCES:
            version = self.version
            if version:
                from . import resources

                resource_cls = resources.prediction_studio.get(version)
                if resource_cls is not None:
                    instance = resource_cls(client=self)
                    object.__setattr__(self, name, instance)
                    return instance
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

    The Pega version is resolved lazily on first access of
    :pyattr:`version`. When ``pega_version=`` is passed explicitly, no
    HTTP request is ever made; otherwise the first read of
    ``client.version`` (or any version-dependent resource such as
    ``client.prediction_studio``) issues a single HTTP probe and caches
    the result on the instance.
    """

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

        self._version: str | None = pega_version
        self._version_resolved: bool = pega_version is not None

        from . import resources

        self.knowledge_buddy = resources.AsyncKnowledgeBuddy(client=self)

    @property
    def version(self) -> str | None:
        """The Pega platform version (e.g. ``"24.2"``).

        Resolved lazily on first access. The underlying
        ``_infer_version`` helper bridges to the async HTTP client via a
        blocking portal, so reading ``client.version`` from synchronous
        code is supported. Pass ``pega_version=`` to the constructor or
        any ``from_*`` classmethod to skip the round-trip entirely.
        """
        if not self._version_resolved:
            self._version = self._infer_version(on_error="warn")
            self._version_resolved = True
        return self._version

    _VERSION_DEPENDENT_RESOURCES = frozenset({"prediction_studio"})

    def __getattr__(self, name: str):
        if name in self._VERSION_DEPENDENT_RESOURCES:
            version = self.version
            if version:
                from . import resources

                resource_cls = resources.prediction_studio.get_async(version)
                if resource_cls is not None:
                    instance = resource_cls(client=self)
                    object.__setattr__(self, name, instance)
                    return instance
            raise AttributeError(
                f"'{name}' is not available because the Pega version could "
                "not be determined. Pass 'pega_version' explicitly when "
                "constructing the client, e.g.:\n"
                "  AsyncInfinity.from_client_id_and_secret(..., pega_version='25.1')",
            )
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'",
        )
