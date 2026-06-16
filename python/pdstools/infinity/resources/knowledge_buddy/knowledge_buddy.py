from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import AliasChoices, BaseModel, Field, Json

from ...internal._exceptions import InternalServerError, InvalidInputs, PegaException
from ...internal._resource import AsyncAPIResource, SyncAPIResource, api_method

if TYPE_CHECKING:
    import httpx
    from collections.abc import Callable


class TextInput(TypedDict):
    """Text input payload for Knowledge Buddy search variables."""

    name: str
    value: str


class FilterAttributes(TypedDict):
    """Filter payload for Knowledge Buddy attribute-based search."""

    name: str
    values: list[dict[Literal["value"], str]]


class AttributeValue(BaseModel):
    """Single attribute value returned in a Knowledge Buddy chunk."""

    value: str


class Attribute(BaseModel):
    """Named attribute attached to a Knowledge Buddy chunk."""

    values: list[AttributeValue]
    name: str


class Chunk(BaseModel):
    """A content chunk returned by Knowledge Buddy search results."""

    attributes: list[Attribute]
    content: str


class SearchResultValue(BaseModel):
    """Structured Knowledge Buddy search result payload."""

    chunks: list[Chunk]


class SearchResult(BaseModel):
    """Single Knowledge Buddy search result entry."""

    name: str
    value: Json[SearchResultValue] | SearchResultValue | str


class BuddyResponse(BaseModel):
    """Response returned after asking a Knowledge Buddy question."""

    question_id: str = Field(validation_alias=AliasChoices("questionID", "question_id"))
    answer: str
    status: str
    search_results: list[SearchResult] | None = Field(
        None,
        validation_alias=AliasChoices("searchResults", "search_results"),
    )


class UnavailableBuddyError(PegaException):
    """Request contains invalid inputs"""


class NoAPIAccessError(PegaException):
    """You do not have access to the API. Contact the administrator."""


# ---------------------------------------------------------------------------
# Write-once mixin: business logic defined as async, works for both
# sync (via @api_method auto-wrapping) and async (native coroutine).
# ---------------------------------------------------------------------------


class _KnowledgeBuddyMixin:
    """Knowledge Buddy business logic — defined once."""

    # Declared for mypy — provided by SyncAPIResource / AsyncAPIResource at runtime
    if TYPE_CHECKING:
        _a_post: Callable[..., Any]
        _a_put: Callable[..., Any]
        custom_exception_hook: Callable[..., Exception | None] | None

    def _install_exception_hook(self):
        self.custom_exception_hook = self._custom_exception_hook

    @api_method
    async def question(
        self,
        question: str,
        buddy: str,
        include_search_results: bool = False,
        question_source: str | None = None,
        question_tag: str | None = None,
        additional_text_inputs: list[TextInput] | None = None,
        filter_attributes: list[FilterAttributes] | None = None,
        user_name: str | None = None,
        user_email: str | None = None,
    ) -> BuddyResponse:
        """Send a question to the Knowledge Buddy.

        Parameters
        ----------
        question : str
            Question to send to the buddy.
        buddy : str
            Buddy name to target.
        include_search_results : bool, default False
            Whether to include the supporting search-result chunks in the response.
        question_source : str | None, default None
            Optional source label for reporting.
        question_tag : str | None, default None
            Optional tag for reporting.
        additional_text_inputs : list[TextInput] | None, default None
            Optional search-variable replacements defined on the buddy.
        filter_attributes : list[FilterAttributes] | None, default None
            Optional attribute filters used to narrow vector-database chunks.
        user_name : str | None, default None
            Optional end-user name to forward with the request.
        user_email : str | None, default None
            Optional end-user email to forward with the request.

        Returns
        -------
        BuddyResponse
            Parsed Knowledge Buddy response payload.
        """
        response = await self._a_post(
            "/prweb/api/knowledgebuddy/v1/question",
            data=dict(
                question=question,
                buddy=buddy,
                includeSearchResults=include_search_results,
                questionSource=question_source,
                questionTag=question_tag,
                additionalTextInputs=additional_text_inputs,
                filterAttributes=filter_attributes,
                userName=user_name,
                userEmail=user_email,
            ),
        )
        return BuddyResponse(**response)

    @api_method
    async def feedback(
        self,
        question_id: str,
        helpful: Literal["Yes", "No", "Unsure"] = "Unsure",
        comments: str | None = None,
    ):
        """Capture feedback for a question asked to the Knowledge Buddy.

        Parameters
        ----------
        question_id : str
            Knowledge Buddy case identifier associated with the original question.
        helpful : Literal["Yes", "No", "Unsure"], default "Unsure"
            Whether the response was helpful.
        comments : str | None, default None
            Optional free-text feedback.
        """
        return await self._a_put(
            "/prweb/api/knowledgebuddy/v1/question/feedback",
            data=dict(
                questionID=question_id,
                helpful=helpful,
                comments=comments,
            ),
        )

    @staticmethod
    def _custom_exception_hook(
        base_url: httpx.URL | str,
        endpoint: str,
        params: dict,
        response: httpx.Response,
    ) -> None | Exception:
        if "Buddy is not available to ask questions." in response.text:
            return UnavailableBuddyError(str(base_url), endpoint, params, response)
        if response.status_code == 401 or response.status_code == 403:
            return NoAPIAccessError(str(base_url), endpoint, params, response)
        if response.status_code == 400:
            return InvalidInputs(str(base_url), endpoint, params, response)
        if response.status_code == 500:
            return InternalServerError(str(base_url), endpoint, params, response)
        return Exception(response.text)


class KnowledgeBuddy(_KnowledgeBuddyMixin, SyncAPIResource):
    """Synchronous Knowledge Buddy resource."""

    def __init__(self, client):
        super().__init__(client)
        self._install_exception_hook()


class AsyncKnowledgeBuddy(_KnowledgeBuddyMixin, AsyncAPIResource):
    """Asynchronous Knowledge Buddy resource."""

    def __init__(self, client):
        super().__init__(client)
        self._install_exception_hook()
