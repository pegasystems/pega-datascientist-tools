from typing import Literal, Optional, TypedDict

import httpx
from pydantic import AliasChoices, BaseModel, Field, Json

from ...internal._exceptions import InternalServerError, InvalidInputs, PegaException
from ...internal._resource import AsyncAPIResource, SyncAPIResource, api_method


class TextInput(TypedDict):
    name: str
    value: str


class FilterAttributes(TypedDict):
    name: str
    values: list[dict[Literal["value"], str]]


class AttributeValue(BaseModel):
    value: str


class Attribute(BaseModel):
    values: list[AttributeValue]
    name: str


class Chunk(BaseModel):
    attributes: list[Attribute]
    content: str


class SearchResultValue(BaseModel):
    chunks: list[Chunk]


class SearchResult(BaseModel):
    name: str
    value: Json[SearchResultValue] | SearchResultValue | str


class BuddyResponse(BaseModel):
    question_id: str = Field(validation_alias=AliasChoices("questionID", "question_id"))
    answer: str
    status: str
    search_results: Optional[list[SearchResult]] = Field(
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

    def _install_exception_hook(self):
        self.custom_exception_hook = self._custom_exception_hook

    @api_method
    async def question(
        self,
        question: str,
        buddy: str,
        include_search_results: bool = False,
        question_source: Optional[str] = None,
        question_tag: Optional[str] = None,
        additional_text_inputs: Optional[list[TextInput]] = None,
        filter_attributes: Optional[list[FilterAttributes]] = None,
        user_name: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> BuddyResponse:
        """Send a question to the Knowledge Buddy.

        Parameters
        ----------
        question: str: (Required)
            Input the question.
        buddy: str (Required)
            Input the buddy name.
            If you do not have the required role to access the buddy,
            an access error will be displayed.
        include_search_results: bool (Default: False)
            If set to true, this property returns chunks of data related to each
            SEARCHRESULTS information variable that is defined for the Knowledge Buddy,
            which is the same information that is returned during a semantic search.
        question_source: str (Optional)
            Input a source for the question based on the use case.
            This information can be used for reporting purposes.
        question_tag: str (Optional)
            Input a tag for the question based on the use case.
            This information can be used for reporting purposes.
        additional_text_inputs: list[TextInput]: (Optional)
            Input the search variable values, where key is the search variable name
            and value is the data that replaces the variable.
            Search variables are defined in the Information section of the Knowledge Buddy.
        filter_attributes: list[FilterAttributes]: (Optional)
            Input the filter attributes to get the filtered chunks from the vector database.
            User-defined attributes ingested with content can be used as filters.
            Filters are recommended to improve the semantic search performance.
            Database indexes can be used further to enhance the search.

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
        comments: Optional[str] = None,
    ):
        """Capture feedback for a question asked to the Knowledge Buddy.

        Parameters
        ----------
        question_id: str: (Required)
            The Knowledge Buddy case Id that is required to capture the feedback.
        helpful: str (Optional)
            Was this comment helpful? Valid values are Yes, No and Unsure.
            Empty value defaults to Unsure.
        comments: str (Optional)
            Text of the comment.

        """
        response = await self._a_put(
            "/prweb/api/knowledgebuddy/v1/question/feedback",
            data=dict(
                questionID=question_id,
                helpful=helpful,
                comments=comments,
            ),
        )
        return response

    @staticmethod
    def _custom_exception_hook(
        base_url: httpx.URL | str,
        endpoint: str,
        params: dict,
        response: httpx.Response,
    ) -> None | Exception:
        if "Buddy is not available to ask questions." in response.text:
            return UnavailableBuddyError(base_url, endpoint, params, response)
        if response.status_code == 401 or response.status_code == 403:
            return NoAPIAccessError(base_url, endpoint, params, response)
        if response.status_code == 400:
            return InvalidInputs(base_url, endpoint, params, response)
        if response.status_code == 500:
            return InternalServerError(base_url, endpoint, params, response)
        return Exception(response.text)


class KnowledgeBuddy(_KnowledgeBuddyMixin, SyncAPIResource):
    def __init__(self, client):
        super().__init__(client)
        self._install_exception_hook()


class AsyncKnowledgeBuddy(_KnowledgeBuddyMixin, AsyncAPIResource):
    def __init__(self, client):
        super().__init__(client)
        self._install_exception_hook()
