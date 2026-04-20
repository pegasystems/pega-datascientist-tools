import re

import pytest
from pdstools.infinity import Infinity
from pdstools.infinity.internal import _auth
from pytest_httpx import HTTPXMock


@pytest.fixture
def mock_auth(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json={"access_token": "ABC", "expires_in": 30})

    return _auth.PegaOAuth(
        "https://pega.com",
        client_id="test_id",
        client_secret="test_secret",
    )


sample_buddy_response = {
    "questionID": "ASK-158002",
    "answer": '{\n  "Title": "What is ADM?",\n  "Answer": "The **Adaptive Decision Manager (ADM)** is a component of the **Pega Platform™** that provides data scientists with tools for creating and managing **Adaptive Models**. These models are **self-learning predictive models** used to predict business outcomes and customer behavior. Key features of ADM include:\\n- **Real-time updates**: Adaptive Models are updated regularly, continuously learning from customer interactions or case outcomes.\\n- **Predictive capabilities**: They help deliver personalized recommendations for customers and optimize workflows.\\n- **Data capture**: The ADM service captures predictor data and outcomes, allowing it to start without any historical information.\\n- **Integration in decision strategies**: Adaptive Models are utilized in decision strategies to enhance the relevance of decisions.",\n  "References": [\n    {\n      "objectId": "objectId",\n      "Title": "Adaptive analytics",\n      "URL": "https://docs.pega.com/bundle/platform/page/platform/decision-management/adaptive-analytics.html"\n    }\n  ]\n}',
    "status": "Results found",
}


def test_buddy_question(mock_auth, httpx_mock: HTTPXMock):
    client = Infinity(base_url="https://pega.com", auth=mock_auth, pega_version="24.2")
    httpx_mock.add_response(url=re.compile(".*question"), json=sample_buddy_response)

    answer = client.knowledge_buddy.question("What is ADM?", "My Buddy")
    assert answer.status == "Results found"
    assert answer.search_results is None
    assert answer.question_id == "ASK-158002"
    assert answer.answer == sample_buddy_response["answer"]

    # httpx_mock.add_response(url=re.compile(".*question"), status_code=401)
    # with pytest.raises(NoAPIAccessError):
    #     client.knowledge_buddy.question("TEST", "TEST_BUDDY")


def test_buddy_feedback(mock_auth, httpx_mock: HTTPXMock):
    client = Infinity(base_url="https://pega.com", auth=mock_auth, pega_version="24.2")
    httpx_mock.add_response(
        url=re.compile(".*feedback"),
        json={"status": "Feedback registered"},
    )
    response = client.knowledge_buddy.feedback(
        "ASK-158002",
        helpful="Yes",
        comments="Correct!",
    )

    assert response == {"status": "Feedback registered"}
