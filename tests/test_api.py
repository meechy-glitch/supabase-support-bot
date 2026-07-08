from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import InternalServerError

from app import llm, rag
from app.main import app
from app.rag import Answer

client = TestClient(app)


def _make_503() -> InternalServerError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(503, request=request)
    return InternalServerError("service unavailable", response=response, body=None)


def test_health_get():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_head():
    resp = client.head("/health")
    assert resp.status_code == 200


def test_chat_returns_answer_and_sources():
    fake = Answer(
        answer="To enable RLS, run `ALTER TABLE foo ENABLE ROW LEVEL SECURITY;`.",
        sources=["https://supabase.com/docs/guides/database/postgres/row-level-security"],
    )
    with patch.object(rag, "answer_question", return_value=fake) as m:
        # main.py imports answer_question directly, so patch it there too
        with patch("app.main.answer_question", return_value=fake):
            resp = client.post("/chat", json={"message": "How do I enable RLS?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"].startswith("To enable RLS")
    assert body["sources"] == [
        "https://supabase.com/docs/guides/database/postgres/row-level-security"
    ]


def test_chat_rejects_empty_message():
    resp = client.post("/chat", json={"message": ""})
    assert resp.status_code == 422


@pytest.mark.parametrize("payload", [{}, {"msg": "hi"}, {"message": 5}])
def test_chat_rejects_malformed_payload(payload):
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 422


def test_chat_returns_503_when_groq_keeps_failing():
    # Retrieval embeds locally (sentence-transformers) + queries Chroma; mock both
    # so only the Groq call is exercised.
    failing_create = MagicMock(side_effect=_make_503())
    with (
        patch.object(rag, "embed_text", return_value=[0.0, 0.1, 0.2]),
        patch.object(rag, "query_by_embedding", return_value=[]),
        patch.object(llm._groq.chat.completions, "create", failing_create),
        patch.object(llm.time, "sleep"),  # don't actually back off in the test
    ):
        resp = client.post("/chat", json={"message": "How do I enable RLS?"})

    assert resp.status_code == 503
    assert failing_create.call_count == 3  # three Groq attempts before giving up
    body = resp.json()
    assert body == {
        "answer": "The assistant is briefly unavailable. Please try again in a moment.",
        "sources": [],
    }
