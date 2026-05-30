from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import rag
from app.main import app
from app.rag import Answer

client = TestClient(app)


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
