"""Tests for the_vault/app.py — lifespan wiring and the /health and /ask endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from missions.the_vault import app as app_module
from missions.the_vault.rag import AnswerResponse, Citation


@pytest.fixture
def fake_rag() -> MagicMock:
    rag = MagicMock(name="rag")
    rag.vector_count.return_value = 7
    return rag


@pytest.fixture
def rag_cls(monkeypatch: pytest.MonkeyPatch, fake_rag: MagicMock) -> MagicMock:
    """Patch the Rag class the lifespan bootstraps; yield the mock."""
    mock = MagicMock(name="Rag")
    mock.bootstrap.return_value = fake_rag
    monkeypatch.setattr(app_module, "Rag", mock)
    return mock


def test_lifespan_bootstraps_and_stores_the_rag(rag_cls: MagicMock, fake_rag: MagicMock):
    app = app_module.create_app()

    with TestClient(app):
        rag_cls.bootstrap.assert_called_once_with()
        assert app.state.rag is fake_rag


def test_health_reports_ok_and_vector_count_after_startup(rag_cls: MagicMock):
    app = app_module.create_app()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "vectors": 7}


def test_health_reports_starting_before_the_rag_is_ready(rag_cls: MagicMock):
    # No `with` block => lifespan doesn't run => app.state.rag is unset.
    client = TestClient(app_module.create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "starting", "vectors": 0}
    rag_cls.bootstrap.assert_not_called()


def test_ask_answers_the_question_via_the_rag_handle(rag_cls: MagicMock, fake_rag: MagicMock):
    fake_rag.answer_question.return_value = AnswerResponse(
        answer="Three times the most recent monthly interest charge.",
        citations=[
            Citation(source_file="acme_lending_policy.md", section="4. Contract Terms", chunk=2)
        ],
        retrieval_seconds=0.0427,
    )
    app = app_module.create_app()

    with TestClient(app) as client:
        response = client.get("/ask", params={"question": "What is the break-cost cap?"})

    fake_rag.answer_question.assert_called_once_with("What is the break-cost cap?")
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Three times the most recent monthly interest charge.",
        "citations": [
            {"source_file": "acme_lending_policy.md", "section": "4. Contract Terms", "chunk": 2}
        ],
        "retrieval_seconds": 0.0427,
    }


def test_ask_serialises_an_unanswerable_question_as_json_null(
    rag_cls: MagicMock, fake_rag: MagicMock
):
    fake_rag.answer_question.return_value = AnswerResponse(
        answer=None, citations=[], retrieval_seconds=None
    )
    app = app_module.create_app()

    with TestClient(app) as client:
        response = client.get("/ask", params={"question": "Who won the 2050 World Cup?"})

    assert response.status_code == 200
    # `answer` must be a real JSON null, not the string "null".
    assert response.json() == {"answer": None, "citations": [], "retrieval_seconds": None}
    assert '"answer":null' in response.text.replace(" ", "")


def test_ask_returns_503_before_the_rag_is_ready(rag_cls: MagicMock):
    # No `with` block => lifespan doesn't run => app.state.rag is unset.
    client = TestClient(app_module.create_app())

    response = client.get("/ask", params={"question": "anything"})

    assert response.status_code == 503
    rag_cls.bootstrap.assert_not_called()


def test_ask_rejects_a_blank_question(rag_cls: MagicMock, fake_rag: MagicMock):
    app = app_module.create_app()

    with TestClient(app) as client:
        response = client.get("/ask", params={"question": ""})

    assert response.status_code == 422
    fake_rag.answer_question.assert_not_called()
