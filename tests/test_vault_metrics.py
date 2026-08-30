"""Tests for the_vault/metrics.py — the /metrics endpoint and its collectors."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from missions.the_vault import app as app_module
from missions.the_vault.rag import Citation, CitedAnswer


@pytest.fixture
def fake_rag() -> MagicMock:
    rag = MagicMock(name="rag")
    rag.vector_count.return_value = 7
    rag.answer_question.return_value = CitedAnswer(answer="ok", citations=[])
    return rag


@pytest.fixture
def rag_cls(monkeypatch: pytest.MonkeyPatch, fake_rag: MagicMock) -> MagicMock:
    """Patch the Rag class the lifespan bootstraps; yield the mock."""
    mock = MagicMock(name="Rag")
    mock.bootstrap.return_value = fake_rag
    monkeypatch.setattr(app_module, "Rag", mock)
    return mock


def _sample(body: str, name: str, **labels: str) -> float:
    """Value of the first sample called ``name`` whose labels match, else 0.0."""
    for family in text_string_to_metric_families(body):
        for s in family.samples:
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                return s.value
    return 0.0


def _scrape(client: TestClient) -> str:
    return client.get("/metrics").text


def test_metrics_endpoint_serves_prometheus_text_format(rag_cls: MagicMock):
    with TestClient(app_module.create_app()) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "version=" in response.headers["content-type"]
    body = response.text
    assert "# HELP vault_ask_requests_total" in body
    assert "# TYPE vault_ask_request_duration_seconds histogram" in body


def test_ask_requests_are_counted_by_method_and_status(rag_cls: MagicMock):
    with TestClient(app_module.create_app()) as client:
        before = _sample(_scrape(client), "vault_ask_requests_total", method="GET", status="200")
        client.get("/ask", params={"question": "anything"})
        after = _sample(_scrape(client), "vault_ask_requests_total", method="GET", status="200")

    assert after - before == 1


def test_only_ask_is_instrumented_at_the_http_layer(rag_cls: MagicMock):
    with TestClient(app_module.create_app()) as client:
        before = _sample(_scrape(client), "vault_ask_request_duration_seconds_count", method="GET")
        client.get("/health")
        client.get("/metrics")
        client.get("/no-such-route")
        after = _sample(_scrape(client), "vault_ask_request_duration_seconds_count", method="GET")

    assert after == before


def test_ask_answered_increments_the_answered_outcome(rag_cls: MagicMock, fake_rag: MagicMock):
    fake_rag.answer_question.return_value = CitedAnswer(
        answer="Three times the most recent monthly interest charge.",
        citations=[
            Citation(source_file="acme_lending_policy.md", section="4. Contract Terms", chunk=2)
        ],
    )
    with TestClient(app_module.create_app()) as client:
        before = _sample(_scrape(client), "vault_ask_total", outcome="answered")
        client.get("/ask", params={"question": "What is the break-cost cap?"})
        body = _scrape(client)

    assert _sample(body, "vault_ask_total", outcome="answered") - before == 1


def test_ask_declined_answer_counts_as_declined(rag_cls: MagicMock, fake_rag: MagicMock):
    fake_rag.answer_question.return_value = CitedAnswer(answer=None, citations=[])
    with TestClient(app_module.create_app()) as client:
        before = _sample(_scrape(client), "vault_ask_total", outcome="declined")
        client.get("/ask", params={"question": "Who won the 2050 World Cup?"})
        body = _scrape(client)

    assert _sample(body, "vault_ask_total", outcome="declined") - before == 1


def test_ask_failure_counts_as_error_outcome(rag_cls: MagicMock, fake_rag: MagicMock):
    fake_rag.answer_question.side_effect = RuntimeError("retrieval exploded")
    app = app_module.create_app()

    with TestClient(app, raise_server_exceptions=False) as client:
        before = _sample(_scrape(client), "vault_ask_total", outcome="error")
        response = client.get("/ask", params={"question": "anything"})
        body = _scrape(client)

    assert response.status_code == 500
    assert _sample(body, "vault_ask_total", outcome="error") - before == 1
    assert _sample(body, "vault_ask_requests_total", method="GET", status="500") >= 1
