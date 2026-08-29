"""Tests for the_vault/app.py — lifespan wiring and the /health endpoint."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from missions.the_vault import app as app_module


@pytest.fixture
def fake_store() -> MagicMock:
    store = MagicMock(name="vector_store")
    store._collection.count.return_value = 7
    return store


@pytest.fixture
def get_vector_store(monkeypatch: pytest.MonkeyPatch, fake_store: MagicMock) -> MagicMock:
    """Patch the ingest entrypoint the lifespan calls; yield the mock."""
    mock = MagicMock(name="get_vector_store", return_value=fake_store)
    monkeypatch.setattr(app_module, "get_vector_store", mock)
    return mock


def test_lifespan_builds_and_stores_the_vector_store(
    get_vector_store: MagicMock, fake_store: MagicMock
):
    app = app_module.create_app()

    with TestClient(app):
        get_vector_store.assert_called_once_with()
        assert app.state.vector_store is fake_store


def test_health_reports_ok_and_vector_count_after_startup(get_vector_store: MagicMock):
    app = app_module.create_app()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "vectors": 7}


def test_health_reports_starting_before_the_store_is_ready(get_vector_store: MagicMock):
    # No `with` block => lifespan doesn't run => app.state.vector_store is unset.
    client = TestClient(app_module.create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "starting", "vectors": 0}
    get_vector_store.assert_not_called()
