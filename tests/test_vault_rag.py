"""Tests for the_vault/rag.py — the Rag store wrapper and its bootstrap.

The OpenAI + Chroma boundaries are patched out so ``Rag.bootstrap`` is
unit-testable without embeddings or a real collection.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from missions.the_vault import rag


def _fake_chroma(count: int) -> MagicMock:
    store = MagicMock()
    store._collection.count.return_value = count
    return store


@pytest.fixture
def wired_bootstrap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[MagicMock, MagicMock]:
    """Patch the embedding + Chroma + chunking boundaries.

    Returns ``(chroma_cls, load_chunks)`` mocks for the test to configure.
    """
    monkeypatch.setattr(rag, "CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(rag, "OpenAIEmbeddings", MagicMock(name="OpenAIEmbeddings"))
    chroma_cls = MagicMock(name="Chroma")
    monkeypatch.setattr(rag, "Chroma", chroma_cls)
    load_chunks = MagicMock(name="load_chunks", return_value=[Document(page_content="x")])
    monkeypatch.setattr(rag, "load_chunks", load_chunks)
    return chroma_cls, load_chunks


def test_bootstrap_ingests_when_collection_empty(
    wired_bootstrap: tuple[MagicMock, MagicMock], tmp_path: Path
):
    chroma_cls, load_chunks = wired_bootstrap
    store = _fake_chroma(count=0)
    chroma_cls.return_value = store

    result = rag.Rag.bootstrap()

    load_chunks.assert_called_once_with()
    store.add_documents.assert_called_once_with(load_chunks.return_value)
    assert result.vector_store is store
    assert (tmp_path / "chroma").is_dir()


def test_bootstrap_skips_ingest_when_already_populated(
    wired_bootstrap: tuple[MagicMock, MagicMock],
):
    chroma_cls, load_chunks = wired_bootstrap
    store = _fake_chroma(count=42)
    chroma_cls.return_value = store

    result = rag.Rag.bootstrap()

    load_chunks.assert_not_called()
    store.add_documents.assert_not_called()
    assert result.vector_store is store


def test_vector_count_delegates_to_the_live_store():
    store = _fake_chroma(count=0)
    handle = rag.Rag(store)

    store._collection.count.return_value = 12

    assert handle.vector_count() == 12
