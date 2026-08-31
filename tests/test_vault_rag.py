"""Tests for the_vault/rag.py — the Rag store wrapper and its bootstrap.

The OpenAI + Chroma boundaries are patched out so ``Rag.bootstrap`` is
unit-testable without embeddings or a real collection.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from prometheus_client import generate_latest

from missions.the_vault import metrics, rag


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
    monkeypatch.setattr(rag, "EMBEDDING_CACHE_DIR", str(tmp_path / "embedding_cache"))
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


def test_build_embeddings_wraps_openai_in_a_disk_backed_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(rag, "EMBEDDING_CACHE_DIR", str(tmp_path / "embedding_cache"))
    monkeypatch.setattr(rag, "OpenAIEmbeddings", MagicMock(name="OpenAIEmbeddings"))

    embeddings = rag._build_embeddings()

    assert isinstance(embeddings, rag.CacheBackedEmbeddings)
    # Query embeddings are cached too, not just document embeddings.
    assert embeddings.query_embedding_store is not None


def test_bootstrap_uses_the_cached_embeddings_for_the_store(
    wired_bootstrap: tuple[MagicMock, MagicMock],
):
    chroma_cls, _ = wired_bootstrap
    chroma_cls.return_value = _fake_chroma(count=1)

    rag.Rag.bootstrap()

    embedding_function = chroma_cls.call_args.kwargs["embedding_function"]
    assert isinstance(embedding_function, rag.CacheBackedEmbeddings)


def test_vector_count_delegates_to_the_live_store():
    store = _fake_chroma(count=0)
    handle = rag.Rag(store)

    store._collection.count.return_value = 12

    assert handle.vector_count() == 12


class _StubRetriever(rag.BaseRetriever):
    """Returns a fixed document list, ignoring the query."""

    docs: list

    def _get_relevant_documents(self, query, *, run_manager):  # noqa: ARG002
        return self.docs


def test_capped_retriever_returns_only_the_first_k_documents():
    base = _StubRetriever(docs=[Document(page_content=str(i)) for i in range(10)])

    capped = rag.CappedRetriever(base=base, k=5)
    docs = capped.invoke("anything")

    assert [d.page_content for d in docs] == ["0", "1", "2", "3", "4"]


def test_capped_retriever_passes_through_when_fewer_than_k():
    base = _StubRetriever(docs=[Document(page_content="a"), Document(page_content="b")])

    docs = rag.CappedRetriever(base=base, k=5).invoke("anything")

    assert [d.page_content for d in docs] == ["a", "b"]


def test_hybrid_retriever_wraps_the_ensemble_in_a_capped_retriever(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rag, "FINAL_K", 5)
    fused = [Document(page_content=str(i)) for i in range(10)]
    monkeypatch.setattr(rag, "EnsembleRetriever", lambda **_: _StubRetriever(docs=fused))
    monkeypatch.setattr(rag.BM25Retriever, "from_documents", lambda *a, **k: MagicMock())

    store = MagicMock()
    store.get.return_value = {"documents": [], "metadatas": []}
    handle = rag.Rag(store)

    retriever = handle._hybrid_retriever()

    assert isinstance(retriever, rag.CappedRetriever)
    assert len(retriever.invoke("q")) == 5
    assert handle._hybrid_retriever() is retriever  # cached


def _metric(line_prefix: str) -> float:
    """Value of the first metrics line starting with ``line_prefix``, or 0.0."""
    for line in generate_latest(metrics.registry).decode().splitlines():
        if line.startswith(line_prefix):
            return float(line.rsplit(maxsplit=1)[1])
    return 0.0


def _hist_count(name: str) -> float:
    return _metric(f"{name}_count ")


def _llm_result(input_tokens: int, output_tokens: int) -> SimpleNamespace:
    """A stand-in LLMResult exposing usage the way langchain-core does."""
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    message = SimpleNamespace(usage_metadata=usage)
    return SimpleNamespace(generations=[[SimpleNamespace(message=message)]], llm_output=None)


def test_metrics_callback_times_only_the_outer_retriever_run():
    cb = rag.MetricsCallback()
    retrieval_before = _hist_count("vault_retrieval_duration_seconds")
    generation_before = _hist_count("vault_generation_duration_seconds")

    outer, dense, bm25, llm = uuid4(), uuid4(), uuid4(), uuid4()
    # EnsembleRetriever: an outer run wrapping the dense + BM25 child legs.
    cb.on_retriever_start({}, "q", run_id=outer)
    cb.on_retriever_start({}, "q", run_id=dense)
    cb.on_retriever_end(["d1", "d2", "d3"], run_id=dense)
    cb.on_retriever_start({}, "q", run_id=bm25)
    cb.on_retriever_end(["d1", "d2", "d3"], run_id=bm25)
    cb.on_retriever_end(["d1", "d2", "d3", "d4"], run_id=outer)
    # Structured output: a single chat-model run.
    cb.on_chat_model_start({}, [], run_id=llm)
    cb.on_llm_end(_llm_result(0, 0), run_id=llm)

    assert _hist_count("vault_retrieval_duration_seconds") - retrieval_before == 1
    assert _hist_count("vault_generation_duration_seconds") - generation_before == 1


def test_metrics_callback_records_chunk_count_of_the_outer_retriever_run_only():
    cb = rag.MetricsCallback()
    count_before = _hist_count("vault_retrieved_chunks")
    sum_before = _metric("vault_retrieved_chunks_sum ")

    outer, child = uuid4(), uuid4()
    cb.on_retriever_start({}, "q", run_id=outer)
    cb.on_retriever_start({}, "q", run_id=child)
    cb.on_retriever_end(["d"] * 5, run_id=child)  # child leg — not recorded
    cb.on_retriever_end(["d"] * 8, run_id=outer)  # fused result set

    assert _hist_count("vault_retrieved_chunks") - count_before == 1
    assert _metric("vault_retrieved_chunks_sum ") - sum_before == 8


def test_metrics_callback_ignores_end_events_for_untracked_runs():
    cb = rag.MetricsCallback()
    before = _hist_count("vault_generation_duration_seconds")

    cb.on_llm_end(_llm_result(0, 0), run_id=uuid4())

    assert _hist_count("vault_generation_duration_seconds") == before


def test_metrics_callback_records_token_usage_from_the_llm_result():
    cb = rag.MetricsCallback()
    prompt_before = _metric('vault_tokens_total{kind="prompt"} ')
    completion_before = _metric('vault_tokens_total{kind="completion"} ')

    llm = uuid4()
    cb.on_chat_model_start({}, [], run_id=llm)
    cb.on_llm_end(_llm_result(1200, 340), run_id=llm)

    assert _metric('vault_tokens_total{kind="prompt"} ') - prompt_before == 1200
    assert _metric('vault_tokens_total{kind="completion"} ') - completion_before == 340
