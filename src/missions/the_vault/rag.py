"""The embedded, queryable corpus."""

import re
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import structlog
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.storage import LocalFileStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from missions.the_vault import metrics
from missions.the_vault.ingest import load_chunks

log = structlog.get_logger()

CHROMA_DIR = ".missions/vault_chroma"
# On-disk cache of embedding vectors
EMBEDDING_CACHE_DIR = ".missions/vault_embedding_cache"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.4-mini"

# Each leg (dense, BM25) fetches TOP_K; their reciprocal-rank fused union is then
# capped to FINAL_K chunks before the LLM to keep the prompt small.
TOP_K = 5
FINAL_K = 8

_BM25_TOKEN = re.compile(r"[a-z0-9]+")


def _bm25_tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumerics for BM25.

    BM25Retriever's default ``str.split`` leaves Markdown attached: a phrase like
    ``**Authority (AFCA)**`` tokenises to ``(AFCA)**``, so a query for ``AFCA``
    never matches it. Splitting on word characters fixes acronym, figure and
    clause-number lookups.
    """
    return _BM25_TOKEN.findall(text.lower())


class Citation(BaseModel):
    source_file: str = Field(description="The source file path used to answer the question.")
    section: str = Field(description="The specific section path or header used.")
    chunk: int = Field(description="The chunk index identifier used.")


class CitedAnswer(BaseModel):
    answer: str | None = Field(
        description=(
            "The final answer to the user query. Use the JSON literal null (never "
            'the string "null" or an empty string) when the retrieved passages do '
            "not contain the answer."
        )
    )
    citations: list[Citation] = Field(
        description=(
            "A distinct, ordered list of all unique source chunks used to assemble this answer."
        )
    )


class AskResponse(CitedAnswer):
    """A ``CitedAnswer`` plus how long context retrieval took, for the /ask route."""

    retrieval_seconds: float | None = Field(
        default=None,
        description=(
            "Wall time in seconds to retrieve context for this answer (hybrid dense "
            "+ BM25 fusion). Null if retrieval did not run or was not timed."
        ),
    )


SYSTEM_PROMPT = (
    "You are The Vault, a retrieval-augmented assistant. Answer the user's question "
    "using only the context passages retrieved from the vault and provided to you. "
    "Each passage carries metadata: its source file, its section path, and its chunk "
    "index.\n"
    "\n"
    "Rules:\n"
    "- Ground every claim in the passages; add nothing that is not in them.\n"
    "- The passages are the authoritative source. If they contradict what you believe "
    "to be true, or contradict common knowledge, follow the passages. Do not correct, "
    'second-guess, or "fix" them with outside knowledge.\n'
    "- When the question asks for a quote, a figure, a clause, a name, or exact "
    "wording, reproduce it verbatim from the passages. Do not paraphrase, round off, "
    "complete a familiar phrasing from memory, or normalise it.\n"
    "- If answering needs facts from two or more passages, combine them and cite each "
    "one you use.\n"
    "- Only when the passages genuinely do not contain the answer, set answer to "
    'the JSON literal null (not the string "null", not "none", not an empty '
    "string) and return an empty citations list. Never fall back to general "
    "knowledge."
)

DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[source: {source} | section: {section} | chunk: {chunk}]\n{page_content}"
)


class MetricsCallback(BaseCallbackHandler):
    """Time the retrieval and generation legs of one ``answer_question`` run.

    The hybrid retriever fans out to child retrievers (dense + BM25), so several
    ``on_retriever_*`` pairs fire per run; we time only the outermost one.
    """

    def __init__(self) -> None:
        self._retrieval_starts: dict[UUID, float] = {}
        self._generation_starts: dict[UUID, float] = {}
        # Wall time of the outermost retrieval span for this run, for callers that
        # want to report it (e.g. the /ask response) rather than only histogram it.
        self.retrieval_seconds: float | None = None

    def on_retriever_start(
        self, serialized: Any, query: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        # EnsembleRetriever emits a start for itself and one for each child leg
        # (dense + BM25). Time only the outermost run: if a retrieval is already
        # in flight, this start is a child — skip it.
        if not self._retrieval_starts:
            self._retrieval_starts[run_id] = time.perf_counter()

    def on_retriever_end(self, documents: Any, *, run_id: UUID, **_: Any) -> None:
        start_time = self._retrieval_starts.pop(run_id, None)
        if start_time is not None:
            self.retrieval_seconds = time.perf_counter() - start_time
            metrics.record_retrieval(self.retrieval_seconds)
            metrics.record_retrieved_chunks(len(documents))

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **_: Any) -> None:
        self._retrieval_starts.pop(run_id, None)

    def on_chat_model_start(
        self, serialized: Any, messages: Any, *, run_id: UUID, **_: Any
    ) -> None:
        self._generation_starts[run_id] = time.perf_counter()

    def on_llm_end(self, response: Any = None, *, run_id: UUID, **_: Any) -> None:
        start_time = self._generation_starts.pop(run_id, None)
        if start_time:
            duration = time.perf_counter() - start_time
            metrics.record_generation(duration)
            usage = _token_usage(response)
            if usage is not None:
                metrics.record_tokens(*usage)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **_: Any) -> None:
        self._generation_starts.pop(run_id, None)


def _token_usage(response: Any) -> tuple[int, int] | None:
    usage = response.generations[0][0].message.usage_metadata
    prompt_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    completion_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    return int(prompt_tokens), int(completion_tokens)


class CappedRetriever(BaseRetriever):
    """Return only the top ``k`` of another retriever's already-ranked results."""

    base: BaseRetriever
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        docs = self.base.invoke(query, config={"callbacks": run_manager.get_child()})
        return docs[: self.k]


def _build_embeddings() -> Embeddings:
    """OpenAI embeddings wrapped in an on-disk vector cache.

    Cached embeddings so a repeated question skips the OpenAI call on retrieval.
    """
    store = LocalFileStore(EMBEDDING_CACHE_DIR)
    return CacheBackedEmbeddings.from_bytes_store(
        OpenAIEmbeddings(model=EMBEDDING_MODEL),
        store,
        namespace=EMBEDDING_MODEL,
        query_embedding_cache=True,
        key_encoder="sha256",
    )


class Rag:
    """The app's handle on the retrieval corpus.

    Build it once at startup with :meth:`bootstrap`.
    """

    def __init__(self, vector_store: Chroma) -> None:
        self._vector_store = vector_store
        self._retriever = None

    @classmethod
    def bootstrap(cls) -> "Rag":
        """Load the persisted store, embedding the corpus only if the collection is empty."""
        persist_dir = Path(CHROMA_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)
        embeddings = _build_embeddings()

        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        if vector_store._collection.count() > 0:
            log.info(
                "vector store already populated, skipping ingest",
                persist_dir=str(persist_dir),
                vectors=vector_store._collection.count(),
            )
            return cls(vector_store)

        log.info("vector store empty, ingesting corpus", persist_dir=str(persist_dir))
        vector_store.add_documents(load_chunks())
        log.info("ingest complete", vectors=vector_store._collection.count())
        return cls(vector_store)

    @property
    def vector_store(self) -> Chroma:
        """The underlying LangChain store.

        Exposed for the ask pipeline until retrieval moves onto this class.
        """
        return self._vector_store

    def vector_count(self) -> int:
        """Number of embedded chunks (one vector each) currently in the store."""
        return self._vector_store._collection.count()

    def _stored_documents(self) -> list[Document]:
        """Every chunk currently in the store, rebuilt from its persisted text + metadata."""
        raw = self._vector_store.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(raw["documents"], raw["metadatas"], strict=False)
        ]

    def _hybrid_retriever(self) -> BaseRetriever:
        """Dense + BM25 retrieval, fused by reciprocal rank and capped to FINAL_K. Cached."""
        if self._retriever is None:
            dense = self._vector_store.as_retriever(search_kwargs={"k": TOP_K})
            keyword = BM25Retriever.from_documents(
                self._stored_documents(), preprocess_func=_bm25_tokenize
            )
            keyword.k = TOP_K
            ensemble = EnsembleRetriever(retrievers=[dense, keyword])
            self._retriever = CappedRetriever(base=ensemble, k=FINAL_K)
        return self._retriever

    def answer_question(self, question: str) -> AskResponse:
        """Retrieve context for ``question`` and answer it with grounded citations."""
        llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        structured_llm = llm.with_structured_output(CitedAnswer)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "Context passages:\n\n{context}\n\nQuestion: {input}"),
            ]
        )

        # structured_llm already yields a CitedAnswer, so pass it straight through
        # instead of the default StrOutputParser (which only accepts strings).
        question_answer_chain = create_stuff_documents_chain(
            structured_llm,
            prompt,
            output_parser=RunnablePassthrough(),
            document_prompt=DOCUMENT_PROMPT,
        )

        rag_chain = create_retrieval_chain(self._hybrid_retriever(), question_answer_chain)

        callback = MetricsCallback()
        response = rag_chain.invoke({"input": question}, config={"callbacks": [callback]})

        for position, document in enumerate(response["context"], start=1):
            log.info(
                "retrieved chunk",
                position=position,
                source=document.metadata.get("source"),
                section=document.metadata.get("section"),
                chunk=document.metadata.get("chunk"),
                preview=document.page_content[:200],
            )

        result: CitedAnswer = response["answer"]
        log.info(
            "answered question",
            question=question,
            citations=len(result.citations),
            retrieval_seconds=callback.retrieval_seconds,
        )
        return AskResponse(
            answer=result.answer,
            citations=result.citations,
            retrieval_seconds=callback.retrieval_seconds,
        )
