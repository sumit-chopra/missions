"""FastAPI application for The Vault.

On startup the corpus is chunked and embedded into a persisted Chroma collection.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

from missions.the_vault.logging import setup_logging
from missions.the_vault.rag import CitedAnswer, Rag


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    load_dotenv()
    app.state.rag = Rag.bootstrap()
    yield


def create_app() -> FastAPI:
    """The Vault FastAPI app."""
    app = FastAPI(title="The Vault", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, object]:
        rag = getattr(app.state, "rag", None)
        return {
            "status": "ok" if rag is not None else "starting",
            "vectors": rag.vector_count() if rag is not None else 0,
        }

    @app.get("/ask", response_model=CitedAnswer)
    def ask(question: Annotated[str, Query(min_length=1, max_length=2000)]) -> CitedAnswer:
        """Answer a question against the embedded corpus with inline citations."""
        rag = getattr(app.state, "rag", None)
        if rag is None:
            raise HTTPException(status_code=503, detail="vector store is not ready yet")
        return rag.answer_question(question)

    return app
