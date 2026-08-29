"""FastAPI application for The Vault.

On startup the corpus is chunked and embedded into a persisted Chroma collection
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from missions.the_vault.ingest import get_vector_store
from missions.the_vault.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    load_dotenv()
    app.state.vector_store = get_vector_store()
    yield


def create_app() -> FastAPI:
    """The Vault FastAPI app."""
    app = FastAPI(title="The Vault", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, object]:
        store = getattr(app.state, "vector_store", None)
        return {
            "status": "ok" if store is not None else "starting",
            "vectors": store._collection.count() if store is not None else 0,
        }

    return app
