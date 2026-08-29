"""FastAPI application for The Vault."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """The Vault FastAPI app."""
    app = FastAPI(title="The Vault", version="0.1.0")

    return app
