"""Prometheus instrumentation for The Vault.

One module owns every collector, the ASGI middleware that feeds the HTTP ones,
and :func:`render` for the ``/metrics`` route. Collectors live on a dedicated
:data:`registry`, not the default one, so a scrape carries only ``vault_*`` —
none of prometheus_client's process/GC/platform gauges.
"""

import time

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

registry = CollectorRegistry()

# Only /ask is instrumented at the HTTP layer; /health and /metrics are noise for
# a scrape and would otherwise dominate the counters.
_INSTRUMENTED_PATH = "/ask"

HTTP_REQUESTS = Counter(
    "vault_ask_requests_total",
    "HTTP requests to /ask, by method and response status code.",
    ["method", "status"],
    registry=registry,
)
HTTP_LATENCY = Histogram(
    "vault_ask_request_duration_seconds",
    "HTTP request latency in seconds for /ask, by method.",
    ["method"],
    registry=registry,
)

ASK_TOTAL = Counter(
    "vault_ask_total",
    "/ask calls by outcome: answered, declined (corpus lacks the answer) or error.",
    ["outcome"],
    registry=registry,
)
RETRIEVAL_LATENCY = Histogram(
    "vault_retrieval_duration_seconds",
    "Time to retrieve context for an /ask call (hybrid dense + BM25 fusion).",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=registry,
)
GENERATION_LATENCY = Histogram(
    "vault_generation_duration_seconds",
    "Time in the chat model generating the answer for an /ask call.",
    buckets=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
    registry=registry,
)
RETRIEVED_CHUNKS = Histogram(
    "vault_retrieved_chunks",
    "Chunks returned by hybrid retrieval for an /ask call (0 = nothing found).",
    buckets=(0, 1, 2, 3, 5, 8, 10, 15, 20),
    registry=registry,
)
TOKENS = Counter(
    "vault_tokens_total",
    "LLM tokens consumed by /ask answer generation, by kind.",
    ["kind"],  # prompt | completion
    registry=registry,
)


def record_ask(outcome: str) -> None:
    """Count one /ask call"""
    ASK_TOTAL.labels(outcome=outcome).inc()


def record_tokens(prompt: int, completion: int) -> None:
    """Add one generation call's prompt/completion token counts."""
    TOKENS.labels(kind="prompt").inc(prompt)
    TOKENS.labels(kind="completion").inc(completion)


def record_retrieval(seconds: float) -> None:
    """Record the wall time of one context-retrieval span."""
    RETRIEVAL_LATENCY.observe(seconds)


def record_generation(seconds: float) -> None:
    """Record the wall time of one answer-generation (chat model) span."""
    GENERATION_LATENCY.observe(seconds)


def record_retrieved_chunks(count: int) -> None:
    """Record how many chunks one retrieval span returned."""
    RETRIEVED_CHUNKS.observe(count)


def render() -> Response:
    """The ``/metrics`` payload in Prometheus text exposition format."""
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Count and time requests to /ask, labelled by method and status."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path != _INSTRUMENTED_PATH:
            return await call_next(request)

        method = request.method
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            _observe(method, "500", start)
            raise
        _observe(method, str(response.status_code), start)
        return response


def _observe(method: str, status: str, start: float) -> None:
    HTTP_LATENCY.labels(method=method).observe(time.perf_counter() - start)
    HTTP_REQUESTS.labels(method=method, status=status).inc()
