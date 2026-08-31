"""Policy + SLA lookup — a thin wrapper over the Mission 2 RAG service."""

from functools import lru_cache

import structlog

from missions.the_vault.rag import CitedAnswer, Rag

log = structlog.get_logger("co_pilot")

@lru_cache(maxsize=1)
def _rag() -> Rag:
    """Cached ``Rag`` handle, bootstrapped once on first lookup."""
    return Rag.bootstrap()


def lookup_policy(query: str) -> CitedAnswer:
    """Answer a free-text Acme policy / SLA question via the Vault.

    Returns a :class:`CitedAnswer`: ``answer`` is null when the corpus has
    nothing on the query, otherwise grounded text backed by ``citations``
    (each with ``source_file``, ``section`` and ``chunk``).
    """
    result = _rag().answer_question(query)
    return result
