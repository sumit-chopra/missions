"""Measure The Vault's retrieval latency — hybrid dense + BM25, LLM excluded.

    uv run python src/missions/the_vault/eval/latency.py
    make eval-latency

Times only the retrieval leg of ``/ask`` (dense HNSW search + BM25 scan +
reciprocal-rank fusion), the number the brief asks to keep low on a warm cache.
The LLM synthesis call is never made here.

The corpus stays in the persisted Chroma store, but the embedding cache is
redirected to a throwaway directory so the run is reproducible regardless of what
earlier ``make eval`` runs left cached. Two passes over the ``eval.json``
questions:

* **cold** — each question asked once against the empty cache, so retrieval pays
  one ``text-embedding-3-small`` round-trip.
* **warm** — the same questions again (``WARM_REPEATS`` times); the embedding is
  now served from ``CacheBackedEmbeddings``' sha256-keyed store, so retrieval is
  just the local HNSW + BM25 work.

Reported as p50 / p90 / max in milliseconds. Cost: ~one embed call per question
on the cold pass (fractions of a cent); the warm pass makes no API calls.
"""

import json
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from missions.the_vault import rag as rag_module
from missions.the_vault.rag import Rag

EVAL_PATH = Path(__file__).parent / "eval.json"
WARM_REPEATS = 3


def _percentile(sorted_samples: list[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    index = min(len(sorted_samples) - 1, int(q * len(sorted_samples)))
    return sorted_samples[index]


def _summary(samples: list[float]) -> str:
    ordered = sorted(samples)
    return (
        f"p50={_percentile(ordered, 0.50) * 1000:7.1f} ms   "
        f"p90={_percentile(ordered, 0.90) * 1000:7.1f} ms   "
        f"max={max(ordered) * 1000:7.1f} ms   (n={len(samples)})"
    )


def _time_pass(retriever, questions: list[str], repeats: int = 1) -> list[float]:
    samples: list[float] = []
    for _ in range(repeats):
        for question in questions:
            start = time.perf_counter()
            retriever.invoke(question)
            samples.append(time.perf_counter() - start)
    return samples


def main() -> int:
    load_dotenv()
    # Redirect the query-embedding cache to a throwaway dir so "cold" is really
    # cold. The corpus vectors live in Chroma and are untouched by this.
    tmp_cache = tempfile.mkdtemp(prefix="vault_latency_cache_")
    rag_module.EMBEDDING_CACHE_DIR = tmp_cache

    rag = Rag.bootstrap()
    # Build the BM25 index once here, off the clock — it is a one-time cost, not
    # part of per-query retrieval latency.
    retriever = rag._hybrid_retriever()
    records = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["questions"]
    questions = [record["question"] for record in records]

    cold = _time_pass(retriever, questions)
    warm = _time_pass(retriever, questions, repeats=WARM_REPEATS)

    print(f"\nvectors={rag.vector_count()}  questions={len(questions)}")
    print(f"cold (uncached embed):  {_summary(cold)}")
    print(f"warm (embed cache hit): {_summary(warm)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
