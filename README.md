# Missions

Three LLM missions in one repo, one dependency tree, one `docker compose up`:

| Mission | Name                | What it is                                                          | Entry                            |
| ------- | ------------------- | ------------------------------------------------------------------ | -------------------------------- |
| 1       | **Glass Cockpit**   | terminal chat loop with SQLite memory + per-call cost/latency telemetry | `make run-chat`                  |
| 2       | **The Vault**       | FastAPI RAG service over a private corpus; hybrid retrieval, inline citations, Prometheus `/metrics` | `make run-vault` → `:8000`       |
| 3       | **The Ops Co-pilot** | OpenAI Agents SDK agent that turns an ops request into a validated action plan or a structured refusal | `make run-copilot`              |

Mission 3's `policy_lookup` tool delegates to Mission 2's retrieval service.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key
- Docker + Docker Compose (optional, only needed to run in a container)

## Quickstart

```bash
cp .env.example .env      # then put your OPENAI_API_KEY in it
make setup                # uv sync --all-extras
make test                 # 137 tests, mocked — no key needed
make run                  # docker compose up: vault + prometheus + copilot
```

| Target       | What it runs                                                            |
| ------------ | --------------------------------------------------------------------- |
| `make setup` | `uv sync --all-extras` — every extra + dev tools into `.venv`        |
| `make test`  | `uv run pytest -q` — unit suite, fully mocked, no network, no key     |
| `make eval`  | grades The Vault (raw vs. rag + citation recall); needs `OPENAI_API_KEY` |
| `make run`   | `docker compose up --build` — the whole stack                        |

Bare `make` lists these plus helpers: `eval-latency` (warm vs. cold retrieval
timing), `run-chat`, `run-vault`, `run-copilot`, `docker-chat`, `docker-copilot`,
`lint`, `format`, `down`, `logs`.

## Configuration

`.env` is gitignored and loaded at startup (`cp .env.example .env`). Only
`OPENAI_API_KEY` is required.

| Variable                   | Default                      | Purpose                              |
| -------------------------- | ---------------------------- | ----------------------------------- |
| `OPENAI_API_KEY`           | —                            | **required**                         |
| `OPENAI_BASE_URL`          | OpenAI's API                 | proxy / compatible gateway override  |

## `docker compose up`

```bash
make run          # == docker compose up --build
```

| Service      | Port   | What it is                                                       |
| ------------ | ------ | ------------------------------------------------------------- |
| `vault`      | `8000` | The Vault RAG service (Mission 2); `/ask`, `/health`, `/metrics` |
| `prometheus` | `9090` | scrapes `vault:8000/metrics` every 10s — <http://localhost:9090> |
| `copilot`    | —      | Ops Co-pilot (Mission 3); runs the demo request once, prints the plan, exits |

Without a valid key `vault` and `copilot` exit on startup (both embed the corpus
on first boot); `prometheus` still comes up. Embeddings persist in the
`vault-data` / `copilot-data` volumes, so a second `up` skips re-embedding.
`make down` stops everything and drops the volumes.

**Chat (Mission 1) is not in `docker compose up`.** It's an interactive REPL, and
`up` streams container logs without forwarding your terminal's stdin — the prompt
would just hang. It sits behind a Compose profile so `up` skips it. Run it with a
TTY attached:

```bash
make docker-chat   # docker compose run --build --rm chat
make run-chat      # or locally, no Docker
```

**Co-pilot with a custom prompt.** `copilot` runs one fixed request under `up`.
To send your own:

```bash
make docker-copilot PROMPT="Draft a follow-up plan for application #A-9999"
make run-copilot    PROMPT="..."     # or locally, no Docker
```

### Run one mission locally (no Docker)

```bash
make run-chat      # Mission 1 REPL
make run-vault     # Mission 2 on http://localhost:8000  (--reload)
make run-copilot   # Mission 3 against PROMPT (default, or PROMPT="...")
```

## The missions

### Mission 1 — Glass Cockpit

A terminal chat loop over an OpenAI chat model. Each exchange is saved as a
*turn* in a local SQLite file and the last 10 turns are replayed as context on
every request, so it remembers across sessions. After each reply it prints a
token / cost / latency stats line to stdout and the same object as JSON to
stderr. Entry: `make run-chat`.

### Mission 2 — The Vault

A FastAPI RAG service over a private Markdown corpus. Entry: `make run-vault` → `:8000`. `GET /health` returns
`{"status": "ok", "vectors": N}` (`"starting"` until ingestion finishes);
`make eval` grades raw-vs-rag answers + citation recall (needs a key). Latest
run: **raw 5/32, RAG 30/32** — full table under **Eval results** below.

**`GET /ask?question=…`** — one query param, `question`, 1–2000 chars (no request
body, ). Response:

```jsonc
{
  "answer": "Acme offers a personal loan from $5,000 to $50,000.",
  "citations": [
    {
      "source_file": "acme_product_info.md",
      "section": "Personal Loan — Product Information > 1. Loan Details",
      "chunk": 0
    }
  ],
  "retrieval_seconds": 0.0043
}
```

`retrieval_seconds` is the wall time of the retrieval leg only — the LLM
synthesis call is excluded. To see the benefit of cache, hit the api and observe `retrieval_seconds` and then reload the same question and observe the difference.

**`GET /metrics`** — Prometheus text format on a dedicated registry (only
`vault_*`, `/ask` is the only
instrumented route: `vault_ask_requests_total`,
`vault_ask_request_duration_seconds`, `vault_ask_total{outcome}` (`answered` /
`declined` / `error`), `vault_retrieval_duration_seconds`,
`vault_retrieved_chunks`, `vault_generation_duration_seconds`,
`vault_tokens_total{kind}` (`prompt` / `completion`).

**Retrieval duration.** A `MetricsCallback` times the retrieval leg and the
generation leg of each `/ask` separately (retrieval is surfaced as
`retrieval_seconds` on the response and histogrammed as
`vault_retrieval_duration_seconds`). The dense HNSW search and BM25 scan are
local and cheap; the bulk of retrieval time is the one OpenAI round-trip to
embed the question. Embeddings go through `CacheBackedEmbeddings` over an
on-disk store (`.missions/vault_embedding_cache/`, sha256 key,
`query_embedding_cache=True`), so **asking the same question again is served from
cache and skips the embed call** — retrieval then collapses to just the local
HNSW + BM25 work. The cache lives in the `vault-data` volume, so repeats stay
fast across restarts.

**Measured** (`make eval-latency`, 2,422 vectors, 32 eval questions, Apple
Silicon, local Chroma; retriever leg only, LLM excluded):

| pass | p50 | p90 | max |
| ---- | --- | --- | --- |
| cold — embedding not cached, one OpenAI round-trip | 122 ms | 181 ms | 702 ms |
| warm — embedding cache hit, local HNSW + BM25 only | 2.7 ms | 4.7 ms | 10 ms |

The warm path is the steady state for a repeated ops question and is ~40× faster:
the embedding round-trip is the whole cold cost, and the cache removes it. The
`eval-latency` script points the cache at a throwaway directory so the cold pass
is genuinely cold; the live equivalent is the `vault_retrieval_duration_seconds`
histogram on `/metrics`.

**Eval results.** `make eval` runs every `eval/eval.json` question twice — once
against the raw chat model with no retrieval, once through the RAG pipeline — and
an LLM judge scores each answer 0/1 against the reference. Answerer and judge are
both `temperature=0`; scores move by ±1–2 run to run on the borderline
paraphrase/cross-document rows.

```
question                                      raw  rag
--------------------------------------------  ---  ---
What is the maximum break-cost a customer c…    0    1
What minimum Equifax comprehensive credit s…    0    1
What interest-rate sensitivity buffer does …    0    1
According to the RG 209 working synthesis, …    0    1
Per the RG 209 synthesis, which Schedule of…    0    1
Where was the One Ring forged?                  0    1
To whom was the third Silmaril entrusted?       0    1
In what year, and in what place, was the oa…    0    1
Quote exactly what Gandalf says in the Silm…    0    1
Acme must acknowledge a hardship notice and…    0    1
A borrower cancels an approved loan within …    0    0
What range of loan amounts does Acme offer …    0    1
What loan terms, in years, can an Acme pers…    0    1
What is Acme's indicative interest rate ran…    0    1
What establishment fee does Acme charge on …    0    1
What is Acme's monthly account fee on the p…    0    0
What fee does Acme charge for a dishonoured…    0    1
How long does Acme have to complete and com…    0    1
What is the maximum length of a repayment p…    0    1
What are Acme Operations' customer support …    0    1
What is Acme's SLA for completing verificat…    0    1
After how many days of an unresolved compla…    1    1
What does AFCA stand for?                       1    1
Which Australian regulator's guidance is RG…    1    1
In Acme's Personal Lending Policy, the brea…    0    1
What minimum buffer does Acme apply to a cu…    0    1
Under the Operations SLA Handbook, what hap…    0    1
Under Acme's Financial Hardship Policy comm…    0    1
Per the RG 209 working synthesis, how long …    1    1
Which provision of the National Consumer Cr…    0    1
Per the RG 209 synthesis, under which secti…    0    1
What is Acme's current variable interest ra…    1    1
--------------------------------------------  ---  ---
total (of 32)                                   5   30


flavour                           n  rag
------------------------------  ---  ---
clause-reference                  4    4
common-knowledge                  2    2
contradicts-common-knowledge      3    3
cross-document                    2    1
paraphrase-vs-quote               1    1
specific-figure                  19   18
unanswerable                      1    1
------------------------------  ---  ---
total                            32   30
```

**Raw 5/32 → RAG 30/32.** The raw model only gets the five questions answerable
from general knowledge (what AFCA stands for, ASIC issuing RG 209, the
`unanswerable` home-loan-rate question it correctly declines) and misses every
corpus-specific figure, clause and quote. Retrieval closes that gap. The two RAG
misses are both multi-fact questions where the judge wants every number present:
the cooling-off + complaint-acknowledgement cross-document question and the
monthly-account-fee row (retrieved, but the answer omitted the "charged in
arrears" detail the reference carries). Citation source accuracy is 0.91;
section accuracy is lower (0.72) because a fact often sits on a header boundary
and the retrieved chunk is filed under the adjacent section.

### Mission 3 — The Ops Co-pilot

An agent (OpenAI Agents SDK) that turns a free-text ops request into a
Pydantic-validated action plan, or a clean structured refusal when no honest
plan is possible. It calls mock CRM / calendar tools plus a `policy_lookup` tool
that delegates to Mission 2's retrieval service. The plan or refusal is printed
as JSON to stdout; the model's reasoning and every tool call are traced to
stderr with PII masking. Exit code: `0` plan, `3` refusal, `1` error. Entry:
`make run-copilot`. 
