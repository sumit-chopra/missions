# Glass Cockpit

A minimal terminal-based conversational LLM client.

## Current Status

The project is a terminal chat loop backed by an OpenAI model. It:

- Accepts user input from the terminal and sends it to the configured model
- Replays the last 10 turns as context, persisted in a local SQLite file
- Exits on `exit`, `quit`, `bye`, `Ctrl+C`, or `Ctrl+D`

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key
- Docker + Docker Compose (optional, only needed to run in a container)

## Setup

```bash
make setup
# or: uv sync
```

## Configuration

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

| Variable                   | Required | Default                      | Purpose                                   |
| -------------------------- | -------- | ---------------------------- | ----------------------------------------- |
| `OPENAI_API_KEY`           | yes      | —                            | Your OpenAI API key                       |
| `MODEL_NAME`               | no       | `gpt-5.4-nano`               | Chat model to use                        |
| `OPENAI_BASE_URL`          | no       | OpenAI's API                 | Override for a proxy / compatible gateway |
| `GLASS_COCKPIT_HISTORY_DB` | no       | `.missions/glass_cockpit.db` | SQLite file holding the last 10 turns     |

`.env` is gitignored and loaded automatically at startup.

## Running the app

```bash
make run
# or: uv run python chat.py
```

### Running with Docker

```bash
make docker
# or: docker compose run --rm chat
```

Use `docker compose run`, not `up` — `up` streams container logs but doesn't
forward your terminal's stdin into the container, so an interactive app like
this one won't see your input. `run` attaches your terminal properly.

## Conversation memory

Each exchange (one user message + the reply) is stored as a *turn* in a small
SQLite database — `.missions/glass_cockpit.db` by default, override with
`GLASS_COCKPIT_HISTORY_DB`. Before every request the most recent 10 turns are
replayed to the model as context, so the assistant remembers earlier messages
within and across sessions. Only the last 10 turns are kept; older rows are
pruned on write. A failed request records nothing. Delete the file to start
fresh; it is gitignored.

## Telemetry

After every LLM call the app reports usage twice:

- a human-readable line on **stdout**, right after the reply:
  `[stats] prompt=12 completion=3 cost=$0.000004 latency=204 ms model=gpt-5.4-mini`
- a one-line JSON object on **stderr**, so a session is newline-delimited JSON
  (JSONL) ready for `jq`:
  `{"model_name":"gpt-5.4-mini","prompt_tokens":12,"completion_tokens":3,"latency_ms":204,"cost_usd":0.000004}`

`cost_usd` comes from a built-in price table (`telemetry.py`). Unknown models report `0.0` rather than a guess.

### Inspecting metrics with `jq`

Drop the reply text (`>/dev/null`) and feed **stderr** into `jq`:

```bash
# live, one object per call as you chat
printf 'Hello\nHow are you\nexit\n' | make run 2> >(jq -c .) >/dev/null

```

The bash idiom `... 2>&1 >/dev/null | jq` does **not** work under zsh — its
`MULTIOS` option also tees stdout into the pipe, so `jq` chokes on the chat
banner. Use `2> >(jq …) >/dev/null` (works in bash and zsh) or the file form
above; `unsetopt multios` also restores the bash behaviour.

## Running tests

```bash
make test
# or: uv run pytest -q
```

Tests live under `tests/`: `test_chat.py` covers the terminal loop (with a fake
client, so no network calls), `test_llm_client.py` covers the OpenAI wrapper,
`test_store.py` covers the SQLite turn store, and `test_telemetry.py` covers the
cost/latency maths and the stats-line format. Each test gets an isolated history
database via an autouse fixture.

## Linting

```bash
make lint
# or: uv run ruff check . && uv run ruff format --check .
```

To automatically format code:

```bash
make format
# or: uv run ruff format .
```

## Project Structure

```text
.
├── .dockerignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── README.md
├── chat.py                 # entry-point shim: `python chat.py`
├── src/
│   └── missions/
│       ├── __init__.py
│       └── glass_cockpit/
│           ├── __init__.py
│           ├── chat.py
│           ├── llm_client.py
│           ├── store.py
│           └── telemetry.py
└── tests/
    ├── conftest.py
    ├── test_chat.py
    ├── test_llm_client.py
    ├── test_store.py
    └── test_telemetry.py
```
