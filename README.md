# Glass Cockpit

A minimal terminal-based conversational LLM client.

## Current Status

The project is a terminal chat loop backed by an OpenAI model. It:

- Accepts user input from the terminal and sends it to the configured model
- Each message is a one-shot prompt (no conversation history)
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

| Variable          | Required | Default       | Purpose                                  |
| ----------------- | -------- | ------------- | ---------------------------------------- |
| `OPENAI_API_KEY`  | yes      | —             | Your OpenAI API key                      |
| `MODEL_NAME`      | no       | `gpt-4o-mini` | Chat model to use                        |
| `OPENAI_BASE_URL` | no       | OpenAI's API  | Override for a proxy / compatible gateway |

`.env` is gitignored and loaded automatically at startup.

## Running the app

```bash
make run
# or: uv run python src/missions/glass_cockpit/chat.py
```

### Running with Docker

```bash
make docker
# or: docker compose run --rm chat
```

Use `docker compose run`, not `up` — `up` streams container logs but doesn't
forward your terminal's stdin into the container, so an interactive app like
this one won't see your input. `run` attaches your terminal properly.

## Running tests

```bash
make test
# or: uv run pytest -q
```

Tests live under `tests/`: `test_chat.py` covers the terminal loop (with a fake
client, so no network calls) and `test_llm_client.py` covers the OpenAI wrapper.

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
├── src/
│   └── missions/
│       ├── __init__.py
│       └── glass_cockpit/
│           ├── __init__.py
│           ├── chat.py
│           └── llm_client.py
└── tests/
    ├── conftest.py
    ├── test_chat.py
    └── test_llm_client.py
```
